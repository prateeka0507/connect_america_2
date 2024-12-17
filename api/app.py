import os
import uuid
import re
import logging
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from docx import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document as LangchainDocument, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langsmith import traceable
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import base64
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import time
import json
import numpy as np
from typing import List
from pydantic import BaseModel, Field
import sys
import traceback
from datetime import datetime

class LLMResponseError(Exception):
    pass

class LLMResponseCutOff(LLMResponseError):
    pass


class LLMNoResponseError(LLMResponseError):
    pass

load_dotenv()

# Database configuration
NEON_DB_URL = os.getenv("POSTGRES_URL")

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://connect-america-2-frontend.vercel.app",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# System instructions
SYSTEM_INSTRUCTIONS = """You are an AI assistant for Connect America's internal support team. Your role is to:
1. Analyze support documents and provide clear, professional responses
2. Convert technical content into easy-to-understand explanations
3. Focus on explaining processes and solutions rather than quoting directly
4. Maintain a professional, helpful tone
5. If information isn't available in the provided context, clearly state that
6. Always respond in English, regardless of the input language

Response Structure and Formatting:
   - Use markdown formatting with clear hierarchical structure
   - Each major section must start with '### ' followed by a number and bold title
   - Format section headers as: ### 1. **Title Here**
   - Use bullet points (-) for detailed explanations
   - Keep responses clear, practical, and focused on support topics

Remember:
- Focus on analyzing the support documentation and explaining concepts naturally
- Keep responses clear, practical, and focused on internal support expertise
"""
app.secret_key = os.urandom(24)  # Set a secret key for sessions

# Access your API keys (set these in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "connect-america"

# Initialize Langchain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-2024-11-20", temperature=0)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = app.logger

def rewrite_query(query, chat_history=None):
    """
    Rewrites the user query to be more specific and searchable using LLM.
    """
    try:
        rewrite_prompt = f"""You are Connect America's internal support assistant. Rewrite this query to be more specific and searchable, taking into account the chat history if provided. Only return the rewritten query without explanations.

        Original query: {query}
        
        Chat history: {json.dumps(chat_history) if chat_history else '[]'}
        
        Rewritten query:"""
        
        response = llm.predict(rewrite_prompt)
        cleaned_response = response.replace("Rewritten query:", "").strip()
        
        logging.debug(f"Original query: {query}")
        logging.debug(f"Rewritten query: {cleaned_response}")
        
        return cleaned_response if cleaned_response else query
        
    except Exception as e:
        logging.error(f"Error in query rewriting: {str(e)}", exc_info=True)
        return query

def verify_database():
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            count = cur.fetchone()['count']
            logging.info(f"Total documents in database: {count}")
            
            cur.execute("SELECT title FROM documents LIMIT 5")
            sample_titles = [row['title'] for row in cur.fetchall()]
            logging.info(f"Sample document titles: {sample_titles}")
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Database verification failed: {str(e)}", exc_info=True)
        return False

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_metadata_from_text(text):
    title = text.split('\n')[0] if text else "Untitled Video"
    return {"title": title}

def upsert_transcript(transcript_text, metadata, index_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(transcript_text)
    
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = f"{metadata['title']}_chunk_{i}"
                chunk_metadata['url'] = metadata.get('url', '')
                chunk_metadata['title'] = metadata.get('title', 'Unknown Video')
                
                # Generate embeddings for the chunk
                chunk_embedding = embeddings.embed_query(chunk)
                
                # Insert into bents table
                cur.execute("""
                    INSERT INTO bents (text, title, url, chunk_id, vector)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET text = EXCLUDED.text, vector = EXCLUDED.vector
                """, (chunk, chunk_metadata['title'], chunk_metadata['url'], 
                      chunk_metadata['chunk_id'], str(chunk_embedding)))
        conn.commit()
    except Exception as e:
        logging.error(f"Error upserting transcript: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(LLMResponseError))
def retry_llm_call(qa_chain, query, chat_history):
    try:
        result = qa_chain({"question": query, "chat_history": chat_history})
        
        if result is None or 'answer' not in result or not result['answer']:
            raise LLMNoResponseError("LLM failed to generate a response")
        
        if result['answer'].endswith('...') or len(result['answer']) < 20:
            raise LLMResponseCutOff("LLM response appears to be cut off")
        return result
    except Exception as e:
        if isinstance(e, LLMResponseError):
            logging.error(f"LLM call failed: {str(e)}")
            raise
        logging.error(f"Unexpected error in LLM call: {str(e)}")
        raise LLMNoResponseError("LLM failed due to an unexpected error")

def connect_to_db():
    return psycopg2.connect(NEON_DB_URL)

def get_embeddings(query):
    try:
        return embeddings.embed_query(query)
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_neon_db(query_embedding, table_name="documents", top_k=5):
    conn = None
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"""
                SELECT id, vector, contents, title, chunk_id 
                FROM {table_name}
                WHERE vector IS NOT NULL
            """
            cur.execute(query)
            rows = cur.fetchall()
            
            similarities = []
            for row in rows:
                try:
                    vector_str = row['vector']
                    vector_values = vector_str.strip('[]').split(',')
                    vector = np.array([float(x.strip()) for x in vector_values])
                    
                    similarity = cosine_similarity(query_embedding, vector)
                    similarities.append((similarity, row))
                except Exception as e:
                    logging.error(f"Error processing vector for row {row['id']}: {str(e)}")
                    continue
            
            similarities.sort(reverse=True, key=lambda x: x[0])
            return [
                {
                    'id': row['id'],
                    'contents': row['contents'],
                    'title': row['title'],
                    'chunk_id': row['chunk_id'],
                    'similarity_score': float(sim)
                }
                for sim, row in similarities[:top_k]
            ]

    except Exception as e:
        logging.error(f"Error in search_neon_db: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def handle_query(query):
    query_embedding = get_embeddings(query)
    results = search_neon_db(query_embedding)
    return results

# Update the custom retriever class
class CustomNeonRetriever(BaseRetriever, BaseModel):
    table_name: str = Field(default="documents")
    
    def _get_relevant_documents(self, query: str) -> List[LangchainDocument]:
        query_embedding = get_embeddings(query)
        results = search_neon_db(query_embedding, self.table_name)
        
        documents = []
        for result in results:
            doc = LangchainDocument(
                page_content=result['contents'],
                metadata={
                    'title': result['title'],
                    'source': self.table_name
                }
            )
            documents.append(doc)
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[LangchainDocument]:
        return await self._get_relevant_documents(query)

@app.route('/')
@app.route('/database')
def serve_spa():
    return jsonify({"message": "Hello from Connect America API"})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'API is running'
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Log request details
        logger.info(f"Request received at: {datetime.now()}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Verify request data
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        # Validate required fields
        if 'message' not in data:
            logger.error("No message field in request")
            return jsonify({'error': 'message field is required'}), 400

        user_query = data['message']
        chat_history = data.get('chat_history', [])

        # Log processing steps
        logger.info(f"Processing user query: {user_query}")
        logger.info(f"Chat history length: {len(chat_history)}")

        # Mock response for testing
        response = {
            'response': f"Echo: {user_query}",
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Successfully processed request")
        return jsonify(response)

    except Exception as e:
        # Detailed error logging
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error("Exception details:")
        logger.error("Type: %s", exc_type)
        logger.error("Value: %s", exc_value)
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'type': str(exc_type),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('message', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Generate embeddings for the query
        query_embedding = get_embeddings(query)
        
        # Get raw results from database
        results = search_neon_db(query_embedding)
        
        # Return only the database results
        return jsonify({
            'results': results,
            'count': len(results)
        }), 200

    except Exception as e:
        logging.error(f"Error in search route: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://connect-america-2-frontend.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Add a test route to verify the API is working
@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Basic error handlers
@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"404 error: {error}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
