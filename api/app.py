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
7. ONLY cite URLs from the source documents that directly support your answer using {{url:}} format

Response Structure and Formatting:
   - Use markdown formatting with clear hierarchical structure
   - Each major section must start with '### ' followed by a number and bold title
   - Format section headers as: ### 1. **Title Here**
   - Use bullet points (-) for detailed explanations
   - Each fact must be cited with {{url:}} format using ONLY the s3_url from the source document where that specific information was found
   - Keep responses clear, practical, and focused on support topics

Example Response:
Question: "What is the battery capacity of the smartwatch?"
Context document s3_url: "s3://docs/specs.pdf" contains: "Battery: 600mAh"
Correct response: "The smartwatch has a 600mAh battery capacity {{url:s3://docs/specs.pdf}}"
"""

app.secret_key = os.urandom(24)

# Initialize API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "connect-america"

# Initialize Langchain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)

logging.basicConfig(level=logging.DEBUG)

def rewrite_query(query, chat_history=None):
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

def process_response(response, source_documents):
    """Extract and verify URLs from the response"""
    url_pattern = r'\{url:(.*?)\}'
    urls_in_response = re.findall(url_pattern, response)
    
    # Create a dictionary of valid URLs from source documents
    valid_urls = {doc.metadata.get('s3_url'): doc for doc in source_documents if doc.metadata.get('s3_url')}
    
    # Use a dictionary to track unique URLs and their content
    unique_urls = {}
    for url in urls_in_response:
        if url in valid_urls and url not in unique_urls:
            unique_urls[url] = {
                'url': url,
                'content': valid_urls[url].page_content
            }
    
    # Clean up the response by removing the URL tags
    cleaned_response = re.sub(url_pattern, '', response)
    
    return cleaned_response, list(unique_urls.values())

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
                SELECT id, vector, contents, title, chunk_id, s3_url 
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
                    's3_url': row['s3_url'],
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
                    'source': self.table_name,
                    's3_url': result['s3_url']
                }
            )
            documents.append(doc)
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[LangchainDocument]:
        return await self._get_relevant_documents(query)

@app.route('/')
def serve_spa():
    return jsonify({"message": "Hello from Connect America API"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        user_query = data.get('message')
        chat_history = data.get('chat_history', [])
        
        if not user_query:
            return jsonify({'error': 'No message provided'}), 400

        try:
            # Format chat history correctly
            formatted_history = []
            for msg in chat_history:
                if isinstance(msg, dict):
                    if 'role' in msg and 'content' in msg:
                        formatted_history.append((msg['content'], ''))
                else:
                    logging.warning(f"Unexpected chat history format: {msg}")
            
            # Detailed relevance check
            relevance_check_prompt = f"""
            Given the following question or message and the chat history, determine if it is:
            1. A greeting or send-off (like "hello", "thank you", "goodbye", or casual messages)
            2. Related to Connect America's core services:
                - Medical Alert Systems and Devices
                - Remote Patient Monitoring Systems
                - Care Management Services
                - Customer Service Operations
                - Medication Management Solutions
                - asking about the company
            3. Related to:
                - Device setup, troubleshooting, or maintenance
                - Patient monitoring procedures
                - Care coordination processes
                - Customer support protocols
                - Medication tracking systems
            4. A follow-up question to the previous conversation about these topics
            5. Contains inappropriate or harmful content
            6. Completely unrelated to Connect America's healthcare services

            If it falls under category 1, respond with 'GREETING'. 
            If it falls under categories 2, 3, or 4 respond with 'RELEVANT'.
            If it falls under category 5, respond with 'INAPPROPRIATE'.
            If it falls under category 6, respond with 'NOT RELEVANT'.

            Chat History:
            {formatted_history[-3:] if formatted_history else "No previous context"}

            Current Question: {user_query}
            
            Response (GREETING, RELEVANT, INAPPROPRIATE, or NOT RELEVANT):
            """
            
            relevance_response = llm.predict(relevance_check_prompt)
            
            # Handle non-relevant cases using LLM
            if "GREETING" in relevance_response.upper():
                greeting_prompt = f"""
                The following message is a greeting or casual message. Please provide a friendly and engaging response as Connect America's support assistant.
                Make sure to mention that you're here to help with internal support topics.

                Message: {user_query}

                Response:
                """
                greeting_response = llm.predict(greeting_prompt)
                return jsonify({
                    'response': greeting_response,
                    'contexts': [],
                    'urls': []
                })

            elif "INAPPROPRIATE" in relevance_response.upper():
                inappropriate_prompt = f"""
                The following message contains inappropriate content. Provide a professional response that:
                1. Maintains a polite and firm tone
                2. Explains that you can only assist with appropriate, work-related queries
                3. Encourages the user to ask a different question related to Connect America's internal support

                Message: {user_query}

                Response:
                """
                inappropriate_response = llm.predict(inappropriate_prompt)
                return jsonify({
                    'response': inappropriate_response,
                    'contexts': [],
                    'urls': []
                })

            elif "NOT RELEVANT" in relevance_response.upper():
                not_relevant_prompt = f"""
                The following question is not related to Connect America's internal support topics. Provide a response that:
                1. Politely acknowledges the question
                2. Explains that you are specialized in Connect America's internal support topics
                3. Provides examples of topics you can help with
                4. Encourages rephrasing the question to relate to internal support matters

                Question: {user_query}

                Response:
                """
                not_relevant_response = llm.predict(not_relevant_prompt)
                return jsonify({
                    'response': not_relevant_response,
                    'contexts': [],
                    'urls': []
                })

            # Process relevant query
            rewritten_query = rewrite_query(user_query, formatted_history)
            
            retriever = CustomNeonRetriever(table_name="documents")
            relevant_docs = retriever.get_relevant_documents(rewritten_query)
            
            # Create prompt with source URLs
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTIONS),
                HumanMessagePromptTemplate.from_template(
                    "Context: {context}\n\n"
                    "Available source URLs:\n{urls}\n\n"
                    "Chat History: {chat_history}\n\n"
                    "Question: {question}"
                )
            ])
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            # Execute the chain with URLs included
            chain_response = qa_chain({
                "question": rewritten_query,
                "chat_history": formatted_history,
                "urls": "\n".join([f"- {doc.metadata.get('s3_url', '')}" for doc in relevant_docs])
            })
            
            # Process the response to extract and verify URLs
            processed_response, verified_urls = process_response(
                chain_response['answer'],
                chain_response['source_documents']
            )
            
            return jsonify({
                'response': processed_response,
                'contexts': [{
                    'content': doc.page_content,
                    's3_url': doc.metadata.get('s3_url')
                } for doc in chain_response['source_documents']],
                'urls': verified_urls
            })
            
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error processing message: {str(e)}'}), 500
            
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('message', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        query_embedding = get_embeddings(query)
        results = search_neon_db(query_embedding)
        
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

if __name__ == '__main__':  
    app.run(debug=True)
