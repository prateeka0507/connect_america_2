// hooks/useChat.ts
import { useState } from 'react';
import { ChatMessage, ChatResponse } from '@/types/chat';

// Direct URL instead of environment variable
const API_URL = 'http://localhost:5000';

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async (content: string) => {
    setLoading(true);
    setError(null);

    try {
      // Format the chat history properly
      const chatHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          chat_history: chatHistory
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add messages to state
      const userMessage: ChatMessage = {
        role: 'user',
        content,
      };
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.response || 'Sorry, I could not process your request.',
      };

      setMessages(prev => [...prev, userMessage, assistantMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      setError(err instanceof Error ? err.message : 'Failed to send message');
    } finally {
      setLoading(false);
    }
  };

  return {
    messages,
    sendMessage,
    loading,
    error
  };
}