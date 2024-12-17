// hooks/useChat.ts
import { useState } from 'react';
import { ChatMessage } from '@/types/chat';

const API_URL = 'https://connect-america-2.vercel.app';

export function useChat() {
  const [messages, setMessages] = useState<(ChatMessage & { urls?: string[] })[]>([]);
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
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          chat_history: chatHistory
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add messages to state
      const userMessage: ChatMessage = {
        role: 'user',
        content,
      };
      
      const assistantMessage: ChatMessage & { urls?: string[] } = {
        role: 'assistant',
        content: data.response || 'Sorry, I could not process your request.',
        urls: data.urls?.map((url: { url: string } | string) => 
          typeof url === 'string' ? url : url.url
        ) || []
      };

      setMessages(prev => [...prev, userMessage, assistantMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      setError(err instanceof Error ? err.message : 'Failed to send message. Please check if the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  return {
    messages,
    loading,
    error,
    sendMessage
  };
}
