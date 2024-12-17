// types/chat.ts
export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface ChatResponse {
  response: string;
  urls: string[];
}

export interface Reference {
  id: string;
  title: string;
  url: string;
}

