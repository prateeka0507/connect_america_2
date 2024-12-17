'use client';
import { useState, useEffect, useRef } from 'react';
import { useChat } from '@/hooks/useChat';
import ReactMarkdown from 'react-markdown';
import { KeyboardEvent } from 'react';

export default function ChatPage() {
  const { messages, sendMessage, loading, error: chatError } = useChat();
  const [inputValue, setInputValue] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const trimmedInput = inputValue.trim();
    
    if (!trimmedInput) return;
    
    setError(null);
    
    try {
      await sendMessage(trimmedInput);
      setInputValue('');
    } catch (error) {
      console.error('Error in handleSubmit:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="flex h-screen relative">
      {/* Mobile Header - Updated with dark blue theme */}
      <div className="lg:hidden fixed top-0 left-0 right-0 h-14 bg-[#0A0F5C] z-40 flex items-center px-4">
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-1.5 hover:bg-blue-900 rounded-lg"
        >
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <div className="ml-4">
          <h1 className="text-lg font-bold text-white">ConnectAmerica</h1>
          <p className="text-sm text-gray-300">AI Support Assistant</p>
        </div>
      </div>

      {/* Left Sidebar - Updated for mobile */}
      <div className={`
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0
        fixed lg:relative
        w-64 h-full
        bg-[#0A0F5C] text-white
        transition-transform duration-300 ease-in-out
        z-40 lg:z-auto
        ${isSidebarOpen ? 'top-14 h-[calc(100%-56px)]' : 'top-0 h-full'} 
        lg:top-0 lg:h-full
      `}>
        <div className="p-4 flex flex-col h-full">
          {/* Hide title on mobile since it's in the header */}
          <div className="mb-8 hidden lg:block">
            <h1 className="text-xl font-bold">ConnectAmerica</h1>
            <p className="text-sm text-gray-300 mt-2">AI Support Assistant</p>
          </div>
          
          <nav className="flex-1">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-teal-400">💬</span>
              <span className="text-teal-400 font-semibold">Recent Chats</span>
            </div>
            
            <div className="space-y-2">
              {messages.length > 0 ? (
                messages
                  .filter(msg => msg.role === 'user')
                  .slice(-5)
                  .map((msg, idx) => (
                    <div key={idx} className="p-2 rounded hover:bg-blue-900 cursor-pointer truncate text-sm">
                      {msg.content}
                    </div>
                  ))
              ) : (
                <div className="text-gray-400 text-sm">No recent chats</div>
              )}
            </div>
          </nav>

          <div className="mt-auto pt-4 border-t border-gray-700">
            <div className="flex items-center gap-2 text-sm text-gray-300 hover:text-white cursor-pointer">
              <span>⚙️</span>
              <span>Settings</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area - Adjusted for mobile header */}
      <div className="flex-1 flex flex-col w-full lg:w-auto">
        {/* Desktop Header - Hidden on mobile */}
        <div className="hidden lg:block bg-white py-4 px-6">
          <h2 className="text-lg font-semibold text-gray-800">Chat Assistant</h2>
          <p className="text-sm text-gray-500">Ask me anything about Connect America</p>
        </div>

        {/* Messages Area - Adjusted top padding for mobile */}
        <div className="flex-1 bg-gray-50 overflow-y-auto px-2 sm:px-4 
          pt-14 lg:pt-0" // Added top padding for mobile header
        >
          {messages.length === 0 && (
            <div className="text-center text-gray-500 mt-10 px-4">
              <p className="text-lg">👋 Welcome to Connect America Support</p>
              <p className="text-sm mt-2">How can I help you today?</p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div 
              key={`${message.role}-${index}`}
              className={`${
                message.role === 'assistant' 
                  ? 'bg-white' 
                  : 'bg-[#F5F7FF]'
              } py-3 sm:py-4 px-3 sm:px-4 mb-1`}
            >
              <div className="text-xs font-medium mb-1 text-gray-500">
                {message.role === 'assistant' ? 'AI Assistant' : 'You'}
              </div>
              <div className="text-gray-700 prose prose-sm sm:prose-base prose-blue max-w-none">
                {message.role === 'assistant' ? (
                  <ReactMarkdown className="markdown-content">
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <div className="user-message text-sm sm:text-base">{message.content}</div>
                )}
              </div>
            </div>
          ))}
          
          {/* Loading and Error states - Responsive */}
          {loading && (
            <div className="bg-white py-3 px-4">
              <div className="flex items-center gap-2 text-gray-500 text-sm">
                <div className="animate-spin">⟳</div>
                <div>Processing your request...</div>
              </div>
            </div>
          )}
          
          {(error || chatError) && (
            <div className="bg-red-50 py-3 px-4 text-red-600 text-sm">
              {error || chatError}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area - With distinct background color */}
        <div className="bg-[#F0F2FF]">
          <form onSubmit={handleSubmit} className="relative flex items-center">
            <textarea 
              className="w-full bg-[#F5F7FF] focus:bg-white
              text-gray-800 text-base sm:text-lg font-medium placeholder-gray-400
              border-0 outline-none resize-none 
              py-6 px-6 sm:px-8
              min-h-[80px] sm:min-h-[100px] max-h-[300px] 
              overflow-auto transition-colors duration-200" 
              placeholder="Type your message here..."
              rows={1}
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                e.target.style.height = 'inherit';
                e.target.style.height = `${e.target.scrollHeight}px`;
              }}
              disabled={loading}
              onKeyDown={handleKeyDown}
            />
            <button 
              type="submit"
              disabled={loading}
              className="absolute right-5 sm:right-8 
              bg-[#0A0F5C] text-white 
              p-4 sm:p-5 
              rounded-lg hover:bg-blue-900 
              transition-colors disabled:opacity-50 
              flex items-center justify-center"
            >
              {loading ? (
                <span className="animate-spin text-2xl">⟳</span>
              ) : (
                <span className="text-2xl">➤</span>
              )}
            </button>
          </form>
        </div>
      </div>

      {/* Overlay for mobile sidebar */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
    </div>
  );
}
