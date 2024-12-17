import React from 'react';

interface ReferencesProps {
  urls: string[];
}

const References: React.FC<ReferencesProps> = ({ urls }) => {
  if (!urls || urls.length === 0) return null;

  const getDocumentTitle = (url: string) => {
    const filename = url.split('/').pop() || '';
    return filename
      .replace(/[-_]/g, ' ')
      .replace(/\.[^/.]+$/, '')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const handleDownload = async (url: string) => {
    try {
      // First try direct download
      const link = document.createElement('a');
      link.href = url;
      link.download = url.split('/').pop() || 'document.pdf';
      link.target = '_blank';
      
      // Try to convert to blob if it's a PDF
      if (url.toLowerCase().endsWith('.pdf')) {
        try {
          const response = await fetch(url, {
            mode: 'cors',
            credentials: 'include',
            headers: {
              'Accept': 'application/pdf'
            }
          });
          const blob = await response.blob();
          link.href = window.URL.createObjectURL(blob);
        } catch (error) {
          console.warn('Blob creation failed, falling back to direct download', error);
        }
      }
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      if (link.href.startsWith('blob:')) {
        window.URL.revokeObjectURL(link.href);
      }
    } catch (error) {
      console.error('Download failed:', error);
      // Fallback: open in new tab
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4 relative z-10 bg-transparent">
      <h2 className="text-xl font-semibold text-gray-800 bg-white px-4 py-2 rounded-lg shadow-sm flex items-center gap-2">
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Referenced Documents
      </h2>
      <div className="flex flex-col gap-3">
        {urls.map((url, index) => (
          <div
            key={index}
            className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-4 rounded-lg border border-gray-200 hover:border-blue-500 transition-colors bg-white shadow-sm hover:shadow-md"
          >
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-blue-700 font-semibold">#{index + 1}</span>
                <h3 className="text-gray-900 text-lg font-medium">
                  {getDocumentTitle(url)}
                </h3>
              </div>
              <p className="text-sm text-gray-500 mt-1">
                {url.includes('.pdf') ? 'PDF Document' : 
                 url.includes('.doc') ? 'Word Document' : 
                 'Document'}
              </p>
            </div>
            <div className="flex gap-2 w-full sm:w-auto">
              <a 
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors flex-1 sm:flex-initial justify-center"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                View
              </a>
              <a 
                href={url}
                className="flex items-center gap-2 px-4 py-2 rounded-md bg-gray-100 text-gray-800 hover:bg-gray-200 transition-colors flex-1 sm:flex-initial justify-center"
                onClick={(e) => {
                  e.preventDefault();
                  handleDownload(url);
                }}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default References; 