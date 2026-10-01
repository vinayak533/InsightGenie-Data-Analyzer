import React, { useState } from 'react';
import UploadWizard from './components/upload/UploadWizard';
import Dashboard from './components/Dashboard';

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleUploadSuccess = (data, file) => {
    setAnalysisResult(data);
    setUploadedFile(file);
  };

  return (
    <div className="min-h-screen text-white p-4 md:p-8">
      <header className="mb-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-neon-purple to-neon-cyan rounded-lg animate-pulse"></div>
          <h1 className="text-3xl font-bold tracking-tighter">
            Insight<span className="text-neon-cyan">Genie</span>
          </h1>
        </div>
        {analysisResult && (
          <button
            onClick={() => setAnalysisResult(null)}
            className="text-sm text-gray-400 hover:text-white underline"
          >
            Analyze New File
          </button>
        )}
      </header>

      <main className="max-w-7xl mx-auto">
        {!analysisResult ? (
          <div className="mt-20">
            <h2 className="text-4xl md:text-5xl font-bold text-center mb-6">
              Unlock the <span className="neon-text">Wisdom</span> in your Data
            </h2>
            <p className="text-center text-gray-400 max-w-2xl mx-auto mb-12 text-lg">
              Upload your CSV. Get instant automated insights, visualization, and ML algorithm recommendations powered by advanced AI heuristics.
            </p>
            <UploadWizard onUploadSuccess={handleUploadSuccess} />
          </div>
        ) : (
          <Dashboard analysisData={analysisResult} file={uploadedFile} />
        )}
      </main>

      <footer className="mt-20 text-center text-gray-600 text-sm">
        &copy; 2025 InsightGenie AI. Built for Data Professionals.
      </footer>
    </div>
  );
}

export default App;
