import React, { useState } from 'react';

const UploadWizard = ({ onUploadSuccess }) => {
    const [dragActive, setDragActive] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = async (file) => {
        if (file.type !== "text/csv" && !file.name.endsWith('.csv')) {
            setError("Please upload a CSV file.");
            return;
        }

        setUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/analyze", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Analysis failed");

            const data = await response.json();
            onUploadSuccess(data, file); // Pass analysis result and file

        } catch (err) {
            setError(err.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto text-center">
            <div
                className={`glass-panel p-10 border-2 border-dashed transition-all duration-300 ${dragActive ? "border-neon-cyan bg-white/5" : "border-gray-600"
                    }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept=".csv"
                    onChange={handleChange}
                />

                <div className="flex flex-col items-center justify-center space-y-4">
                    <div className="text-6xl">📂</div>
                    <h2 className="text-2xl font-bold text-white">Upload your Dataset</h2>
                    <p className="text-gray-400">Drag & drop your CSV file here, or click to browse</p>

                    {uploading ? (
                        <div className="text-neon-cyan animate-pulse">Analyzing Data...</div>
                    ) : (
                        <label htmlFor="file-upload" className="btn-primary inline-block">
                            Select File
                        </label>
                    )}

                    {error && <p className="text-red-500 mt-2">{error}</p>}
                </div>
            </div>
        </div>
    );
};

export default UploadWizard;
