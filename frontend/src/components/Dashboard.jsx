import React, { useState } from 'react';
import HealthCheck from './analytics/HealthCheck';
import MLRecommendations from './analytics/MLRecommendations';
import Visualizations from './analytics/Visualizations';
import MLWorkflow from './analytics/MLWorkflow';
import GlassCard from './layout/GlassCard';

const Dashboard = ({ analysisData, file }) => {
    const [recommendations, setRecommendations] = useState([]);
    const [workflow, setWorkflow] = useState([]);
    const [problemType, setProblemType] = useState(null);
    const [detectedTarget, setDetectedTarget] = useState(null);
    const [activeTab, setActiveTab] = useState('viz');

    const fetchRecommendations = async (targetColumn) => {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("target_column", targetColumn || '');

        try {
            const response = await fetch("http://localhost:8000/recommend", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            setRecommendations(data.recommendations);
            setWorkflow(data.workflow); // Set workflow steps
            setProblemType(data.problem_type);
            setDetectedTarget(data.target_variable);
        } catch (err) {
            console.error("Failed to get recs", err);
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-[calc(100vh-140px)] min-h-[600px]">
            {/* LEFT COLUMN: Dataset Info (25%) */}
            <div className="col-span-12 md:col-span-3 flex flex-col gap-6 h-full overflow-y-auto pr-2 custom-scrollbar">
                <div className="flex-shrink-0">
                    <HealthCheck health={analysisData.health} />
                </div>
                <div className="flex-grow">
                    <GlassCard title="Feature Statistics" className="h-full overflow-hidden flex flex-col">
                        <div className="overflow-y-auto flex-grow custom-scrollbar pr-2">
                            <div className="space-y-3">
                                {analysisData.columns.map(col => (
                                    <div key={col.name} className="p-2 bg-white/5 rounded border border-white/5 hover:border-neon-cyan/30 transition-colors">
                                        <div className="flex justify-between items-center mb-1">
                                            <div className="text-neon-cyan font-bold truncate w-2/3" title={col.name}>{col.name}</div>
                                            <div className="text-[10px] text-gray-500 bg-black/40 px-1 rounded">{col.dtype}</div>
                                        </div>
                                        <div className="text-xs text-gray-400 flex justify-between">
                                            <span>Unique: {col.unique_count}</span>
                                            <span className={col.missing_count > 0 ? "text-red-400" : "text-gray-500"}>
                                                Nulls: {col.missing_count}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </GlassCard>
                </div>
            </div>

            {/* MIDDLE COLUMN: Visualization & Data (50%) */}
            <div className="col-span-12 md:col-span-6 flex flex-col gap-4 h-full">
                {/* Tabs */}
                <div className="flex gap-4 border-b border-gray-800 pb-2">
                    <button
                        onClick={() => setActiveTab('viz')}
                        className={`text-lg font-bold px-4 py-2 rounded transition-all ${activeTab === 'viz' ? 'bg-neon-cyan/10 text-neon-cyan border-b-2 border-neon-cyan' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                        Visualizations
                    </button>
                    <button
                        onClick={() => setActiveTab('data')}
                        className={`text-lg font-bold px-4 py-2 rounded transition-all ${activeTab === 'data' ? 'bg-neon-purple/10 text-neon-magenta border-b-2 border-neon-magenta' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                        Data Preview
                    </button>
                </div>

                {/* Content */}
                <div className="flex-grow overflow-hidden relative">
                    {activeTab === 'viz' ? (
                        <Visualizations
                            data={analysisData.sample_data}
                            columns={analysisData.columns}
                            correlation={analysisData.correlation_matrix}
                        />
                    ) : (
                        <GlassCard title="Raw Data (First 5 Rows)" className="h-full overflow-hidden flex flex-col">
                            <div className="overflow-auto custom-scrollbar">
                                <table className="w-full text-left text-sm text-gray-400">
                                    <thead className="text-gray-200 border-b border-gray-700 bg-black/20 sticky top-0">
                                        <tr>
                                            {analysisData.columns.map(col => (
                                                <th key={col.name} className="p-3 whitespace-nowrap">{col.name}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysisData.head.map((row, i) => (
                                            <tr key={i} className="border-b border-gray-800 hover:bg-white/5">
                                                {analysisData.columns.map(col => (
                                                    <td key={col.name} className="p-3 whitespace-nowrap max-w-[150px] truncate">{row[col.name]}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </GlassCard>
                    )}
                </div>
            </div>

            {/* RIGHT COLUMN: Intelligence (25%) */}
            <div className="col-span-12 md:col-span-3 flex flex-col gap-6 h-full overflow-y-auto custom-scrollbar pl-2">
                <div className="flex-shrink-0">
                    <MLRecommendations
                        recommendations={recommendations}
                        problemType={problemType}
                        detectedTarget={detectedTarget}
                        onGetRecommendations={fetchRecommendations}
                    />
                </div>
                <div className="flex-grow">
                    <MLWorkflow steps={workflow} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
