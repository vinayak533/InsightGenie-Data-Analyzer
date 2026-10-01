import React, { useState } from 'react';
import GlassCard from '../layout/GlassCard';

const MLRecommendations = ({ recommendations, problemType, detectedTarget, targetCol, onGetRecommendations }) => {
    const [target, setTarget] = useState(targetCol || '');
    const [loading, setLoading] = useState(false);

    const handleRecommend = async () => {
        if (!target) return;
        setLoading(true);
        await onGetRecommendations(target);
        setLoading(false);
    };

    return (
        <GlassCard title="AI Analyst Recommendations" className="h-full">
            <div className="space-y-4">
                <div className="flex gap-2 mb-4">
                    <input
                        type="text"
                        placeholder="Target Column (optional)"
                        className="w-full bg-black/30 border border-gray-600 text-white p-2 rounded focus:border-neon-cyan outline-none"
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                    />
                    <button
                        onClick={handleRecommend}
                        disabled={loading}
                        className="btn-secondary whitespace-nowrap"
                    >
                        {loading ? 'Thinking...' : 'Analyze Goal'}
                    </button>
                </div>

                {problemType && (
                    <div className="mb-4 p-3 bg-neon-purple/10 border border-neon-purple rounded text-center space-y-1">
                        {detectedTarget && (
                            <div className="text-sm text-gray-300">
                                Target Column: <span className="font-bold text-white">{detectedTarget}</span>
                            </div>
                        )}
                        <div>
                            Problem Type: <strong className="text-neon-cyan">{problemType}</strong>
                        </div>
                    </div>
                )}

                <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
                    {recommendations && recommendations.length > 0 ? (
                        recommendations.map((rec, idx) => (
                            <div key={idx} className="p-3 bg-white/5 rounded border border-white/10 hover:border-neon-cyan transition-colors">
                                <div className="flex justify-between items-center mb-1">
                                    <h4 className="font-bold text-neon-cyan">{rec.algorithm}</h4>
                                    <span className="text-xs bg-gray-800 px-2 py-1 rounded text-gray-400">{rec.type}</span>
                                </div>
                                <p className="text-sm text-gray-300 italic">
                                    "{rec.reasoning}"
                                </p>
                            </div>
                        ))
                    ) : (
                        <div className="text-center text-gray-500 py-10">
                            Enter a target variable above to get tailored ML advice.
                        </div>
                    )}
                </div>
            </div>
        </GlassCard>
    );
};

export default MLRecommendations;
