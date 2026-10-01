import React from 'react';
import GlassCard from '../layout/GlassCard';

const HealthCheck = ({ health }) => {
    if (!health) return null;

    const getHealthColor = (score) => {
        if (score >= 90) return 'text-green-400';
        if (score >= 70) return 'text-yellow-400';
        return 'text-red-500';
    };

    return (
        <GlassCard title="Dataset Health" className="h-full">
            <div className="flex flex-col items-center justify-center">
                <div className="relative w-32 h-32 flex items-center justify-center mb-4">
                    <svg viewBox="0 0 36 36" className="w-full h-full">
                        <path
                            className="text-gray-700"
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                        />
                        <path
                            className={getHealthColor(health.health_score)}
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                            strokeDasharray={`${health.health_score}, 100`}
                        />
                    </svg>
                    <span className={`absolute text-2xl font-bold ${getHealthColor(health.health_score)}`}>
                        {health.health_score}%
                    </span>
                </div>

                <div className="w-full space-y-2 text-sm text-gray-300">
                    <div className="flex justify-between">
                        <span>Rows:</span>
                        <span className="text-white">{health.total_rows}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Columns:</span>
                        <span className="text-white">{health.total_columns}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Missing Values:</span>
                        <span className="text-red-400">{health.missing_values_count}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Duplicates:</span>
                        <span className="text-yellow-400">{health.duplicate_rows}</span>
                    </div>
                </div>
            </div>
        </GlassCard>
    );
};

export default HealthCheck;
