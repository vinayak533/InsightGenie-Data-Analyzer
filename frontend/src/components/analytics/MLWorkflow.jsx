import React from 'react';
import GlassCard from '../layout/GlassCard';

const MLWorkflow = ({ steps }) => {
    return (
        <GlassCard title="ML Workflow Steps" className="h-full overflow-y-auto">
            <div className="space-y-4">
                {steps && steps.length > 0 ? (
                    steps.map((step, index) => (
                        <div key={index} className="flex gap-3">
                            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-neon-purple/20 border border-neon-purple flex items-center justify-center text-xs text-neon-cyan font-bold">
                                {index + 1}
                            </div>
                            <div className="text-sm text-gray-300">
                                {step.split(':').length > 1 ? (
                                    <>
                                        <span className="text-white font-semibold">{step.split(':')[0]}:</span>
                                        <span className="text-gray-400">{step.split(':')[1]}</span>
                                    </>
                                ) : (
                                    step
                                )}
                            </div>
                        </div>
                    ))
                ) : (
                    <div className="text-gray-500 text-center text-sm py-4">
                        Genereate ML recommendations to see the workflow.
                    </div>
                )}
            </div>
        </GlassCard>
    );
};

export default MLWorkflow;
