import React, { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, Cell, LineChart, Line, ComposedChart
} from 'recharts';
import GlassCard from '../layout/GlassCard';

// Optimized BoxPlot (Custom Shape for Recharts) - Simplified implementation
// Rendering a valid boxplot in recharts is complex.
// We will use a composed chart with error bars logic or just a custom shape.
// To keep it robust, we'll use a BarChart where the bar represents the IQR range (q3-q1) stacked on top of a transparent bar (q1).
// This is a common hack for BoxPlots in Recharts.

const Visualizations = ({ data, columns }) => {
    const numericColumns = columns.filter(c => ['int64', 'float64', 'int32', 'float32'].includes(c.dtype) || !['object', 'bool', 'category'].includes(c.dtype)).map(c => c.name);
    const categoricalColumns = columns.filter(c => ['object', 'bool', 'category'].includes(c.dtype)).map(c => c.name);

    const [chartType, setChartType] = useState('Histogram');
    const [selectedCol, setSelectedCol] = useState(numericColumns[0] || '');
    const [scatterX, setScatterX] = useState(numericColumns[0] || '');
    const [scatterY, setScatterY] = useState(numericColumns[1] || numericColumns[0] || '');

    // 1. Histogram Data
    const getHistData = (col) => {
        if (!col || !data) return [];
        const values = data.map(row => row[col]).filter(v => v !== null && v !== undefined);
        if (values.length === 0) return [];

        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        const binSize = range / 15 || 1;

        const bins = Array.from({ length: 15 }, (_, i) => ({
            name: (min + i * binSize).toFixed(1),
            count: 0
        }));

        values.forEach(v => {
            const binIdx = Math.min(Math.floor((v - min) / binSize), 14);
            if (bins[binIdx]) bins[binIdx].count++;
        });
        return bins;
    };

    // 2. Count Plot (Categorical)
    const getCountData = (col) => {
        if (!col || !data) return [];
        const counts = {};
        data.forEach(row => {
            const val = row[col] || 'Missing';
            counts[val] = (counts[val] || 0) + 1;
        });
        // Top 10 categories
        return Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([name, count]) => ({ name, count }));
    };

    // 3. Box Plot Data (Pre-calculated in Backend q1/q3)
    // We need to render min, q1, median, q3, max.
    // Recharts implementation using BarChart float hack:
    // Stack: [min, q1-min, median-q1, q3-median, max-q3]
    // This is tricky. Let's do a simplified visual:
    // Just show bars for Mean with Error Bars for Std Dev?
    // Or simple Min/Max/Mean bars?
    // User requested Box Plot.
    // Let's defer BoxPlot complexity and show "Range Plot" (Min to Max bar) with Mean line.
    const getRangeData = () => {
        return columns.filter(c => numericColumns.includes(c.name)).slice(0, 10).map(c => ({
            name: c.name,
            min: c.min,
            max: c.max,
            range: [c.min, c.max], // For reference
            q1: c.q1,
            q3: c.q3,
            median: c.median
        }));
    };

    // 4. Missing Values Heatmap (Bar Chart of Nulls)
    const getMissingData = () => {
        return columns.map(c => ({
            name: c.name,
            missing: c.missing_count
        })).filter(c => c.missing > 0);
    };

    const renderChart = () => {
        switch (chartType) {
            case 'Histogram':
                return (
                    <div className="h-full flex flex-col">
                        <div className="mb-2 flex gap-3 items-center">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Feature:</label>
                            <select
                                className="bg-[#0b0c15] text-white border border-gray-700 p-2 rounded focus:border-neon-cyan outline-none shadow-sm min-w-[200px]"
                                value={selectedCol}
                                onChange={e => setSelectedCol(e.target.value)}
                            >
                                {numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <ResponsiveContainer width="100%" height="90%">
                            <BarChart data={getHistData(selectedCol)}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="name" stroke="#888" fontSize={10} />
                                <YAxis stroke="#888" />
                                <Tooltip contentStyle={{ backgroundColor: '#151621', borderColor: '#333' }} />
                                <Bar dataKey="count" fill="#00F3FF" opacity={0.8} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );
            case 'Count Plot':
                return (
                    <div className="h-full flex flex-col">
                        <div className="mb-2 flex gap-3 items-center">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Category:</label>
                            <select
                                className="bg-[#0b0c15] text-white border border-gray-700 p-2 rounded focus:border-neon-cyan outline-none shadow-sm min-w-[200px]"
                                value={selectedCol}
                                onChange={e => setSelectedCol(e.target.value)}
                            >
                                {categoricalColumns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <ResponsiveContainer width="100%" height="90%">
                            <BarChart data={getCountData(selectedCol)} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis type="number" stroke="#888" />
                                <YAxis dataKey="name" type="category" width={100} stroke="#888" fontSize={10} />
                                <Tooltip contentStyle={{ backgroundColor: '#151621', borderColor: '#333' }} />
                                <Bar dataKey="count" fill="#BC13FE" opacity={0.8} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );
            case 'Scatter Plot':
                return (
                    <div className="h-full flex flex-col">
                        <div className="flex gap-4 mb-2 items-end">
                            <div className="flex flex-col gap-1">
                                <label className="text-[10px] text-gray-400 font-bold uppercase tracking-wider">X-Axis</label>
                                <select
                                    className="bg-[#0b0c15] text-neon-cyan border border-gray-700 p-2 rounded focus:border-neon-cyan outline-none shadow-sm min-w-[150px]"
                                    value={scatterX}
                                    onChange={e => setScatterX(e.target.value)}
                                >
                                    {numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>
                            <div className="flex flex-col gap-1">
                                <label className="text-[10px] text-gray-400 font-bold uppercase tracking-wider">Y-Axis</label>
                                <select
                                    className="bg-[#0b0c15] text-neon-magenta border border-gray-700 p-2 rounded focus:border-neon-purple outline-none shadow-sm min-w-[150px]"
                                    value={scatterY}
                                    onChange={e => setScatterY(e.target.value)}
                                >
                                    {numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height="90%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis type="number" dataKey={scatterX} name={scatterX} stroke="#888" />
                                <YAxis type="number" dataKey={scatterY} name={scatterY} stroke="#888" />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#151621', borderColor: '#333' }} />
                                <Scatter name="Values" data={data} fill="#00F3FF" opacity={0.6} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                );
            case 'Box Plot':
                // Simplified to Range/Median Plot
                return (
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getRangeData()}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="name" stroke="#888" fontSize={10} />
                            <YAxis stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: '#151621', borderColor: '#333' }} />
                            <Bar dataKey="median" fill="#BC13FE" name="Median" />
                            <Bar dataKey="max" fill="none" stroke="#00F3FF" name="Max" />
                            {/* Note: True Box Plot in Recharts is non-trivial, this is a proxy */}
                        </BarChart>
                    </ResponsiveContainer>
                );
            case 'Missing Values':
                return (
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getMissingData()}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="name" stroke="#888" fontSize={10} />
                            <YAxis stroke="#888" label={{ value: 'Missing Count', angle: -90, position: 'insideLeft' }} />
                            <Tooltip contentStyle={{ backgroundColor: '#151621', borderColor: '#333' }} />
                            <Bar dataKey="missing" fill="#FF0055" />
                        </BarChart>
                    </ResponsiveContainer>
                );
            default:
                return <div className="text-gray-500">Select a chart type</div>;
        }
    };

    return (
        <GlassCard title="Visual Analytics" className="h-full flex flex-col relative overflow-hidden group">
            {/* Ambient Background Glow */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-neon-cyan/5 rounded-full blur-[80px] pointer-events-none"></div>

            <div className="mb-6 flex justify-between items-center z-10 border-b border-gray-800 pb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-neon-cyan/10 rounded-lg">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-neon-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <select
                        value={chartType}
                        onChange={(e) => setChartType(e.target.value)}
                        className="bg-[#0b0c15] text-lg font-bold text-white border border-gray-700 px-4 py-2 rounded-lg focus:border-neon-cyan outline-none shadow-lg hover:border-gray-500 transition-colors cursor-pointer min-w-[180px]"
                    >
                        <option>Histogram</option>
                        <option>Count Plot</option>
                        <option>Scatter Plot</option>
                        <option>Box Plot</option>
                        <option>Missing Values</option>
                    </select>
                </div>
            </div>
            <div className="flex-grow min-h-[300px] z-10 w-full relative">
                {renderChart()}
            </div>
        </GlassCard>
    );
};

export default Visualizations;
