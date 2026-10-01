import React from 'react';

const GlassCard = ({ children, className = '', title }) => {
  return (
    <div className={`glass-panel p-6 ${className}`}>
      {title && (
        <h3 className="text-xl mb-4 font-semibold text-gray-100 border-b border-gray-700 pb-2">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
};

export default GlassCard;
