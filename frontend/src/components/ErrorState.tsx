import React from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Data Retrieval Error',
  message,
  onRetry,
  className = '',
}) => {
  return (
    <div
      className={`bg-roast/90 border border-[#E05D52]/40 rounded-lg p-8 text-center flex flex-col items-center max-w-lg mx-auto ${className}`}
    >
      <div className="w-10 h-10 rounded-full bg-[#E05D52]/15 border border-[#E05D52]/30 flex items-center justify-center text-[#E05D52] mb-3">
        <AlertCircle className="w-5 h-5" />
      </div>
      <h4 className="text-base font-semibold text-ivory font-display">{title}</h4>
      <p className="text-xs text-taupe mt-1 mb-4 leading-relaxed">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded bg-chocolate hover:bg-mocha text-ivory border border-chocolate transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5 text-bronze" />
          Retry Request
        </button>
      )}
    </div>
  );
};
