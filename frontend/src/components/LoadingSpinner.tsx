import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading analytical data...',
  className = '',
}) => {
  return (
    <div className={`flex flex-col items-center justify-center py-16 text-center ${className}`}>
      <Loader2 className="w-8 h-8 text-bronze animate-spin mb-3" />
      <span className="text-sm font-medium text-ivory">{message}</span>
      <span className="text-xs text-taupe mt-1">Grounded in authoritative datasets</span>
    </div>
  );
};
