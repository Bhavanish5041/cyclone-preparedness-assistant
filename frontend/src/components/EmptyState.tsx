import React from 'react';
import { Search } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No administrative units found',
  description = 'Try adjusting your search criteria, district scope, or priority band filters.',
  action,
  className = '',
}) => {
  return (
    <div className={`p-10 text-center flex flex-col items-center justify-center text-taupe ${className}`}>
      <div className="w-12 h-12 rounded-full bg-roast border border-chocolate flex items-center justify-center text-bronze mb-3">
        <Search className="w-5 h-5" />
      </div>
      <h4 className="text-sm font-semibold text-ivory">{title}</h4>
      <p className="text-xs text-taupe max-w-sm mt-1 mb-4 leading-relaxed">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
};
