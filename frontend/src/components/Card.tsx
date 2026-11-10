import React from 'react';

interface CardProps {
  title?: string;
  subtitle?: string;
  badge?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  badge,
  actions,
  children,
  className = '',
}) => {
  return (
    <div
      className={`bg-espresso border border-chocolate rounded-lg p-5 transition-colors ${className}`}
    >
      {(title || subtitle || actions) && (
        <div className="flex items-start justify-between gap-4 pb-3.5 mb-4 border-b border-chocolate/60">
          <div>
            <div className="flex items-center gap-2">
              {title && (
                <h3 className="font-display text-sm font-semibold tracking-tight text-ivory">
                  {title}
                </h3>
              )}
              {badge && (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-chocolate text-taupe border border-mocha">
                  {badge}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="text-xs text-taupe mt-0.5 leading-normal">
                {subtitle}
              </p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
};
