import React from 'react';

interface HeaderProps {
  title: string;
  subtitle: string;
  badge?: string;
  actions?: React.ReactNode;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  badge,
  actions,
}) => {
  return (
    <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 pb-5 border-b border-chocolate/70 mb-6">
      <div className="space-y-1.5">
        <div className="flex items-center gap-2.5">
          <h1 className="font-display text-2xl sm:text-[28px] font-bold text-ivory tracking-tight">
            {title}
          </h1>
          {badge && (
            <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-chocolate text-bronze border border-mocha">
              {badge}
            </span>
          )}
        </div>
        <p className="text-xs sm:text-[13px] text-taupe max-w-3xl leading-relaxed">
          {subtitle}
        </p>
      </div>
      {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
    </div>
  );
};
