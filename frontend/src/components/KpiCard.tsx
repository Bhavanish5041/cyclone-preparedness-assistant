import React from 'react';

interface KpiCardProps {
  title: string;
  value: string | number;
  caption?: string;
  badge?: string;
  accentColor?: string; // hex
  icon?: React.ReactNode;
}

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  caption,
  badge,
  accentColor = '#C18B58',
  icon,
}) => {
  return (
    <div
      className="bg-espresso border border-chocolate rounded-lg p-4 sm:p-5 flex flex-col justify-between relative overflow-hidden"
      style={{ borderTop: `2px solid ${accentColor}` }}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-[11px] font-mono uppercase tracking-wider text-taupe">
          {title}
        </span>
        <div className="flex items-center gap-1.5">
          {badge && (
            <span
              className="text-[10px] font-mono px-1.5 py-0.5 rounded font-medium"
              style={{
                backgroundColor: `${accentColor}18`,
                color: accentColor,
                border: `1px solid ${accentColor}35`,
              }}
            >
              {badge}
            </span>
          )}
          {icon && <span className="text-taupe">{icon}</span>}
        </div>
      </div>

      <div className="my-1.5">
        <span className="font-display text-xl sm:text-2xl font-bold tracking-tight text-ivory tabular-nums leading-tight">
          {value}
        </span>
      </div>

      {caption && (
        <span className="text-xs text-taupe mt-0.5 leading-normal">
          {caption}
        </span>
      )}
    </div>
  );
};
