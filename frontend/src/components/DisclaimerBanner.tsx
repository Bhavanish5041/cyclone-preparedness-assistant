import React from 'react';
import { AlertCircle, Info, ShieldAlert } from 'lucide-react';

interface DisclaimerBannerProps {
  title?: string;
  text: string;
  variant?: 'warning' | 'info' | 'critical';
  className?: string;
}

export const DisclaimerBanner: React.FC<DisclaimerBannerProps> = ({
  title = 'Assessment Scope & Data Limitations',
  text,
  variant = 'info',
  className = '',
}) => {
  const styles = {
    warning: {
      bg: 'bg-amber-950/25',
      border: 'border-amber-700/40',
      text: 'text-amber-200/90',
      titleColor: 'text-amber-300',
      icon: <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />,
    },
    critical: {
      bg: 'bg-red-950/25',
      border: 'border-red-700/40',
      text: 'text-red-200/90',
      titleColor: 'text-red-300',
      icon: <ShieldAlert className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />,
    },
    info: {
      bg: 'bg-chocolate/40',
      border: 'border-mocha',
      text: 'text-taupe',
      titleColor: 'text-ivory',
      icon: <Info className="w-4 h-4 text-bronze mt-0.5 flex-shrink-0" />,
    },
  }[variant];

  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg border text-xs sm:text-[13px] leading-relaxed ${styles.bg} ${styles.border} ${styles.text} ${className}`}
    >
      {styles.icon}
      <div className="space-y-0.5 min-w-0">
        <span className={`font-display text-xs font-semibold tracking-normal block ${styles.titleColor}`}>
          {title}
        </span>
        <p className="text-ivory/85 font-normal leading-relaxed">
          {text}
        </p>
      </div>
    </div>
  );
};
