import React from 'react';
import { BAND_COLORS, TIER_COLORS } from '../utils/constants';

interface BadgeProps {
  type: 'band' | 'tier' | 'cluster' | 'custom';
  value: string | number;
  label?: string;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  type,
  value,
  label,
  className = '',
}) => {
  if (type === 'band') {
    const bandKey = String(value);
    const bandStyle = BAND_COLORS[bandKey] || {
      bg: 'bg-chocolate',
      text: 'text-ivory',
      border: 'border-mocha',
      label: bandKey,
    };
    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${bandStyle.bg} ${bandStyle.text} ${bandStyle.border} ${className}`}
      >
        {label || bandStyle.label}
      </span>
    );
  }

  if (type === 'tier') {
    const tierKey = String(value);
    const tierStyle = TIER_COLORS[tierKey] || {
      bg: 'bg-chocolate',
      text: 'text-ivory',
      border: 'border-mocha',
    };
    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${tierStyle.bg} ${tierStyle.text} ${tierStyle.border} ${className}`}
      >
        {label || `${tierKey} Tier`}
      </span>
    );
  }

  if (type === 'cluster') {
    const clusterId = Number(value);
    const isUnassigned = isNaN(clusterId) || clusterId < 0;
    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${
          isUnassigned
            ? 'bg-chocolate text-taupe border-mocha'
            : 'bg-mocha/70 text-bronze-light border-walnut'
        } ${className}`}
      >
        {isUnassigned ? 'Unassigned' : `Cluster ${clusterId}`}
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-chocolate text-ivory border border-mocha ${className}`}
    >
      {label || String(value)}
    </span>
  );
};
