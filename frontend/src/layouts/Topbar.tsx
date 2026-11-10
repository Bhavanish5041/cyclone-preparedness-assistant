import React from 'react';
import { Database, MapPin, Menu } from 'lucide-react';

interface TopbarProps {
  onToggleSidebar?: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({ onToggleSidebar }) => {
  return (
    <header className="h-14 bg-espresso/95 border-b border-chocolate px-4 sm:px-6 lg:px-8 flex items-center justify-between sticky top-0 backdrop-blur-md z-20">
      <div className="flex items-center gap-3">
        {onToggleSidebar && (
          <button
            onClick={onToggleSidebar}
            className="lg:hidden p-1.5 text-taupe hover:text-ivory hover:bg-chocolate rounded transition-colors"
            aria-label="Open navigation menu"
          >
            <Menu className="w-4 h-4" />
          </button>
        )}

        <div className="flex items-center gap-2 text-xs text-taupe">
          <MapPin className="w-3.5 h-3.5 text-bronze flex-shrink-0" />
          <span className="text-taupe-muted hidden sm:inline">Study Area:</span>
          <span className="font-medium text-ivory">
            Dakshina Kannada, Udupi, Uttara Kannada
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2.5 sm:gap-3 text-xs">
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-chocolate border border-mocha text-ivory font-mono text-[11px]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span>Dataset Available</span>
          <span className="text-taupe hidden md:inline font-mono tabular-nums">(1,879 Units)</span>
        </div>

        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded bg-chocolate/50 border border-mocha/60 text-taupe text-[11px]">
          <Database className="w-3.5 h-3.5 text-bronze" />
          <span>Census 2011 & Spatial Baselines</span>
        </div>
      </div>
    </header>
  );
};
