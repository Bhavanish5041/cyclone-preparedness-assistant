import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  Bot,
  Compass,
  Layers,
  LayoutDashboard,
  Map,
  Shield,
  X,
} from 'lucide-react';

interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

const navigationItems = [
  {
    name: 'Command Center',
    path: '/',
    icon: LayoutDashboard,
    badge: '1,879',
  },
  {
    name: 'Geospatial Intelligence',
    path: '/map',
    icon: Map,
    badge: 'GIS',
  },
  {
    name: 'Community Explorer',
    path: '/communities',
    icon: Compass,
  },
  {
    name: 'District Intelligence',
    path: '/districts',
    icon: BarChart3,
  },
  {
    name: 'Analytics & Models',
    path: '/analytics',
    icon: Activity,
  },
  {
    name: 'Preparedness Intelligence Assistant',
    shortName: 'AI Decision Assistant',
    path: '/assistant',
    icon: Bot,
    badge: 'Grounded',
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ isOpen = false, onClose }) => {
  return (
    <>
      {/* Mobile/Tablet Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-30 lg:hidden"
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed lg:sticky top-0 left-0 h-screen w-64 bg-espresso border-r border-chocolate flex flex-col justify-between flex-shrink-0 select-none z-40 transition-transform duration-200 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Brand & Organization Header */}
        <div>
          <div className="p-5 border-b border-chocolate flex items-center justify-between">
            <NavLink to="/" onClick={onClose} className="flex items-center gap-3 group">
              <div className="w-8 h-8 rounded bg-chocolate border border-mocha flex items-center justify-center text-bronze group-hover:border-bronze transition-colors">
                <Shield className="w-4 h-4" />
              </div>
              <div>
                <div className="font-display text-sm font-bold tracking-tight text-ivory flex items-center gap-1.5">
                  <span>CYCLONE INTEL</span>
                </div>
                <div className="text-[10px] text-bronze font-mono uppercase tracking-wider">
                  Coastal Karnataka DSS
                </div>
              </div>
            </NavLink>

            {onClose && (
              <button
                onClick={onClose}
                className="lg:hidden p-1.5 text-taupe hover:text-ivory hover:bg-chocolate rounded transition-colors"
                aria-label="Close menu"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Navigation Links */}
          <nav className="p-3.5 space-y-1">
            <div className="px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider text-taupe-muted">
              System Modules
            </div>
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  onClick={onClose}
                  className={({ isActive }) =>
                    `flex items-center justify-between px-3 py-2 rounded text-[13px] font-medium transition-colors ${
                      isActive
                        ? 'bg-chocolate text-ivory border-l-2 border-bronze font-semibold shadow-sm'
                        : 'text-taupe hover:bg-chocolate/60 hover:text-ivory'
                    }`
                  }
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <Icon className="w-4 h-4 flex-shrink-0 text-bronze" />
                    <span className="truncate">{item.shortName || item.name}</span>
                  </div>
                  {item.badge && (
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-mocha/80 text-taupe border border-walnut/60 flex-shrink-0">
                      {item.badge}
                    </span>
                  )}
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* System Baseline Footer */}
        <div className="p-4 border-t border-chocolate bg-roast/60 text-xs">
          <div className="text-[10px] font-mono uppercase tracking-wider text-taupe-muted mb-2 flex items-center gap-1.5">
            <Layers className="w-3 h-3 text-bronze" />
            <span>Operational Baseline</span>
          </div>
          <div className="space-y-1.5 text-taupe text-[11px]">
            <div className="flex justify-between items-center">
              <span>Study Units:</span>
              <span className="font-mono text-ivory font-medium tabular-nums">1,879 Units</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Decision Core:</span>
              <span className="text-bronze font-mono">Deterministic AHP</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Provenance:</span>
              <span className="text-ivory">Census 2011 & GIS</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
