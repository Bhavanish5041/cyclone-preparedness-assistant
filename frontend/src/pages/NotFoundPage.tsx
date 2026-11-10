import React from 'react';
import { Link } from 'react-router-dom';
import { Home } from 'lucide-react';
import { Card } from '../components/Card';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="py-24 text-center max-w-md mx-auto">
      <Card className="p-8">
        <h1 className="text-4xl font-bold font-mono text-sky-400 mb-2">404</h1>
        <h2 className="text-lg font-semibold text-slate-100 mb-2">Page Not Found</h2>
        <p className="text-xs text-brand-400 mb-6 leading-relaxed">
          The requested intelligence view or administrative unit route could not be found in the system registry.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-sky-600 hover:bg-sky-500 text-white transition-colors"
        >
          <Home className="w-4 h-4" />
          Return to Command Center
        </Link>
      </Card>
    </div>
  );
};
