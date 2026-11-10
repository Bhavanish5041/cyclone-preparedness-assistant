import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { KpiCard } from '../components/KpiCard';
import { Badge } from '../components/Badge';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { Card } from '../components/Card';
import { formatNumber, formatPercent } from '../utils/formatting';

describe('UI Component Library Tests', () => {
  it('renders KpiCard with title, formatted value, and caption', () => {
    render(
      <KpiCard
        title="Total Study Units"
        value={formatNumber(1879)}
        caption="100% Accounted Units"
        badge="Authoritative"
        accentColor="#0284C7"
      />
    );

    expect(screen.getByText('Total Study Units')).toBeInTheDocument();
    expect(screen.getByText('1,879')).toBeInTheDocument();
    expect(screen.getByText('100% Accounted Units')).toBeInTheDocument();
    expect(screen.getByText('Authoritative')).toBeInTheDocument();
  });

  it('renders Badge for Priority Band 1 Critical correctly', () => {
    render(<Badge type="band" value="BAND_1_CRITICAL" />);
    expect(screen.getByText('Band 1: Critical')).toBeInTheDocument();
  });

  it('renders Badge for AHP Preparedness Tier correctly', () => {
    render(<Badge type="tier" value="High" />);
    expect(screen.getByText('High Tier')).toBeInTheDocument();
  });

  it('renders Badge for Vulnerability Cluster correctly', () => {
    render(<Badge type="cluster" value={3} />);
    expect(screen.getByText('Cluster 3')).toBeInTheDocument();
  });

  it('renders DisclaimerBanner with text and custom title', () => {
    const notice = 'AHP assessments hold absolute operational precedence for emergency allocations.';
    render(
      <DisclaimerBanner
        title="Scientific Safeguard"
        text={notice}
        variant="warning"
      />
    );

    expect(screen.getByText('Scientific Safeguard')).toBeInTheDocument();
    expect(screen.getByText(notice)).toBeInTheDocument();
  });

  it('renders Card container with title and children', () => {
    render(
      <Card title="Analytical Module" subtitle="Sub-header text">
        <div data-testid="child-content">Active analytical output</div>
      </Card>
    );

    expect(screen.getByText('Analytical Module')).toBeInTheDocument();
    expect(screen.getByText('Sub-header text')).toBeInTheDocument();
    expect(screen.getByTestId('child-content')).toBeInTheDocument();
  });

  it('formats numbers and percentages accurately', () => {
    expect(formatNumber(1879)).toBe('1,879');
    expect(formatNumber(0)).toBe('0');
    expect(formatPercent(94.326)).toBe('94.3%');
    expect(formatPercent(8.249, 2)).toBe('8.25%');
  });
});
