import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowRight,
  Building,
  HeartPulse,
  Home,
  MapPin,
  Shield,
  Users,
  Waves,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchDashboardOverview } from '../services/api';
import { Card } from '../components/Card';
import { KpiCard } from '../components/KpiCard';
import { Header } from '../components/Header';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorState } from '../components/ErrorState';
import { formatNumber, formatPercent } from '../utils/formatting';
import { BAND_COLORS, TIER_COLORS } from '../utils/constants';

export const DashboardPage: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboardOverview'],
    queryFn: fetchDashboardOverview,
  });

  if (isLoading) {
    return <LoadingSpinner message="Aggregating regional preparedness baseline across 1,879 units..." />;
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Failed to load dashboard overview"
        message={error instanceof Error ? error.message : 'Unknown error occurred'}
        onRetry={() => refetch()}
      />
    );
  }

  const b1 = data.priority_band_counts.BAND_1_CRITICAL || 0;
  const b2 = data.priority_band_counts.BAND_2_SPECIALIZED || 0;
  const b3 = data.priority_band_counts.BAND_3_MODERATE || 0;
  const b4 = data.priority_band_counts.BAND_4_ADEQUATE || 0;
  const total = data.total_units;

  // Chart data: Tier Distribution
  const tierChartData = [
    { name: 'High Tier (>60)', count: data.preparedness_tier_counts.High || 0, color: TIER_COLORS.High.hex },
    { name: 'Medium Tier (40–60)', count: data.preparedness_tier_counts.Medium || 0, color: TIER_COLORS.Medium.hex },
    { name: 'Low Tier (<40)', count: data.preparedness_tier_counts.Low || 0, color: TIER_COLORS.Low.hex },
  ];

  // Chart data: Priority Bands
  const bandChartData = [
    { name: 'Band 1: Critical', count: b1, color: BAND_COLORS.BAND_1_CRITICAL.hex, desc: 'Evacuation Hardening' },
    { name: 'Band 2: Specialized', count: b2, color: BAND_COLORS.BAND_2_SPECIALIZED.hex, desc: 'Sector Deficit Triggers' },
    { name: 'Band 3: Moderate', count: b3, color: BAND_COLORS.BAND_3_MODERATE.hex, desc: 'Baseline Monitoring' },
    { name: 'Band 4: Adequate', count: b4, color: BAND_COLORS.BAND_4_ADEQUATE.hex, desc: 'High Baseline Resilience' },
  ];

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <Header
        title="Regional Preparedness Overview"
        subtitle="An integrated view of community preparedness, infrastructure access and spatial vulnerability across coastal Karnataka."
        badge="Command Center"
      />

      {/* Methodological Scope Disclaimer */}
      <DisclaimerBanner
        title="Assessment Scope & Analytical Governance"
        text="This platform provides spatial decision support across 1,879 administrative units in Dakshina Kannada, Udupi, and Uttara Kannada. Deterministic AHP scores govern operational prioritization; machine learning surrogates and vulnerability clusters provide empirical baselines. Demographic and infrastructure metrics reflect Census 2011 authoritative data."
        variant="info"
      />

      {/* Primary KPI Metrics Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5 sm:gap-4">
        <KpiCard
          title="Administrative Units"
          value={formatNumber(data.total_units)}
          caption="100% Surveyed Revenue Units"
          badge="Total"
          accentColor="#C18B58"
          icon={<Building className="w-4 h-4 text-taupe" />}
        />
        <KpiCard
          title="Band 1: Critical"
          value={formatNumber(b1)}
          caption={`${formatPercent((b1 / total) * 100)} · Evacuation Hardening`}
          badge="Priority 1"
          accentColor={BAND_COLORS.BAND_1_CRITICAL.hex}
          icon={<AlertTriangle className="w-4 h-4 text-[#E05D52]" />}
        />
        <KpiCard
          title="Band 2: Specialized"
          value={formatNumber(b2)}
          caption={`${formatPercent((b2 / total) * 100)} · Deficit-Triggered`}
          badge="Priority 2"
          accentColor={BAND_COLORS.BAND_2_SPECIALIZED.hex}
          icon={<Shield className="w-4 h-4 text-[#E5983B]" />}
        />
        <KpiCard
          title="Band 3: Moderate"
          value={formatNumber(b3)}
          caption={`${formatPercent((b3 / total) * 100)} · Baseline Monitoring`}
          badge="Priority 3"
          accentColor={BAND_COLORS.BAND_3_MODERATE.hex}
          icon={<Home className="w-4 h-4 text-[#48A976]" />}
        />
        <KpiCard
          title="Band 4: Adequate"
          value={formatNumber(b4)}
          caption={`${formatPercent((b4 / total) * 100)} · High Structural Readiness`}
          badge="Priority 4"
          accentColor={BAND_COLORS.BAND_4_ADEQUATE.hex}
          icon={<Shield className="w-4 h-4 text-[#4EA5D9]" />}
        />
      </div>

      {/* Dominant Visual Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-6">
        {/* Operational Priority Allocation Funnel (7 cols) */}
        <div className="lg:col-span-7">
          <Card
            title="Operational Priority Allocation"
            subtitle="Deterministic resource allocation categories based on AHP composite scores and sector deficit triggers"
            badge="Resource Deployment"
          >
            {/* Visual Funnel Bar */}
            <div className="mb-4">
              <div className="flex h-3 w-full rounded-full overflow-hidden bg-roast border border-chocolate">
                <div style={{ width: `${(b1 / total) * 100}%` }} className="bg-[#E05D52]" title={`Band 1: ${b1} units`} />
                <div style={{ width: `${(b2 / total) * 100}%` }} className="bg-[#E5983B]" title={`Band 2: ${b2} units`} />
                <div style={{ width: `${(b3 / total) * 100}%` }} className="bg-[#48A976]" title={`Band 3: ${b3} units`} />
                <div style={{ width: `${(b4 / total) * 100}%` }} className="bg-[#4EA5D9]" title={`Band 4: ${b4} units`} />
              </div>
              <div className="flex justify-between items-center text-[11px] font-mono text-taupe mt-1.5 px-0.5">
                <span className="text-[#E05D52] font-semibold">Band 1: {formatPercent((b1 / total) * 100)}</span>
                <span className="text-[#E5983B] font-semibold">Band 2: {formatPercent((b2 / total) * 100)}</span>
                <span className="text-[#48A976] font-semibold">Band 3: {formatPercent((b3 / total) * 100)}</span>
                <span className="text-[#4EA5D9] font-semibold">Band 4: {formatPercent((b4 / total) * 100)}</span>
              </div>
            </div>

            {/* Horizontal Bar Chart */}
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={bandChartData}
                  layout="vertical"
                  margin={{ left: 5, right: 30, top: 5, bottom: 5 }}
                >
                  <XAxis
                    type="number"
                    stroke="#76503A"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: '#30221B' }}
                    tick={{ fill: '#A98F7C' }}
                  />
                  <YAxis
                    dataKey="name"
                    type="category"
                    stroke="#A98F7C"
                    fontSize={12}
                    tickLine={false}
                    axisLine={{ stroke: '#30221B' }}
                    width={140}
                    tick={{ fill: '#E9DCCB' }}
                  />
                  <Tooltip
                    cursor={{ fill: '#30221B', opacity: 0.3 }}
                    contentStyle={{
                      backgroundColor: '#211713',
                      borderColor: '#493326',
                      borderRadius: '6px',
                      fontSize: '12px',
                      color: '#E9DCCB',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                    }}
                    itemStyle={{ color: '#E9DCCB' }}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {bandChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Actionable Allocation Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-4 border-t border-chocolate/80 text-center">
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#E05D52] font-bold font-mono block text-base tabular-nums">{b1}</span>
                <span className="text-taupe text-[11px] block mt-0.5">Band 1: Critical</span>
                <span className="text-[10px] text-taupe/70 font-mono">Immediate Stage</span>
              </div>
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#E5983B] font-bold font-mono block text-base tabular-nums">{b2}</span>
                <span className="text-taupe text-[11px] block mt-0.5">Band 2: Specialized</span>
                <span className="text-[10px] text-taupe/70 font-mono">Deficit Targeted</span>
              </div>
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#48A976] font-bold font-mono block text-base tabular-nums">{b3}</span>
                <span className="text-taupe text-[11px] block mt-0.5">Band 3: Moderate</span>
                <span className="text-[10px] text-taupe/70 font-mono">Resilience Monitor</span>
              </div>
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#4EA5D9] font-bold font-mono block text-base tabular-nums">{b4}</span>
                <span className="text-taupe text-[11px] block mt-0.5">Band 4: Adequate</span>
                <span className="text-[10px] text-taupe/70 font-mono">Baseline Sufficient</span>
              </div>
            </div>
          </Card>
        </div>

        {/* Structural AHP Capacity (5 cols) */}
        <div className="lg:col-span-5">
          <Card
            title="AHP Preparedness Tiers"
            subtitle="Deterministic multi-criteria composite scores (>60 High, 40–60 Medium, <40 Low)"
            badge="Intrinsic Capacity"
          >
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={tierChartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                    dataKey="count"
                  >
                    {tierChartData.map((entry, index) => (
                      <Cell key={`tier-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#211713',
                      borderColor: '#493326',
                      borderRadius: '6px',
                      fontSize: '12px',
                      color: '#E9DCCB',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                    }}
                    itemStyle={{ color: '#E9DCCB' }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    formatter={(val) => <span className="text-xs text-taupe font-medium">{val}</span>}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Tier Breakdown Stats */}
            <div className="grid grid-cols-3 gap-2 pt-4 border-t border-chocolate/80 text-center">
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#4EA5D9] font-bold font-mono block text-base tabular-nums">
                  {data.preparedness_tier_counts.High}
                </span>
                <span className="text-taupe text-[11px] block mt-0.5">High Tier (&gt;60)</span>
                <span className="text-[10px] text-taupe/70 font-mono">
                  {formatPercent((data.preparedness_tier_counts.High / total) * 100)}
                </span>
              </div>
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#E5983B] font-bold font-mono block text-base tabular-nums">
                  {data.preparedness_tier_counts.Medium}
                </span>
                <span className="text-taupe text-[11px] block mt-0.5">Medium (40–60)</span>
                <span className="text-[10px] text-taupe/70 font-mono">
                  {formatPercent((data.preparedness_tier_counts.Medium / total) * 100)}
                </span>
              </div>
              <div className="p-2.5 rounded bg-roast/60 border border-chocolate">
                <span className="text-[#E05D52] font-bold font-mono block text-base tabular-nums">
                  {data.preparedness_tier_counts.Low}
                </span>
                <span className="text-taupe text-[11px] block mt-0.5">Low Tier (&lt;40)</span>
                <span className="text-[10px] text-taupe/70 font-mono">
                  {formatPercent((data.preparedness_tier_counts.Low / total) * 100)}
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* District Preparedness Comparison Cards */}
      <div>
        <div className="flex items-center justify-between mb-3.5">
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-bronze" />
            <h3 className="font-display text-base font-semibold text-ivory">
              District Preparedness Comparison
            </h3>
          </div>
          <Link
            to="/districts"
            className="text-xs font-mono text-bronze hover:text-ivory inline-flex items-center gap-1 transition-colors"
          >
            Detailed District Benchmark <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.districts.map((d) => (
            <Card key={d.district_name} className="hover:border-mocha transition-all">
              <div className="flex items-center justify-between mb-3 pb-2.5 border-b border-chocolate/60">
                <span className="font-display font-semibold text-ivory text-base">
                  {d.district_name}
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-roast text-taupe border border-chocolate">
                  {formatNumber(d.total_units)} Units
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between items-center py-1 border-b border-chocolate/40">
                  <span className="text-taupe">Mean AHP Preparedness:</span>
                  <span className="font-mono font-medium text-ivory tabular-nums">
                    {d.mean_preparedness_score} / 100
                  </span>
                </div>
                <div className="flex justify-between items-center py-1 border-b border-chocolate/40">
                  <span className="text-taupe">Band 1 Critical Units:</span>
                  <span className="font-mono font-semibold text-[#E05D52] tabular-nums">
                    {d.band_counts.BAND_1_CRITICAL || 0}
                  </span>
                </div>
                <div className="flex justify-between items-center py-1 border-b border-chocolate/40">
                  <span className="text-taupe">Mean Shelter Distance:</span>
                  <span className="font-mono text-ivory/90 tabular-nums">{d.mean_dist_nearest_shelter_km} km</span>
                </div>
                <div className="flex justify-between items-center py-1 border-b border-chocolate/40">
                  <span className="text-taupe">Mean Hospital Distance:</span>
                  <span className="font-mono text-ivory/90 tabular-nums">{d.mean_dist_nearest_hospital_km} km</span>
                </div>
                <div className="flex justify-between items-center py-1">
                  <span className="text-taupe">Mean Shoreline Distance:</span>
                  <span className="font-mono text-ivory/90 tabular-nums">{d.mean_dist_coast_km} km</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Regional Infrastructure & Demographic Baseline */}
      <Card
        title="Regional Infrastructure & Demographic Baseline"
        subtitle="Authoritative structural access indicators aggregated across Census 2011 records"
        badge="Census 2011"
      >
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 sm:gap-4 text-center">
          <div className="p-4 rounded bg-roast/60 border border-chocolate">
            <Users className="w-4 h-4 mx-auto text-bronze mb-2" />
            <div className="font-display text-2xl font-bold font-mono text-ivory tabular-nums">
              {formatNumber(data.infrastructure_highlights.total_population)}
            </div>
            <div className="text-xs text-taupe mt-1">Census Population</div>
          </div>
          <div className="p-4 rounded bg-roast/60 border border-chocolate">
            <Home className="w-4 h-4 mx-auto text-[#E5983B] mb-2" />
            <div className="font-display text-2xl font-bold font-mono text-ivory tabular-nums">
              {formatNumber(data.infrastructure_highlights.total_households)}
            </div>
            <div className="text-xs text-taupe mt-1">Total Households</div>
          </div>
          <div className="p-4 rounded bg-roast/60 border border-chocolate">
            <Waves className="w-4 h-4 mx-auto text-[#48A976] mb-2" />
            <div className="font-display text-2xl font-bold font-mono text-ivory tabular-nums">
              {formatNumber(data.infrastructure_highlights.coastal_units_under_5km)}
            </div>
            <div className="text-xs text-taupe mt-1">Coastal Units (&lt;5 km)</div>
          </div>
          <div className="p-4 rounded bg-roast/60 border border-chocolate">
            <HeartPulse className="w-4 h-4 mx-auto text-[#E05D52] mb-2" />
            <div className="font-display text-2xl font-bold font-mono text-ivory tabular-nums">
              {data.infrastructure_highlights.pucca_road_pct}%
            </div>
            <div className="text-xs text-taupe mt-1">Pucca Road Connectivity</div>
          </div>
        </div>
      </Card>
    </div>
  );
};
