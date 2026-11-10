import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Bar,
  BarChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchDistricts } from '../services/api';
import { Header } from '../components/Header';
import { Card } from '../components/Card';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorState } from '../components/ErrorState';
import { formatNumber } from '../utils/formatting';
import { BAND_COLORS } from '../utils/constants';

export const DistrictsPage: React.FC = () => {
  const [selectedDistrictName, setSelectedDistrictName] = useState<string>('Dakshina Kannada');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['districtsList'],
    queryFn: fetchDistricts,
  });

  if (isLoading) {
    return <LoadingSpinner message="Calculating comparative district analytics across study area..." />;
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Failed to load district intelligence"
        message={error instanceof Error ? error.message : 'Unknown district error'}
        onRetry={() => refetch()}
      />
    );
  }

  const districts = data.districts || [];
  const activeDistrict = districts.find((d) => d.district_name === selectedDistrictName) || districts[0];

  // Chart data: Priority bands across all 3 districts
  const districtBandsChartData = districts.map((d) => ({
    name: d.district_name,
    'Band 1 Critical': d.band_counts.BAND_1_CRITICAL || 0,
    'Band 2 Specialized': d.band_counts.BAND_2_SPECIALIZED || 0,
    'Band 3 Moderate': d.band_counts.BAND_3_MODERATE || 0,
    'Band 4 Adequate': d.band_counts.BAND_4_ADEQUATE || 0,
  }));

  // Chart data: Healthcare & Shelter Distances
  const spatialAccessChartData = districts.map((d) => ({
    name: d.district_name,
    'Shelter Distance (km)': d.mean_dist_nearest_shelter_km,
    'Hospital Distance (km)': d.mean_dist_nearest_hospital_km,
    'Shoreline Distance (km)': d.mean_dist_coast_km,
  }));

  return (
    <div className="space-y-6">
      <Header
        title="Comparative District Intelligence"
        subtitle="Cross-district analytical benchmarking across Dakshina Kannada, Udupi, and Uttara Kannada."
        badge="District Intelligence"
      />

      <DisclaimerBanner
        title="Metric Distinction Notice"
        text="AHP Preparedness Tiers (structural capacity) and Operational Priority Bands (resource allocation) are distinct analytical formulations. Never conflate an AHP Medium classification with a Band 2 prioritization."
        variant="info"
      />

      {/* 3 District Comparative Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {districts.map((d) => {
          const isSelected = d.district_name === activeDistrict.district_name;
          return (
            <div
              key={d.district_name}
              onClick={() => setSelectedDistrictName(d.district_name)}
              className={`cursor-pointer rounded-lg p-5 border transition-all ${
                isSelected
                  ? 'bg-espresso border-bronze ring-1 ring-bronze/40 shadow-xl'
                  : 'bg-espresso/80 border-chocolate hover:border-mocha'
              }`}
            >
              <div className="flex items-center justify-between mb-3 pb-2.5 border-b border-chocolate">
                <h3 className="font-display font-bold text-ivory text-base">{d.district_name}</h3>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-roast text-taupe border border-chocolate">
                  {formatNumber(d.total_units)} Units
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-chocolate/50">
                  <span className="text-taupe">Mean AHP Score:</span>
                  <span className="font-mono font-semibold text-[#4EA5D9] tabular-nums">
                    {d.mean_preparedness_score} / 100
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-chocolate/50">
                  <span className="text-taupe">Band 1 Critical Units:</span>
                  <span className="font-mono font-semibold text-[#E05D52] tabular-nums">
                    {d.band_counts.BAND_1_CRITICAL || 0}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-chocolate/50">
                  <span className="text-taupe">Shelter Proximity (Avg):</span>
                  <span className="font-mono text-ivory/90 tabular-nums">
                    {d.mean_dist_nearest_shelter_km} km
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-taupe">Pucca Road Connectivity:</span>
                  <span className="font-mono text-ivory/90 tabular-nums">
                    {d.pct_pucca_road}%
                  </span>
                </div>
              </div>

              <div className="mt-3 pt-2.5 border-t border-chocolate text-right">
                <span className="text-[11px] text-bronze font-medium font-mono">
                  {isSelected ? 'Currently Selected' : 'Click to inspect taluks'} &rarr;
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparative Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
        {/* Priority Bands by District */}
        <Card
          title="Priority Band Distribution by District"
          subtitle="Comparative allocation across Band 1 Critical through Band 4 Adequate"
          badge="Allocation Breakdown"
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={districtBandsChartData} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
                <XAxis
                  dataKey="name"
                  stroke="#76503A"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#30221B' }}
                  tick={{ fill: '#A98F7C' }}
                />
                <YAxis
                  stroke="#76503A"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#30221B' }}
                  tick={{ fill: '#A98F7C' }}
                />
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
                  verticalAlign="top"
                  height={36}
                  formatter={(val) => <span className="text-xs text-taupe font-medium">{val}</span>}
                />
                <Bar dataKey="Band 1 Critical" fill={BAND_COLORS.BAND_1_CRITICAL.hex} stackId="a" />
                <Bar dataKey="Band 2 Specialized" fill={BAND_COLORS.BAND_2_SPECIALIZED.hex} stackId="a" />
                <Bar dataKey="Band 3 Moderate" fill={BAND_COLORS.BAND_3_MODERATE.hex} stackId="a" />
                <Bar dataKey="Band 4 Adequate" fill={BAND_COLORS.BAND_4_ADEQUATE.hex} stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Spatial Access & Proximity */}
        <Card
          title="Spatial Proximity & Facility Access Comparison"
          subtitle="Average geodesic distances in kilometers to key survival assets"
          badge="Geospatial Proximity"
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={spatialAccessChartData} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
                <XAxis
                  dataKey="name"
                  stroke="#76503A"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#30221B' }}
                  tick={{ fill: '#A98F7C' }}
                />
                <YAxis
                  stroke="#76503A"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#30221B' }}
                  tick={{ fill: '#A98F7C' }}
                />
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
                  verticalAlign="top"
                  height={36}
                  formatter={(val) => <span className="text-xs text-taupe font-medium">{val}</span>}
                />
                <Bar dataKey="Shelter Distance (km)" fill="#E5983B" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Hospital Distance (km)" fill="#E05D52" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Shoreline Distance (km)" fill="#4EA5D9" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Selected District Deep-Dive & Taluk Breakdown */}
      {activeDistrict && (
        <Card
          title={`${activeDistrict.district_name}: Detailed Sub-District Breakdown`}
          subtitle="Taluk-level aggregations and operational priority band distribution"
          badge="Taluk Level"
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 mb-5 text-center text-xs">
            <div className="p-3.5 rounded bg-roast border border-chocolate">
              <span className="text-taupe block text-[11px]">Total Units</span>
              <span className="text-xl font-bold font-mono text-ivory mt-0.5 block tabular-nums">
                {formatNumber(activeDistrict.total_units)}
              </span>
            </div>
            <div className="p-3.5 rounded bg-roast border border-chocolate">
              <span className="text-taupe block text-[11px]">Score Range</span>
              <span className="text-xl font-bold font-mono text-ivory mt-0.5 block tabular-nums">
                {activeDistrict.min_preparedness_score} – {activeDistrict.max_preparedness_score}
              </span>
            </div>
            <div className="p-3.5 rounded bg-roast border border-chocolate">
              <span className="text-taupe block text-[11px]">Census Population</span>
              <span className="text-xl font-bold font-mono text-ivory mt-0.5 block tabular-nums">
                {formatNumber(activeDistrict.total_population)}
              </span>
            </div>
            <div className="p-3.5 rounded bg-roast border border-chocolate">
              <span className="text-taupe block text-[11px]">Drinking Water Security</span>
              <span className="text-xl font-bold font-mono text-ivory mt-0.5 block tabular-nums">
                {activeDistrict.pct_drinking_water_secure}%
              </span>
            </div>
          </div>

          {activeDistrict.subdistricts && activeDistrict.subdistricts.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-roast border-b border-chocolate text-taupe font-mono uppercase text-[10px] tracking-wider">
                    <th className="p-3">Taluk / Subdistrict</th>
                    <th className="p-3 text-right">Total Units</th>
                    <th className="p-3 text-right">Mean AHP Score</th>
                    <th className="p-3 text-right">Band 1 Critical</th>
                    <th className="p-3 text-right">Band 2 Specialized</th>
                    <th className="p-3 text-right">Band 3 Moderate</th>
                    <th className="p-3 text-right">Band 4 Adequate</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-chocolate/60">
                  {activeDistrict.subdistricts.map((sub) => (
                    <tr key={sub.subdistrict_name} className="hover:bg-chocolate/20 transition-colors">
                      <td className="p-3 font-medium text-ivory">{sub.subdistrict_name}</td>
                      <td className="p-3 text-right font-mono text-ivory/80 tabular-nums">{sub.total_units}</td>
                      <td className="p-3 text-right font-mono font-semibold text-[#4EA5D9] tabular-nums">
                        {sub.mean_preparedness_score}
                      </td>
                      <td className="p-3 text-right font-mono font-semibold text-[#E05D52] tabular-nums">
                        {sub.band_counts.BAND_1_CRITICAL || 0}
                      </td>
                      <td className="p-3 text-right font-mono text-[#E5983B] tabular-nums">
                        {sub.band_counts.BAND_2_SPECIALIZED || 0}
                      </td>
                      <td className="p-3 text-right font-mono text-[#48A976] tabular-nums">
                        {sub.band_counts.BAND_3_MODERATE || 0}
                      </td>
                      <td className="p-3 text-right font-mono text-[#4EA5D9] tabular-nums">
                        {sub.band_counts.BAND_4_ADEQUATE || 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-xs text-taupe text-center py-4">
              Sub-district breakdowns available in detailed district dossier.
            </div>
          )}
        </Card>
      )}
    </div>
  );
};
