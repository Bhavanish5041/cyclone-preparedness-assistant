import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Clock,
  Home,
  Info,
  Shield,
} from 'lucide-react';
import { fetchCommunityDetail } from '../services/api';
import { Header } from '../components/Header';
import { Card } from '../components/Card';
import { KpiCard } from '../components/KpiCard';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorState } from '../components/ErrorState';
import { formatNumber, formatPercent } from '../utils/formatting';
import { BAND_COLORS } from '../utils/constants';

export const CommunityDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['communityDetail', id],
    queryFn: () => fetchCommunityDetail(id || ''),
    enabled: !!id,
  });

  if (isLoading) {
    return <LoadingSpinner message={`Retrieving inspection dossier for unit '${id}'...`} />;
  }

  if (error || !data) {
    return (
      <div className="space-y-4">
        <Link
          to="/communities"
          className="inline-flex items-center gap-1.5 text-xs text-taupe hover:text-ivory transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Community Registry
        </Link>
        <ErrorState
          title="Community Dossier Not Found"
          message={error instanceof Error ? error.message : `Could not locate administrative unit '${id}'.`}
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  const formatBandName = (band: string) => {
    switch (band) {
      case 'BAND_1_CRITICAL':
        return 'Band 1: Critical';
      case 'BAND_2_SPECIALIZED':
        return 'Band 2: Specialized';
      case 'BAND_3_MODERATE':
        return 'Band 3: Moderate';
      case 'BAND_4_ADEQUATE':
        return 'Band 4: Adequate';
      default:
        return band.replace('BAND_', 'Band ').replace(/_/g, ' ');
    }
  };

  const bandFormatted = formatBandName(data.priority_band);

  return (
    <div className="space-y-6">
      {/* Navigation Breadcrumb */}
      <div className="flex items-center justify-between">
        <Link
          to="/communities"
          className="inline-flex items-center gap-1.5 text-xs text-taupe hover:text-ivory transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Community Registry
        </Link>
        <div className="text-xs font-mono text-taupe">
          PC11 ID: <span className="text-ivory font-medium">{data.pc11_id}</span> • SHRID:{' '}
          <span className="text-ivory/80">{data.shrid2}</span>
        </div>
      </div>

      <Header
        title={`${data.village_name} (${data.district_name})`}
        subtitle={`Administrative Unit Dossier: Taluk ${data.subdistrict_name || 'N/A'}, Type: ${data.unit_type}, Coordinates: ${data.lat}, ${data.lon}`}
        badge={`Priority Rank #${data.priority_rank}`}
      />

      {/* Authoritative AHP & Operational Priority KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="AHP Preparedness Score"
          value={`${data.preparedness_score} / 100`}
          caption="Authoritative Deterministic Index"
          badge={data.preparedness_tier}
          accentColor="#4EA5D9"
          icon={<Shield className="w-4 h-4 text-[#4EA5D9]" />}
        />
        <KpiCard
          title="Operational Allocation"
          value={bandFormatted}
          caption="Decision-Support Tier"
          badge={`Rank #${data.priority_rank}`}
          accentColor={BAND_COLORS[data.priority_band]?.hex || '#E05D52'}
          icon={<AlertTriangle className="w-4 h-4 text-[#E05D52]" />}
        />
        <KpiCard
          title="Within-Band Rank"
          value={`#${data.band_rank}`}
          caption={`Sorted deterministically inside ${bandFormatted}`}
          badge="Priority Queue"
          accentColor="#E5983B"
          icon={<Clock className="w-4 h-4 text-[#E5983B]" />}
        />
        <KpiCard
          title="Vulnerability Archetype"
          value={data.vulnerability_archetype_label || 'Uttara Kannada Fallback'}
          caption={data.cluster_assignment_status}
          badge={data.vulnerability_archetype_id !== null && data.vulnerability_archetype_id !== undefined ? `Cluster ${data.vulnerability_archetype_id}` : 'Fallback'}
          accentColor="#9D6DE6"
          icon={<Home className="w-4 h-4 text-[#9D6DE6]" />}
        />
      </div>

      {/* Precedence Safeguard Alert */}
      {data.ahp_precedence_note && (
        <DisclaimerBanner
          title="Assessment Scope & Precedence Directive"
          text={data.ahp_precedence_note}
          variant="critical"
        />
      )}

      {/* Grid: 5 Domain Scores & ML Surrogate Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* AHP Domain Scores */}
        <Card
          title="Deterministic AHP Domain Breakdown"
          subtitle="Authoritative structural capacity indicators across 5 disaster management dimensions"
          badge="AHP 5-Domain"
        >
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-ivory font-medium">Medical & Emergency Healthcare</span>
                <span className="font-mono text-ivory/90 tabular-nums">{data.score_medical} / 100</span>
              </div>
              <div className="w-full bg-roast rounded-full h-2 border border-chocolate">
                <div
                  className="bg-[#4EA5D9] h-2 rounded-full"
                  style={{ width: `${Math.min(100, data.score_medical)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-ivory font-medium">Cyclone Shelters & Evacuation Safe Havens</span>
                <span className="font-mono text-ivory/90 tabular-nums">{data.score_shelters} / 100</span>
              </div>
              <div className="w-full bg-roast rounded-full h-2 border border-chocolate">
                <div
                  className="bg-[#E5983B] h-2 rounded-full"
                  style={{ width: `${Math.min(100, data.score_shelters)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-ivory font-medium">Civil Infrastructure & Road Connectivity</span>
                <span className="font-mono text-ivory/90 tabular-nums">{data.score_infrastructure} / 100</span>
              </div>
              <div className="w-full bg-roast rounded-full h-2 border border-chocolate">
                <div
                  className="bg-[#48A976] h-2 rounded-full"
                  style={{ width: `${Math.min(100, data.score_infrastructure)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-ivory font-medium">Early Warning Dissemination & Monitoring</span>
                <span className="font-mono text-ivory/90 tabular-nums">{data.score_early_warning} / 100</span>
              </div>
              <div className="w-full bg-roast rounded-full h-2 border border-chocolate">
                <div
                  className="bg-[#C18B58] h-2 rounded-full"
                  style={{ width: `${Math.min(100, data.score_early_warning)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-ivory font-medium">Socioeconomic Resilience Baseline</span>
                <span className="font-mono text-ivory/90 tabular-nums">{data.score_socioeconomic} / 100</span>
              </div>
              <div className="w-full bg-roast rounded-full h-2 border border-chocolate">
                <div
                  className="bg-[#9D6DE6] h-2 rounded-full"
                  style={{ width: `${Math.min(100, data.score_socioeconomic)}%` }}
                />
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3.5 border-t border-chocolate text-xs text-taupe">
            <span className="font-medium text-ivory">Deficit Rationale:</span>{' '}
            {data.primary_deficit_rationale}
          </div>
        </Card>

        {/* Machine Learning Surrogate Diagnostics */}
        <Card
          title="Machine Learning Surrogate Model Diagnostics"
          subtitle="Logistic Regression classifier approximating AHP tiers (surrogate approximation, not physical hazard risk)"
          badge="ML Classifier"
        >
          <div className="space-y-4">
            <div className="p-3.5 rounded bg-roast border border-chocolate space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-taupe">Predicted Capacity Tier:</span>
                <span className="font-semibold text-ivory">
                  {data.predicted_preparedness_tier || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-taupe">AHP vs ML Agreement:</span>
                <span
                  className={`font-semibold ${
                    data.ahp_agreement === 'FULL_AGREEMENT' ? 'text-[#48A976]' : 'text-[#E5983B]'
                  }`}
                >
                  {data.ahp_agreement || 'EVALUATED'}
                </span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-taupe">Boundary Case Status:</span>
                <span className="font-mono text-ivory">
                  {data.is_boundary_case ? 'YES (High Ambiguity)' : 'NO (Clear Separation)'}
                </span>
              </div>
            </div>

            {/* Class Probabilities */}
            <div className="space-y-2 text-xs">
              <div className="text-taupe font-medium">Surrogate Class Probabilities:</div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-ivory/80">P(High Tier):</span>
                    <span className="font-mono tabular-nums text-ivory">{formatPercent((data.prob_high || 0) * 100)}</span>
                  </div>
                  <div className="w-full bg-roast rounded-full h-1.5 border border-chocolate">
                    <div
                      className="bg-[#4EA5D9] h-1.5 rounded-full"
                      style={{ width: `${(data.prob_high || 0) * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-ivory/80">P(Medium Tier):</span>
                    <span className="font-mono tabular-nums text-ivory">{formatPercent((data.prob_medium || 0) * 100)}</span>
                  </div>
                  <div className="w-full bg-roast rounded-full h-1.5 border border-chocolate">
                    <div
                      className="bg-[#E5983B] h-1.5 rounded-full"
                      style={{ width: `${(data.prob_medium || 0) * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-ivory/80">P(Low Tier):</span>
                    <span className="font-mono tabular-nums text-ivory">{formatPercent((data.prob_low || 0) * 100)}</span>
                  </div>
                  <div className="w-full bg-roast rounded-full h-1.5 border border-chocolate">
                    <div
                      className="bg-[#E05D52] h-1.5 rounded-full"
                      style={{ width: `${(data.prob_low || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="text-[11px] text-taupe leading-relaxed bg-roast/60 p-2.5 rounded border border-chocolate">
              <Info className="w-3.5 h-3.5 text-bronze inline mr-1" />
              {data.ml_disclaimer}
            </div>
          </div>
        </Card>
      </div>

      {/* Operational Response Directives */}
      <Card
        title="Grounded Operational Directives & Action Plan"
        subtitle="Field inspection recommendations triggered deterministically by empirical deficits"
        badge="Field Directives"
      >
        <div className="space-y-2.5">
          {data.actionable_advisories.map((adv, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2.5 p-3 rounded bg-roast/60 border border-chocolate text-xs"
            >
              <CheckCircle2 className="w-4 h-4 text-[#48A976] mt-0.5 flex-shrink-0" />
              <span className="text-ivory/90 leading-relaxed">{adv}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Spatial Proximity & Demographic Baseline Indicators */}
      <Card
        title="Spatial Infrastructure & Demographic Indicators"
        subtitle="Decennial Census 2011 and official healthcare/coastal registry baselines"
        badge="Census 2011"
      >
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3.5 sm:gap-4 text-xs">
          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Distance to Coast</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5 tabular-nums">
              {data.dist_coast_km} km
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">Natural Earth 10m</div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Nearest Cyclone Shelter</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5 tabular-nums">
              {data.dist_nearest_shelter_km} km
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">{data.n_shelters_10km} shelters in 10 km</div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Nearest Apex Hospital</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5 tabular-nums">
              {data.dist_nearest_hospital_km} km
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">MoHFW Directory</div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Nearest PHC / CHC</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5 tabular-nums">
              {data.dist_nearest_phc_km} km
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">
              {data.has_phc_or_chc ? 'In Village' : 'Outside Village'}
            </div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Total Population</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5 tabular-nums">
              {formatNumber(data.population)}
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">Census 2011 Baseline</div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Vulnerable Pop Share</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5 tabular-nums">
              {formatPercent(data.vulnerable_pop_share * 100)}
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">SC + ST Proportion</div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">All-Weather Pucca Road</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5">
              {data.has_pucca_road ? 'Connected' : 'Unpaved'}
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">Civil Connectivity</div>
          </div>

          <div className="p-3.5 rounded bg-roast/60 border border-chocolate">
            <div className="text-taupe">Treated Tap Water</div>
            <div className="text-base font-bold font-mono text-ivory mt-0.5">
              {data.drinking_water_score > 0.5 ? 'Secure' : 'Unrecorded / Well'}
            </div>
            <div className="text-[10px] text-taupe/70 mt-0.5">Score: {data.drinking_water_score}</div>
          </div>
        </div>

        <div className="mt-4 pt-3.5 border-t border-chocolate text-xs text-taupe space-y-1">
          <div>
            <span className="font-semibold text-ivory">Data Vintage Limitations:</span>{' '}
            {data.provenance_and_limitations.operational_limitations}
          </div>
        </div>
      </Card>
    </div>
  );
};
