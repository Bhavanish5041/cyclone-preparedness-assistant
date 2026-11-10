import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  CheckCircle2,
  Cpu,
  Layers,
  ShieldAlert,
} from 'lucide-react';
import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchClustersAnalytics, fetchModelPerformance } from '../services/api';
import { Header } from '../components/Header';
import { Card } from '../components/Card';
import { KpiCard } from '../components/KpiCard';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorState } from '../components/ErrorState';
import { formatPercent } from '../utils/formatting';
import { CLUSTER_COLORS } from '../utils/constants';

export const AnalyticsPage: React.FC = () => {
  const modelQuery = useQuery({
    queryKey: ['modelPerformance'],
    queryFn: fetchModelPerformance,
  });

  const clusterQuery = useQuery({
    queryKey: ['clustersAnalytics'],
    queryFn: fetchClustersAnalytics,
  });

  if (modelQuery.isLoading || clusterQuery.isLoading) {
    return <LoadingSpinner message="Loading surrogate model diagnostics and archetype catalog..." />;
  }

  if (modelQuery.error || clusterQuery.error) {
    return (
      <ErrorState
        title="Failed to load model diagnostics"
        message={
          (modelQuery.error instanceof Error && modelQuery.error.message) ||
          (clusterQuery.error instanceof Error && clusterQuery.error.message) ||
          'Unknown analytical error'
        }
        onRetry={() => {
          modelQuery.refetch();
          clusterQuery.refetch();
        }}
      />
    );
  }

  const model = modelQuery.data!;
  const clusters = clusterQuery.data!;

  // Confusion matrix classes & grid
  const cmClasses = model.test_confusion_matrix.classes;
  const cmMatrix = model.test_confusion_matrix.matrix;

  // Archetype chart data
  const archetypeChartData = clusters.clusters.map((c) => ({
    name: `Cluster ${c.cluster_id}`,
    label: c.label,
    Healthcare: c.dimension_means.healthcare_deficit || 0,
    Demographic: c.dimension_means.demographic_vulnerability || 0,
    Infrastructure: c.dimension_means.infrastructure_deficit || 0,
    Coastal: c.dimension_means.coastal_exposure || 0,
  }));

  return (
    <div className="space-y-6">
      <Header
        title="Analytics & Model Intelligence"
        subtitle="Empirical evaluation of the Logistic Regression surrogate model and K-Means vulnerability archetypes."
        badge="ML Diagnostics"
      />

      <DisclaimerBanner
        title="Model Purpose & Operational Scope"
        text={model.disclaimer}
        variant="warning"
      />

      {/* Model Performance High-Level KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Test Set Accuracy"
          value={formatPercent(model.test_metrics.accuracy * 100)}
          caption={`Hold-out evaluation (N=${model.split_info.test_size || 282})`}
          badge="Hold-out"
          accentColor="#4EA5D9"
          icon={<CheckCircle2 className="w-4 h-4 text-[#4EA5D9]" />}
        />
        <KpiCard
          title="Macro F1 Score"
          value={formatPercent(model.test_metrics.macro_f1 * 100)}
          caption="Unweighted across High, Med, Low"
          badge="Surrogate"
          accentColor="#E5983B"
          icon={<Activity className="w-4 h-4 text-[#E5983B]" />}
        />
        <KpiCard
          title="Overall Concordance"
          value={formatPercent(model.concordance_rate_overall)}
          caption="AHP tier vs ML surrogate match across 1,879"
          badge="Agreement"
          accentColor="#48A976"
          icon={<ShieldAlert className="w-4 h-4 text-[#48A976]" />}
        />
        <KpiCard
          title="Input Features"
          value={`${model.features_count} Features`}
          caption="Strictly ordered canonical schema"
          badge="Scikit-Learn"
          accentColor="#9D6DE6"
          icon={<Cpu className="w-4 h-4 text-[#9D6DE6]" />}
        />
      </div>

      {/* Confusion Matrix and Split Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
        {/* Test Confusion Matrix */}
        <Card
          title="Hold-Out Test Confusion Matrix"
          subtitle="Rows: Authoritative AHP Class • Columns: Predicted Surrogate Tier (N=282)"
          badge="Confusion Matrix"
        >
          <div className="overflow-x-auto py-2">
            <table className="w-full text-center text-xs border-collapse">
              <thead>
                <tr className="border-b border-chocolate">
                  <th className="p-3 text-left text-taupe font-mono text-[11px] uppercase">True \ Pred</th>
                  {cmClasses.map((cls) => (
                    <th key={cls} className="p-3 font-semibold text-ivory font-mono text-[11px]">
                      Pred {cls}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-chocolate/60">
                {cmClasses.map((rowClass, rIdx) => (
                  <tr key={rowClass}>
                    <td className="p-3 text-left font-medium text-ivory/90">
                      True {rowClass}
                    </td>
                    {cmClasses.map((colClass, cIdx) => {
                      const count = cmMatrix[rIdx]?.[cIdx] ?? 0;
                      const isDiagonal = rIdx === cIdx;
                      return (
                        <td
                          key={colClass}
                          className={`p-3 font-mono font-semibold tabular-nums ${
                            isDiagonal
                              ? 'bg-[#48A976]/20 text-[#48A976] border border-[#48A976]/40'
                              : count > 0
                              ? 'bg-[#E5983B]/20 text-[#E5983B] border border-[#E5983B]/40'
                              : 'text-taupe/40'
                          }`}
                        >
                          {count}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[11px] text-taupe mt-3 pt-2.5 border-t border-chocolate/60 leading-normal">
            Diagonal cells represent concordant classifications. Off-diagonal cells represent surrogate variance.
          </div>
        </Card>

        {/* Train / Validation / Test Breakdown */}
        <Card
          title="Stratified Train / Val / Test Performance"
          subtitle="70% Train (1,315) • 15% Val (282) • 15% Test (282)"
          badge="Model Splits"
        >
          <div className="space-y-4 text-xs">
            <div className="p-3.5 rounded bg-roast border border-chocolate space-y-2">
              <div className="flex justify-between items-center py-1 border-b border-chocolate/50">
                <span className="text-taupe">Training Set Accuracy:</span>
                <span className="font-mono font-semibold text-ivory tabular-nums">
                  {formatPercent(model.train_metrics.accuracy * 100)}
                </span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-chocolate/50">
                <span className="text-taupe">Validation Set Accuracy:</span>
                <span className="font-mono font-semibold text-ivory tabular-nums">
                  {formatPercent(model.val_metrics.accuracy * 100)}
                </span>
              </div>
              <div className="flex justify-between items-center py-1">
                <span className="text-taupe">Hold-Out Test Accuracy:</span>
                <span className="font-mono font-semibold text-[#4EA5D9] tabular-nums">
                  {formatPercent(model.test_metrics.accuracy * 100)}
                </span>
              </div>
            </div>

            <div className="text-[11px] text-taupe leading-relaxed bg-roast/60 p-3 rounded border border-chocolate">
              <span className="font-semibold text-ivory block mb-1">Architecture Specification:</span>
              Pipeline: <code className="text-bronze font-mono">RobustScaler(quantile_range=[25.0, 75.0])</code> &rarr;{' '}
              <code className="text-[#E5983B] font-mono">LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')</code>.
            </div>
          </div>
        </Card>
      </div>

      {/* Discovered Vulnerability Archetypes (k=4) */}
      <div>
        <div className="flex items-center justify-between mb-3.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-bronze" />
            <h3 className="font-display text-base font-semibold text-ivory">
              Vulnerability Archetypes (K-Means k=4)
            </h3>
          </div>
          <span className="text-xs font-mono text-taupe">622 Units Assigned (Dakshina Kannada & Udupi)</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {clusters.clusters.map((c) => {
            const color = CLUSTER_COLORS[c.cluster_id]?.hex || '#C18B58';
            return (
              <Card key={c.cluster_id} className="flex flex-col justify-between hover:border-mocha transition-all">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className="text-xs font-bold font-mono px-2 py-0.5 rounded"
                      style={{ backgroundColor: `${color}18`, color, border: `1px solid ${color}40` }}
                    >
                      Cluster {c.cluster_id}
                    </span>
                    <span className="text-xs font-mono text-taupe tabular-nums">{c.count} Units</span>
                  </div>

                  <h4 className="font-display font-semibold text-ivory text-sm mb-1.5">{c.label}</h4>
                  <p className="text-xs text-taupe leading-relaxed mb-3">
                    {c.detailed_description}
                  </p>
                </div>

                <div className="pt-2.5 border-t border-chocolate text-xs">
                  <span className="text-taupe block text-[11px] font-mono uppercase">Intervention Focus:</span>
                  <span className="text-ivory/90 block mt-0.5 text-xs font-normal">
                    {c.primary_intervention_need}
                  </span>
                </div>
              </Card>
            );
          })}
        </div>

        {/* Archetype Dimension Deficit Comparison Chart */}
        <Card
          title="Cross-Archetype Dimension Deficit Comparison"
          subtitle="Mean normalized deficit scores across Healthcare, Demographic, Infrastructure, and Coastal dimensions"
          badge="Deficit Radar / Comparison"
        >
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsBarChart data={archetypeChartData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30221B" />
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
                  domain={[0, 1]}
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
                  formatter={(val) => <span className="text-xs text-taupe font-medium">{val} Deficit</span>}
                />
                <Bar dataKey="Healthcare" fill="#E05D52" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Demographic" fill="#E5983B" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Infrastructure" fill="#9D6DE6" radius={[2, 2, 0, 0]} />
                <Bar dataKey="Coastal" fill="#4EA5D9" radius={[2, 2, 0, 0]} />
              </RechartsBarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Feature Schema Catalog */}
      <Card
        title="Surrogate Model 26-Feature Canonical Schema"
        subtitle="Ordered feature vector fed to scikit-learn pipeline for tier classification"
        badge="Canonical 26-Vector"
      >
        <div className="flex flex-wrap gap-1.5">
          {model.features.map((feat, idx) => (
            <span
              key={feat}
              className="text-[11px] font-mono px-2.5 py-1 rounded bg-roast border border-chocolate text-ivory/80"
            >
              <span className="text-taupe mr-1">{idx + 1}.</span>
              {feat}
            </span>
          ))}
        </div>
      </Card>
    </div>
  );
};
