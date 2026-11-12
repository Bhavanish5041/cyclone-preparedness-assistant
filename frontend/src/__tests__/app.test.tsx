import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { App, clearApiCache } from '../App';

describe('CYCLONE INTEL Unified App Integration', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    clearApiCache();
    window.location.hash = '#/overview';
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const mockApiResponse = (url: string, data: any, status = 200) => {
    return Promise.resolve({
      ok: status >= 200 && status < 300,
      status,
      headers: {
        get: (h: string) => (h.toLowerCase() === 'content-type' ? 'application/json' : null),
      },
      json: () => Promise.resolve(data),
    } as Response);
  };

  it('renders Overview view with authoritative metrics and scientific labels', async () => {
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const u = String(input);
      if (u.includes('/dashboard/overview')) {
        return mockApiResponse(u, {
          total_units: 1879,
          districts_count: 3,
          assigned_clusters_count: 622,
          unassigned_clusters_count: 1257,
          priority_band_counts: {
            BAND_1_CRITICAL: 155,
            BAND_2_SPECIALIZED: 829,
            BAND_3_MODERATE: 299,
            BAND_4_ADEQUATE: 596,
          },
          preparedness_tier_counts: { High: 596, Medium: 1128, Low: 155 },
          cluster_counts: {},
          districts: [
            {
              district_name: 'Dakshina Kannada',
              total_units: 369,
              mean_preparedness_score: 61.06,
              critical_units_count: 2,
              band_counts: { BAND_1_CRITICAL: 2, BAND_2_SPECIALIZED: 123, BAND_3_MODERATE: 45, BAND_4_ADEQUATE: 199 },
            },
            {
              district_name: 'Udupi',
              total_units: 253,
              mean_preparedness_score: 62.45,
              critical_units_count: 0,
              band_counts: { BAND_1_CRITICAL: 0, BAND_2_SPECIALIZED: 80, BAND_3_MODERATE: 50, BAND_4_ADEQUATE: 123 },
            },
            {
              district_name: 'Uttara Kannada',
              total_units: 1257,
              mean_preparedness_score: 50.59,
              critical_units_count: 153,
              band_counts: { BAND_1_CRITICAL: 153, BAND_2_SPECIALIZED: 626, BAND_3_MODERATE: 204, BAND_4_ADEQUATE: 274 },
            },
          ],
          infrastructure_highlights: {
            total_population: 4680000,
            total_households: 1050000,
            mean_dist_coast_km: 19.8,
            mean_dist_nearest_shelter_km: 31.4,
            mean_dist_nearest_hospital_km: 36.2,
            mean_dist_nearest_phc_km: 14.1,
            pucca_road_pct: 78.5,
            coastal_units_under_5km: 249,
          },
          data_disclaimer: 'Authoritative baseline assessment',
        });
      }
      if (u.includes('/analytics/preparedness')) {
        return mockApiResponse(u, {
          total_units: 1879,
          tier_distribution: { High: 596, Medium: 1128, Low: 155 },
          score_summary: { mean: 54.24, median: 53.11, min: 20.14, max: 81.54, std: 10.72 },
          domain_averages: {
            medical: 51.35,
            shelters: 25.26,
            infrastructure: 53.63,
            early_warning: 72.56,
            socioeconomic: 89.81,
          },
          district_averages: {},
          disclaimer: '',
        });
      }
      if (u.includes('/health')) {
        return mockApiResponse(u, { status: 'ok', dataset_loaded: true, total_units: 1879 });
      }
      return mockApiResponse(u, {});
    });

    render(<App />);

    // Scientific label verification
    expect(await screen.findByText('Preparedness Overview')).toBeInTheDocument();
    expect(screen.getByText(/Preparedness Priorities/)).toBeInTheDocument();
    expect(screen.getByText(/Coastal Vulnerability Watch/)).toBeInTheDocument();
    expect(screen.getByText('Live')).toBeInTheDocument();

    // Metric verification
    expect(screen.getByText(/1,879 administrative units across 3 coastal districts/)).toBeInTheDocument();
    expect(screen.getByText('Dakshina Kannada')).toBeInTheDocument();
    expect(screen.getByText('Udupi')).toBeInTheDocument();
    expect(screen.getByText('Uttara Kannada')).toBeInTheDocument();
  });

  it('renders Communities view with table, filters, and drawer dossier on row click', async () => {
    window.location.hash = '#/communities';

    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const u = String(input);
      if (u.includes('/communities?')) {
        return mockApiResponse(u, {
          total: 1,
          page: 1,
          limit: 25,
          total_pages: 1,
          items: [
            {
              area_id: '11-29-569-05522-608691',
              pc11_id: '608691',
              shrid2: '11-29-569-05522-608691',
              village_name: 'Tallur',
              district_name: 'Udupi',
              subdistrict_name: 'Kundapura',
              unit_type: 'VILLAGE',
              lat: 13.6598,
              lon: 74.7104,
              preparedness_score: 74.34,
              preparedness_tier: 'High',
              priority_band: 'BAND_4_ADEQUATE',
              priority_rank: 1780,
              band_rank: 512,
              dist_coast_km: 1.8,
              population: 8942,
            },
          ],
        });
      }
      if (u.includes('/communities/11-29-569-05522-608691')) {
        return mockApiResponse(u, {
          area_id: '11-29-569-05522-608691',
          pc11_id: '608691',
          shrid2: '11-29-569-05522-608691',
          village_name: 'Tallur',
          district_name: 'Udupi',
          subdistrict_name: 'Kundapura',
          unit_type: 'VILLAGE',
          lat: 13.6598,
          lon: 74.7104,
          preparedness_score: 74.34,
          preparedness_tier: 'High',
          priority_band: 'BAND_4_ADEQUATE',
          priority_rank: 1780,
          band_rank: 512,
          score_medical: 71.94,
          score_shelters: 68.59,
          score_infrastructure: 68.0,
          score_early_warning: 82.11,
          score_socioeconomic: 88.58,
          triggers_activated: [],
          primary_deficit_rationale: 'Adequate preparedness infrastructure.',
          actionable_advisories: ['Maintain pre-monsoon storm-surge shelter readiness.'],
          predicted_preparedness_tier: 'High',
          prob_high: 0.9997,
          prob_medium: 0.0003,
          prob_low: 0.0,
          ahp_agreement: 'FULL_AGREEMENT',
          ml_disclaimer:
            'Deterministic AHP composite score maintains absolute operational authority. Logistic Regression serves strictly as an empirical AHP tier surrogate.',
          vulnerability_archetype_label: 'High-Exposure Coastal Rural Belt',
          tauktae_affected: true,
          dist_coast_km: 1.8,
          dist_nearest_shelter_km: 0.9,
          dist_nearest_hospital_km: 5.2,
          population: 8942,
          elevation_mean: 12.5,
          n_shelters_10km: 3,
        });
      }
      if (u.includes('/dashboard/overview')) {
        return mockApiResponse(u, { priority_band_counts: { BAND_1_CRITICAL: 155 } });
      }
      if (u.includes('/health')) {
        return mockApiResponse(u, { status: 'ok' });
      }
      return mockApiResponse(u, {});
    });

    render(<App />);

    expect(await screen.findByText('Unit Priority Register')).toBeInTheDocument();
    expect(await screen.findByText('Tallur')).toBeInTheDocument();
    expect(screen.getByText('74.34')).toBeInTheDocument();

    // Click Tallur to open dossier drawer
    fireEvent.click(screen.getByText('Tallur'));

    // Verify dossier drawer content
    expect(await screen.findByText('Historic Cyclone Tauktae Impact (2021)')).toBeInTheDocument();
    expect(screen.getByText('High-Exposure Coastal Rural Belt')).toBeInTheDocument();
    expect(screen.getByText('Maintain pre-monsoon storm-surge shelter readiness.')).toBeInTheDocument();
    expect(screen.getByText(/Deterministic AHP composite score maintains absolute operational authority/)).toBeInTheDocument();
  });

  it('renders Analytics view with confusion matrix and vulnerability archetypes including unassigned', async () => {
    window.location.hash = '#/analytics';

    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const u = String(input);
      if (u.includes('/analytics/preparedness')) {
        return mockApiResponse(u, {
          total_units: 1879,
          tier_distribution: { High: 596, Medium: 1128, Low: 155 },
          score_summary: { mean: 54.24, median: 53.11, min: 20.14, max: 81.54, std: 10.72 },
          domain_averages: { medical: 51.35, shelters: 25.26, infrastructure: 53.63, early_warning: 72.56, socioeconomic: 89.81 },
          district_averages: {
            'Dakshina Kannada': { medical: 65.86, shelters: 35.12, infrastructure: 60.1, early_warning: 75.2, socioeconomic: 91.2 },
          },
        });
      }
      if (u.includes('/analytics/priority-bands')) {
        return mockApiResponse(u, {
          total_units: 1879,
          band_distribution: { BAND_1_CRITICAL: 155, BAND_2_SPECIALIZED: 829, BAND_3_MODERATE: 299, BAND_4_ADEQUATE: 596 },
          trigger_counts: { coastal_proximity_under_5km: 316, shelter_deficit_under_20: 865 },
        });
      }
      if (u.includes('/analytics/clusters')) {
        return mockApiResponse(u, {
          total_units: 1879,
          assigned_count: 622,
          unassigned_count: 1257,
          clusters: [
            {
              cluster_id: 0,
              label: 'Urban & Peri-Urban Well-Resourced Corridor',
              count: 130,
              primary_intervention_need: 'Stormwater pump maintenance',
              detailed_description: 'Communities with low healthcare deficit.',
            },
          ],
        });
      }
      if (u.includes('/analytics/model-performance')) {
        return mockApiResponse(u, {
          model_name: 'village-preparedness-classifier',
          model_type: 'Multinomial Logistic Regression',
          train_metrics: { accuracy: 0.9589, macro_f1: 0.9352, weighted_f1: 0.9587 },
          val_metrics: { accuracy: 0.9397, macro_f1: 0.934, weighted_f1: 0.9395 },
          test_metrics: { accuracy: 0.9433, macro_f1: 0.906, weighted_f1: 0.9422 },
          test_confusion_matrix: {
            classes: ['High', 'Low', 'Medium'],
            matrix: [
              [85, 0, 5],
              [0, 17, 6],
              [3, 2, 164],
            ],
          },
          concordance_rate_overall: 95.37,
        });
      }
      if (u.includes('/health')) return mockApiResponse(u, { status: 'ok' });
      if (u.includes('/dashboard/overview')) return mockApiResponse(u, { priority_band_counts: {} });
      return mockApiResponse(u, {});
    });

    render(<App />);

    expect(await screen.findByText('Preparedness Analytics')).toBeInTheDocument();
    expect(screen.getByText('54.24')).toBeInTheDocument(); // Mean score
    expect(screen.getByText('53.11')).toBeInTheDocument(); // Median score
    expect(screen.getAllByText('94.3%').length).toBeGreaterThanOrEqual(1); // Test accuracy
    expect(screen.getByText('95.4%')).toBeInTheDocument(); // Concordance rate
    expect(screen.getByText('Urban & Peri-Urban Well-Resourced Corridor')).toBeInTheDocument();
    expect(screen.getByText('Unassigned (Uttara Kannada Fallback)')).toBeInTheDocument();
  });

  it('displays ErrorBox on API failure when demo fallback is disabled', async () => {
    window.location.hash = '#/communities';

    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const u = String(input);
      if (u.includes('/communities?')) {
        return Promise.reject(new Error('Network connection refused'));
      }
      if (u.includes('/health')) {
        return mockApiResponse(u, { status: 'ok' });
      }
      if (u.includes('/dashboard/overview')) {
        return mockApiResponse(u, { priority_band_counts: {} });
      }
      return mockApiResponse(u, {});
    });

    render(<App />);

    expect(await screen.findByText("Couldn't load this data")).toBeInTheDocument();
    expect(screen.getByText('Network connection refused')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('activates Presentation Demo Mode, steps through guided walkthrough, and executes sample prompts', async () => {
    window.location.hash = '#/overview';

    global.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      if (u.includes('/dashboard/overview')) {
        return mockApiResponse(u, {
          total_units: 1879,
          districts_count: 3,
          assigned_clusters_count: 622,
          unassigned_clusters_count: 1257,
          priority_band_counts: { BAND_1_CRITICAL: 155, BAND_2_SPECIALIZED: 829, BAND_3_MODERATE: 299, BAND_4_ADEQUATE: 596 },
          preparedness_tier_counts: { High: 596, Medium: 1128, Low: 155 },
          districts: [],
          infrastructure_highlights: { total_population: 4680000, pucca_road_pct: 78.5, coastal_units_under_5km: 249 },
        });
      }
      if (u.includes('/health')) {
        return mockApiResponse(u, { status: 'healthy', dataset_loaded: true, total_units: 1879 });
      }
      if (u.includes('/communities?')) {
        return mockApiResponse(u, {
          total: 1,
          page: 1,
          limit: 25,
          total_pages: 1,
          items: [{
            area_id: '11-29-569-05522-608691',
            pc11_id: '608691',
            shrid2: '11-29-569-05522-608691',
            village_name: 'Tallur',
            district_name: 'Udupi',
            subdistrict_name: 'Kundapura',
            unit_type: 'VILLAGE',
            preparedness_score: 74.34,
            preparedness_tier: 'High',
            priority_band: 'BAND_4_ADEQUATE',
            priority_rank: 1780,
          }],
        });
      }
      if (u.includes('/communities/11-29-569-05522-608691') || u.includes('/communities/608691')) {
        return mockApiResponse(u, {
          area_id: '11-29-569-05522-608691',
          pc11_id: '608691',
          village_name: 'Tallur',
          district_name: 'Udupi',
          subdistrict_name: 'Kundapura',
          preparedness_score: 74.34,
          preparedness_tier: 'High',
          priority_band: 'BAND_4_ADEQUATE',
          score_medical: 71.94,
          score_shelters: 68.59,
          score_infrastructure: 68.0,
          score_early_warning: 82.11,
          score_socioeconomic: 88.58,
          triggers_activated: [],
          actionable_advisories: ['Maintain storm-surge shelters.'],
        });
      }
      if (u.includes('/assistant/chat') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body || '{}'));
        return mockApiResponse(u, {
          answer: `Deterministic AHP assessment: Tallur achieves an authoritative score of 74.34 (High Tier, Band 4 Adequate).`,
          tools_executed: ['explain_preparedness_assessment'],
          tool_records: [{ tool_name: 'explain_preparedness_assessment', success: true, execution_time_ms: 2.1 }],
          is_fallback: false,
          scientific_precedence_note: 'AHP composite preparedness score maintains operational authority.',
        });
      }
      return mockApiResponse(u, {});
    });

    render(<App />);

    // 1. Visible entry point
    const demoBtn = await screen.findByRole('button', { name: /Toggle Presentation Demo Mode/i });
    expect(demoBtn).toBeInTheDocument();
    expect(demoBtn).toHaveTextContent('Demo Mode');

    // 2. Click to activate Demo Mode
    fireEvent.click(demoBtn);
    const controls = await screen.findByRole('region', { name: /Presentation Demo Controls/i });
    expect(controls).toBeInTheDocument();
    expect(within(controls).getByText('Step 1 of 7')).toBeInTheDocument();
    expect(within(controls).getByText('Command Center')).toBeInTheDocument();
    expect(within(controls).getByText(/Census 2011 & Spatial Baseline/)).toBeInTheDocument();

    // 3. Step forward: Next to Map
    const nextBtn = within(controls).getByRole('button', { name: /Next/i });
    fireEvent.click(nextBtn);
    expect(within(controls).getByText('Step 2 of 7')).toBeInTheDocument();
    expect(within(controls).getByText('Community Preparedness Map')).toBeInTheDocument();

    // 4. Step forward: Next to Communities Explorer
    fireEvent.click(nextBtn);
    expect(within(controls).getByText('Step 3 of 7')).toBeInTheDocument();
    expect(within(controls).getByText('Community Explorer')).toBeInTheDocument();

    // 5. Step forward: Next to Community Dossier (Tallur PC11 608691)
    fireEvent.click(nextBtn);
    expect(within(controls).getByText('Step 4 of 7')).toBeInTheDocument();
    expect(within(controls).getByText('Community Dossier (Tallur)')).toBeInTheDocument();
    expect(await screen.findByText('Maintain storm-surge shelters.')).toBeInTheDocument();
    expect(screen.getAllByText('Tallur').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('74.34').length).toBeGreaterThanOrEqual(1);

    // 6. Direct jump to Step 7 (AI Decision Support Assistant)
    const step7Btn = within(controls).getByRole('button', { name: /Jump to step 7/i });
    fireEvent.click(step7Btn);
    expect(within(controls).getByText('Step 7 of 7')).toBeInTheDocument();
    expect(within(controls).getByText('AI Decision Support Assistant')).toBeInTheDocument();

    // Verify all 6 clickable presentation sample prompts are present
    expect(screen.getByText('Explain why Tallur is classified as Band 4.')).toBeInTheDocument();
    expect(screen.getByText('Compare Dakshina Kannada and Udupi.')).toBeInTheDocument();
    expect(screen.getByText('Explain the four vulnerability archetypes.')).toBeInTheDocument();
    expect(screen.getByText('Identify Band 1 communities using the actual dataset.')).toBeInTheDocument();
    expect(screen.getByText('Explain how AHP preparedness tiers differ from operational priority bands.')).toBeInTheDocument();
    expect(screen.getByText('Explain preparedness measures based on dataset-backed vulnerabilities.')).toBeInTheDocument();

    // 7. Click sample prompt and verify live execution
    const samplePromptBtn = screen.getByText('Explain why Tallur is classified as Band 4.');
    fireEvent.click(samplePromptBtn);

    expect(await screen.findByText(/Tallur achieves an authoritative score of 74.34/)).toBeInTheDocument();
    expect(screen.getByText(/explain_preparedness_assessment/)).toBeInTheDocument();

    // 8. Reset Demo Session
    const resetBtn = screen.getByRole('button', { name: /Reset Session/i });
    fireEvent.click(resetBtn);
    expect(screen.getByText('Step 1 of 7')).toBeInTheDocument();

    // 9. Exit Demo Mode
    const exitBtn = screen.getByRole('button', { name: /Exit presentation mode/i });
    fireEvent.click(exitBtn);
    expect(screen.queryByRole('region', { name: /Presentation Demo Controls/i })).not.toBeInTheDocument();
  });
});
