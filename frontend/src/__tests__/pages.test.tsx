import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { DistrictsPage } from '../pages/DistrictsPage';
import { AssistantPage } from '../pages/AssistantPage';
import * as api from '../services/api';

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

describe('Page Integration Tests', () => {
  it('renders DashboardPage with mocked authoritative data', async () => {
    vi.spyOn(api, 'fetchDashboardOverview').mockResolvedValueOnce({
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
      preparedness_tier_counts: {
        High: 596,
        Medium: 1128,
        Low: 155,
      },
      cluster_counts: {},
      districts: [
        {
          district_name: 'Dakshina Kannada',
          total_units: 369,
          mean_preparedness_score: 61.06,
          band_counts: { BAND_1_CRITICAL: 2, BAND_2_SPECIALIZED: 123, BAND_3_MODERATE: 45, BAND_4_ADEQUATE: 199 },
          tier_counts: { High: 199, Medium: 168, Low: 2 },
          mean_dist_nearest_shelter_km: 34.9,
          mean_dist_nearest_hospital_km: 37.8,
          mean_dist_coast_km: 18.2,
          critical_units_count: 2,
        },
      ],
      infrastructure_highlights: {
        total_population: 3200000,
        total_households: 750000,
        mean_dist_coast_km: 15.4,
        mean_dist_nearest_shelter_km: 25.1,
        mean_dist_nearest_hospital_km: 38.2,
        mean_dist_nearest_phc_km: 8.5,
        pucca_road_pct: 88.5,
        coastal_units_under_5km: 249,
      },
      data_disclaimer: 'Scientific Governance Baseline Notice',
    });

    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <BrowserRouter>
          <DashboardPage />
        </BrowserRouter>
      </QueryClientProvider>
    );

    expect(await screen.findByText('Regional Preparedness Overview')).toBeInTheDocument();
    expect(screen.getByText('1,879')).toBeInTheDocument();
    expect(screen.getAllByText('155').length).toBeGreaterThanOrEqual(1);
  });

  it('renders DistrictsPage with comparative district profiles', async () => {
    vi.spyOn(api, 'fetchDistricts').mockResolvedValueOnce({
      total_districts: 3,
      districts: [
        {
          district_name: 'Dakshina Kannada',
          total_units: 369,
          mean_preparedness_score: 61.06,
          min_preparedness_score: 25.0,
          max_preparedness_score: 88.5,
          band_counts: { BAND_1_CRITICAL: 2, BAND_2_SPECIALIZED: 123, BAND_3_MODERATE: 45, BAND_4_ADEQUATE: 199 },
          tier_counts: { High: 199, Medium: 168, Low: 2 },
          cluster_counts: {},
          mean_dist_coast_km: 18.2,
          mean_dist_nearest_shelter_km: 34.9,
          mean_dist_nearest_hospital_km: 37.8,
          mean_dist_nearest_phc_km: 8.5,
          pct_pucca_road: 88.5,
          pct_drinking_water_secure: 72.0,
          total_population: 2089649,
          total_households: 442125,
          mean_vulnerable_pop_share: 0.12,
          subdistricts: [],
        },
      ],
    });

    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <BrowserRouter>
          <DistrictsPage />
        </BrowserRouter>
      </QueryClientProvider>
    );

    expect(await screen.findByText('Comparative District Intelligence')).toBeInTheDocument();
    expect(screen.getByText('Dakshina Kannada')).toBeInTheDocument();
  });

  it('renders AssistantPage and suggested inquiry prompts', () => {
    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <BrowserRouter>
          <AssistantPage />
        </BrowserRouter>
      </QueryClientProvider>
    );

    expect(screen.getAllByText('Preparedness Intelligence Assistant').length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByText('Explain why Tallur has its current priority classification.')
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Ask an operational/)).toBeInTheDocument();
  });
});
