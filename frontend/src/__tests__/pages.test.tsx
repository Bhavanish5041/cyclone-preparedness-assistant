import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { DistrictsPage } from '../pages/DistrictsPage';
import { AssistantPage } from '../pages/AssistantPage';
import { CommunitiesPage } from '../pages/CommunitiesPage';
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

  it('renders CommunitiesPage with authoritative community records and filters', async () => {
    vi.spyOn(api, 'fetchCommunities').mockResolvedValueOnce({
      total: 1879,
      page: 1,
      limit: 20,
      total_pages: 94,
      items: [
        {
          area_id: '608678',
          pc11_id: '608678',
          shrid2: '11-29-569-00000-608678',
          village_name: 'Tallur',
          district_name: 'Udupi',
          subdistrict_name: 'Kundapura',
          unit_type: 'VILLAGE',
          lat: 13.6358,
          lon: 74.7089,
          preparedness_score: 65.4,
          preparedness_tier: 'High',
          priority_band: 'BAND_4_ADEQUATE',
          priority_rank: 1800,
          band_rank: 500,
          vulnerability_archetype_id: 3,
          vulnerability_archetype_label: 'Low Vulnerability / High Preparedness',
          population: 3450,
          dist_coast_km: 4.2,
          dist_nearest_shelter_km: 1.8,
          dist_nearest_hospital_km: 5.1,
        },
      ],
    });

    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <BrowserRouter>
          <CommunitiesPage />
        </BrowserRouter>
      </QueryClientProvider>
    );

    expect(await screen.findByText('Tallur')).toBeInTheDocument();
    expect(screen.getByText('Community Explorer')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Search by community name/)).toBeInTheDocument();
    expect(screen.getByText('1,879')).toBeInTheDocument();
  });

  it('renders CommunitiesPage empty state when no matching communities', async () => {
    vi.spyOn(api, 'fetchCommunities').mockResolvedValueOnce({
      total: 0,
      page: 1,
      limit: 20,
      total_pages: 1,
      items: [],
    });

    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <BrowserRouter>
          <CommunitiesPage />
        </BrowserRouter>
      </QueryClientProvider>
    );

    expect(await screen.findByText('No communities match your criteria')).toBeInTheDocument();
    expect(
      screen.getByText(/Try modifying search query, district filter/)
    ).toBeInTheDocument();
  });

  it('renders CommunitiesPage error state when API fails with Failed to fetch', async () => {
    vi.spyOn(api, 'fetchCommunities').mockRejectedValueOnce(new Error('Failed to fetch'));

    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <BrowserRouter>
          <CommunitiesPage />
        </BrowserRouter>
      </QueryClientProvider>
    );

    expect(await screen.findByText('Error loading community table')).toBeInTheDocument();
    expect(screen.getByText('Failed to fetch')).toBeInTheDocument();
  });
});
