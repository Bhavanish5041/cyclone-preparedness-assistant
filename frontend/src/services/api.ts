import {
  ChatRequest,
  ChatResponse,
  CommunitiesListResponse,
  CommunityDetailResponse,
  DashboardOverviewResponse,
  DistrictListResponse,
  DistrictSummary,
  MapCommunitiesResponse,
  ModelPerformanceResponse,
  PreparednessAnalyticsResponse,
  PriorityBandsAnalyticsResponse,
  ClustersAnalyticsResponse,
  ToolDefinition,
} from '../types';

const API_BASE = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail || errorJson.message || errorDetail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export async function fetchDashboardOverview(): Promise<DashboardOverviewResponse> {
  const res = await fetch(`${API_BASE}/dashboard/overview`);
  return handleResponse<DashboardOverviewResponse>(res);
}

export async function fetchDistricts(): Promise<DistrictListResponse> {
  const res = await fetch(`${API_BASE}/districts`);
  return handleResponse<DistrictListResponse>(res);
}

export async function fetchDistrictDetail(districtId: string): Promise<DistrictSummary> {
  const res = await fetch(`${API_BASE}/districts/${encodeURIComponent(districtId)}`);
  return handleResponse<DistrictSummary>(res);
}

export interface CommunityQueryParams {
  search?: string;
  district?: string;
  tier?: string;
  priority_band?: string;
  archetype_id?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  page?: number;
  limit?: number;
}

export async function fetchCommunities(params: CommunityQueryParams = {}): Promise<CommunitiesListResponse> {
  const searchParams = new URLSearchParams();
  if (params.search) searchParams.append('search', params.search);
  if (params.district) searchParams.append('district', params.district);
  if (params.tier) searchParams.append('tier', params.tier);
  if (params.priority_band) searchParams.append('priority_band', params.priority_band);
  if (params.archetype_id !== undefined && params.archetype_id !== null) {
    searchParams.append('archetype_id', String(params.archetype_id));
  }
  if (params.sort_by) searchParams.append('sort_by', params.sort_by);
  if (params.sort_order) searchParams.append('sort_order', params.sort_order);
  if (params.page) searchParams.append('page', String(params.page));
  if (params.limit) searchParams.append('limit', String(params.limit));

  const url = `${API_BASE}/communities${searchParams.toString() ? `?${searchParams.toString()}` : ''}`;
  const res = await fetch(url);
  return handleResponse<CommunitiesListResponse>(res);
}

export async function fetchCommunityDetail(communityId: string): Promise<CommunityDetailResponse> {
  const res = await fetch(`${API_BASE}/communities/${encodeURIComponent(communityId)}`);
  return handleResponse<CommunityDetailResponse>(res);
}

export async function fetchMapCommunities(params: {
  district?: string;
  tier?: string;
  priority_band?: string;
  archetype_id?: number;
} = {}): Promise<MapCommunitiesResponse> {
  const searchParams = new URLSearchParams();
  if (params.district) searchParams.append('district', params.district);
  if (params.tier) searchParams.append('tier', params.tier);
  if (params.priority_band) searchParams.append('priority_band', params.priority_band);
  if (params.archetype_id !== undefined && params.archetype_id !== null) {
    searchParams.append('archetype_id', String(params.archetype_id));
  }

  const url = `${API_BASE}/maps/communities${searchParams.toString() ? `?${searchParams.toString()}` : ''}`;
  const res = await fetch(url);
  return handleResponse<MapCommunitiesResponse>(res);
}

export async function fetchPreparednessAnalytics(): Promise<PreparednessAnalyticsResponse> {
  const res = await fetch(`${API_BASE}/analytics/preparedness`);
  return handleResponse<PreparednessAnalyticsResponse>(res);
}

export async function fetchPriorityBandsAnalytics(): Promise<PriorityBandsAnalyticsResponse> {
  const res = await fetch(`${API_BASE}/analytics/priority-bands`);
  return handleResponse<PriorityBandsAnalyticsResponse>(res);
}

export async function fetchClustersAnalytics(): Promise<ClustersAnalyticsResponse> {
  const res = await fetch(`${API_BASE}/analytics/clusters`);
  return handleResponse<ClustersAnalyticsResponse>(res);
}

export async function fetchModelPerformance(): Promise<ModelPerformanceResponse> {
  const res = await fetch(`${API_BASE}/analytics/model-performance`);
  return handleResponse<ModelPerformanceResponse>(res);
}

export async function fetchAssistantTools(): Promise<ToolDefinition[]> {
  const res = await fetch(`${API_BASE}/assistant/tools`);
  return handleResponse<ToolDefinition[]>(res);
}

export async function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  const payload = {
    message: req.query,
    query: req.query,
    conversation_history: (req.history || []).map((m) => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.content,
    })),
    community_identifier: req.context_community_id,
  };
  const res = await fetch(`${API_BASE}/assistant/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<ChatResponse>(res);
}
