/** TypeScript Type Definitions matching FastAPI Pydantic Models */

export interface DistrictOverviewItem {
  district_name: string;
  total_units: number;
  mean_preparedness_score: number;
  band_counts: Record<string, number>;
  tier_counts: Record<string, number>;
  mean_dist_nearest_shelter_km: number;
  mean_dist_nearest_hospital_km: number;
  mean_dist_coast_km: number;
  critical_units_count: number;
}

export interface DashboardOverviewResponse {
  total_units: number;
  districts_count: number;
  assigned_clusters_count: number;
  unassigned_clusters_count: number;
  priority_band_counts: {
    BAND_1_CRITICAL: number;
    BAND_2_SPECIALIZED: number;
    BAND_3_MODERATE: number;
    BAND_4_ADEQUATE: number;
  };
  preparedness_tier_counts: {
    High: number;
    Medium: number;
    Low: number;
  };
  cluster_counts: Record<string, number>;
  districts: DistrictOverviewItem[];
  infrastructure_highlights: {
    total_population: number;
    total_households: number;
    mean_dist_coast_km: number;
    mean_dist_nearest_shelter_km: number;
    mean_dist_nearest_hospital_km: number;
    mean_dist_nearest_phc_km: number;
    pucca_road_pct: number;
    coastal_units_under_5km: number;
  };
  data_disclaimer: string;
}

export interface SubdistrictSummary {
  subdistrict_name: string;
  total_units: number;
  mean_preparedness_score: number;
  band_counts: Record<string, number>;
  critical_units_count: number;
}

export interface DistrictSummary {
  district_name: string;
  district_code?: string;
  total_units: number;
  mean_preparedness_score: number;
  min_preparedness_score: number;
  max_preparedness_score: number;
  band_counts: Record<string, number>;
  tier_counts: Record<string, number>;
  cluster_counts: Record<string, number>;
  mean_dist_coast_km: number;
  mean_dist_nearest_shelter_km: number;
  mean_dist_nearest_hospital_km: number;
  mean_dist_nearest_phc_km: number;
  pct_pucca_road: number;
  pct_drinking_water_secure: number;
  total_population: number;
  total_households: number;
  mean_vulnerable_pop_share: number;
  subdistricts?: SubdistrictSummary[];
}

export interface DistrictListResponse {
  total_districts: number;
  districts: DistrictSummary[];
}

export interface CommunityListItem {
  area_id: string;
  pc11_id: string;
  shrid2: string;
  village_name: string;
  district_name: string;
  subdistrict_name?: string;
  unit_type: string;
  lat: number;
  lon: number;
  preparedness_score: number;
  preparedness_tier: string;
  priority_band: string;
  priority_rank: number;
  band_rank: number;
  vulnerability_archetype_id?: number;
  vulnerability_archetype_label?: string;
  population?: number;
  dist_coast_km?: number;
  dist_nearest_shelter_km?: number;
  dist_nearest_hospital_km?: number;
}

export interface CommunitiesListResponse {
  total: number;
  page: number;
  limit: number;
  total_pages: number;
  items: CommunityListItem[];
}

export interface CommunityDetailResponse {
  area_id: string;
  pc11_id: string;
  shrid2: string;
  village_name: string;
  district_name: string;
  subdistrict_name?: string;
  unit_type: string;
  lat: number;
  lon: number;

  preparedness_score: number;
  preparedness_tier: string;
  score_medical: number;
  score_shelters: number;
  score_infrastructure: number;
  score_early_warning: number;
  score_socioeconomic: number;

  priority_band: string;
  priority_rank: number;
  band_rank: number;
  triggers_activated: string[];
  primary_deficit_rationale: string;
  actionable_advisories: string[];

  predicted_preparedness_tier?: string;
  prob_high?: number;
  prob_medium?: number;
  prob_low?: number;
  probability_margin?: number;
  is_boundary_case?: boolean;
  ahp_agreement?: string;
  ahp_precedence_note?: string;
  ml_disclaimer: string;

  vulnerability_archetype_id?: number;
  vulnerability_archetype_label?: string;
  cluster_assignment_status: string;
  archetype_intervention_need?: string;

  population: number;
  households: number;
  pop_density_sqkm: number;
  vulnerable_pop_share: number;
  sc_share: number;
  st_share: number;
  dist_coast_km: number;
  elevation_mean: number;
  dist_nearest_shelter_km: number;
  n_shelters_10km: number;
  dist_nearest_hospital_km: number;
  dist_nearest_phc_km: number;
  has_phc_or_chc: boolean;
  road_connectivity_score: number;
  drinking_water_score: number;
  has_pucca_road: boolean;
  bus_connectivity: boolean;
  tauktae_affected?: boolean;

  provenance_and_limitations: {
    census_vintage: number;
    healthcare_source: string;
    coastline_source: string;
    shelter_source: string;
    operational_limitations: string;
  };
}

export interface MapCommunityPoint {
  area_id: string;
  pc11_id: string;
  name: string;
  district: string;
  subdistrict?: string;
  lat: number;
  lon: number;
  score: number;
  tier: string;
  band: string;
  priority_rank: number;
  archetype_id?: number;
  dist_coast_km?: number;
  dist_nearest_shelter_km?: number;
  dist_nearest_hospital_km?: number;
  population?: number;
}

export interface MapCommunitiesResponse {
  total: number;
  points: MapCommunityPoint[];
}

export interface PreparednessAnalyticsResponse {
  total_units: number;
  tier_distribution: Record<string, number>;
  score_summary: {
    mean: number;
    median: number;
    min: number;
    max: number;
    std: number;
  };
  domain_averages: Record<string, number>;
  district_averages: Record<string, Record<string, number>>;
  disclaimer: string;
}

export interface PriorityBandsAnalyticsResponse {
  total_units: number;
  band_distribution: Record<string, number>;
  trigger_counts: Record<string, number>;
  district_band_crosstab: Record<string, Record<string, number>>;
  disclaimer: string;
}

export interface ClusterDetail {
  cluster_id: number;
  label: string;
  count: number;
  district_distribution: Record<string, number>;
  primary_intervention_need: string;
  detailed_description: string;
  dimension_means: Record<string, number>;
}

export interface ClustersAnalyticsResponse {
  total_units: number;
  assigned_count: number;
  unassigned_count: number;
  clusters: ClusterDetail[];
  disclaimer: string;
}

export interface ModelMetrics {
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
}

export interface ConfusionMatrixData {
  classes: string[];
  matrix: number[][];
}

export interface ModelPerformanceResponse {
  model_name: string;
  model_type: string;
  framework: string;
  features_count: number;
  features: string[];
  split_info: Record<string, any>;
  train_metrics: ModelMetrics;
  val_metrics: ModelMetrics;
  test_metrics: ModelMetrics;
  test_confusion_matrix: ConfusionMatrixData;
  test_classification_report: Record<string, any>;
  concordance_rate_overall: number;
  disclaimer: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  tool_executions?: ToolExecutionRecord[];
}

export interface ToolExecutionRecord {
  tool_name: string;
  arguments: Record<string, any>;
  status?: string;
  success?: boolean;
  execution_time_ms?: number;
  result_summary?: string;
  error?: string | null;
}

export interface ChatRequest {
  query: string;
  history?: ChatMessage[];
  context_community_id?: string;
}

export interface ChatResponse {
  answer: string;
  conversation_status: string;
  tools_executed?: string[];
  tool_records?: ToolExecutionRecord[];
  tool_executions?: ToolExecutionRecord[];
  structured_evidence?: Record<string, any>;
  disclaimer?: string;
  duration_ms?: number;
  total_duration_ms?: number;
  model_name?: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
}
