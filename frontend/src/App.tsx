/**
 * ============================================================================
 *  CYCLONE INTEL — Coastal Karnataka Preparedness Command Center
 *  Single-file React frontend (drop-in replacement for frontend/src/App.tsx)
 * ============================================================================
 *
 *  HOW TO INTEGRATE
 *  1. Replace `frontend/src/App.tsx` with this file (it exports `App` + default).
 *     main.tsx already does `import { App } from './App'` — nothing else to change.
 *  2. Uses ONLY packages already in package.json:
 *       react, lucide-react, recharts, react-leaflet/leaflet, react-markdown
 *  3. API base defaults to `/api` (Vite proxy → FastAPI :8000).
 *     Override with env var VITE_API_BASE (e.g. https://my-api.onrender.com/api).
 *  4. If the backend is unreachable, the UI falls back to realistic DEMO data
 *     (a "Demo data" pill appears in the top bar). Set USE_DEMO_FALLBACK=false
 *     to show error states instead.
 *  5. Optional: set BACKGROUND_PHOTO_URL to a real storm photo. By default a
 *     fully procedural, animated storm sky (clouds, lightning, rain, fields)
 *     is rendered — no image assets needed.
 *
 *  Endpoints used:
 *   GET  /dashboard/overview          GET /analytics/preparedness
 *   GET  /analytics/priority-bands    GET /analytics/clusters
 *   GET  /analytics/model-performance GET /districts
 *   GET  /communities?…               GET /communities/{id}
 *   GET  /maps/communities?…          POST /assistant/chat
 *   GET  /health
 * ============================================================================
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowRight, ArrowUpRight, BarChart3, Bell, Bot, Building2, ChevronLeft,
  ChevronRight, CloudLightning, CloudRain, Cloudy, Crosshair, Droplets, Filter, Gauge, HeartPulse,
  Home, Landmark, LayoutGrid, Loader2, LogOut, MapPin, Map as MapIcon, Navigation, Plus,
  Radio, Search, Send, Settings, ShieldCheck, Sparkles, Sun, Users, Waves, Wind, Wrench, X, Zap,
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis,
  Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { CircleMarker, MapContainer, TileLayer, Tooltip as LTooltip, useMap } from 'react-leaflet';
import ReactMarkdown from 'react-markdown';
import 'leaflet/dist/leaflet.css';

/* ───────────────────────────── CONFIG ───────────────────────────── */

const API_BASE: string = ((import.meta as any).env?.VITE_API_BASE as string) || '/api';
const USE_DEMO_FALLBACK = ((import.meta as any).env?.VITE_ENABLE_DEMO === "true");
const BACKGROUND_PHOTO_URL = ''; // e.g. 'https://images.unsplash.com/....jpg'
const OFFICER_NAME = 'District Control Room';

/* ───────────────────────────── TOKENS ───────────────────────────── */

const BAND: Record<string, { hex: string; label: string; short: string }> = {
  BAND_1_CRITICAL: { hex: '#FF6B5E', label: 'Band 1 · Critical', short: 'Critical' },
  BAND_2_SPECIALIZED: { hex: '#F5A94A', label: 'Band 2 · Specialized', short: 'Specialized' },
  BAND_3_MODERATE: { hex: '#5FD39A', label: 'Band 3 · Moderate', short: 'Moderate' },
  BAND_4_ADEQUATE: { hex: '#6EC3F5', label: 'Band 4 · Adequate', short: 'Adequate' },
};
const TIER: Record<string, string> = { High: '#6EC3F5', Medium: '#F5A94A', Low: '#FF6B5E' };
const ARCHETYPE: Record<number, { hex: string; label: string }> = {
  0: { hex: '#6EC3F5', label: 'Urban & Peri-Urban Well-Resourced Corridor' },
  1: { hex: '#F5A94A', label: 'Remote Foothill & Healthcare-Isolated' },
  2: { hex: '#B794F6', label: 'Socioeconomically Vulnerable Interior' },
  3: { hex: '#FF6B5E', label: 'High-Exposure Coastal Rural Belt' },
};
const DISTRICTS = ['Dakshina Kannada', 'Udupi', 'Uttara Kannada'];
const DISTRICT_VIEW: Record<string, { c: [number, number]; z: number }> = {
  '': { c: [13.75, 74.75], z: 8 },
  'Dakshina Kannada': { c: [12.85, 75.15], z: 10 },
  Udupi: { c: [13.45, 74.85], z: 10 },
  'Uttara Kannada': { c: [14.6, 74.6], z: 9 },
};
const DOMAIN_META: Record<string, { label: string; icon: React.ElementType }> = {
  medical: { label: 'Medical', icon: HeartPulse },
  shelters: { label: 'Shelters', icon: Home },
  infrastructure: { label: 'Infra', icon: Wrench },
  early_warning: { label: 'Warning', icon: Radio },
  socioeconomic: { label: 'Socio-eco', icon: Users },
};

/* ───────────────────────────── UTILS ───────────────────────────── */

const fmt = (n: any, d = 0) =>
  n === null || n === undefined || Number.isNaN(Number(n))
    ? '—'
    : Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (n: any, d = 0) => (n === null || n === undefined ? '—' : `${fmt(Number(n) * (Number(n) <= 1 ? 100 : 1), d)}%`);
const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(' ');
const titleCase = (s: string) => s.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
const sumVals = (o: Record<string, number> = {}) => Object.values(o).reduce((a, b) => a + (Number(b) || 0), 0);

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ───────────────────────────── DEMO DATA ───────────────────────────── */

const DEMO = (() => {
  const rnd = mulberry32(42);
  const prefixes = ['Kod', 'Ull', 'Bel', 'Kum', 'Hon', 'Mal', 'Kar', 'Sur', 'Pad', 'Kot', 'Bant', 'Him', 'Mur', 'Gok', 'Shir', 'Kal', 'Man', 'Byn', 'Ank', 'Sir'];
  const suffixes = ['ala', 'uru', 'avar', 'ady', 'agiri', 'apura', 'ebail', 'ikere', 'oor', 'ahalli', 'ady', 'angadi'];
  const subs: Record<string, string[]> = {
    'Dakshina Kannada': ['Mangaluru', 'Bantwal', 'Puttur', 'Belthangady', 'Sullia'],
    Udupi: ['Udupi', 'Kundapura', 'Karkala', 'Kapu', 'Byndoor'],
    'Uttara Kannada': ['Karwar', 'Ankola', 'Kumta', 'Honnavar', 'Bhatkal', 'Sirsi'],
  };
  const geo: Record<string, { lat: [number, number]; coast: (lat: number) => number }> = {
    'Dakshina Kannada': { lat: [12.5, 13.1], coast: (l) => 74.82 + (l - 12.5) * 0.12 },
    Udupi: { lat: [13.1, 13.9], coast: (l) => 74.7 - (l - 13.1) * 0.12 },
    'Uttara Kannada': { lat: [13.9, 15.1], coast: (l) => 74.5 - (l - 13.9) * 0.22 },
  };
  const counts: Record<string, number> = { 'Dakshina Kannada': 369, Udupi: 253, 'Uttara Kannada': 1257 };
  const items: any[] = [];
  let id = 0;
  DISTRICTS.forEach((d) => {
    for (let i = 0; i < counts[d]; i++) {
      const g = geo[d];
      const lat = g.lat[0] + rnd() * (g.lat[1] - g.lat[0]);
      const inland = Math.pow(rnd(), 1.3) * 0.75;
      const lon = g.coast(lat) + inland;
      const coastKm = inland * 105;
      const base = 0.62 - inland * 0.35 + (rnd() - 0.5) * 0.35 + (d === 'Udupi' ? 0.05 : d === 'Uttara Kannada' ? -0.05 : 0.02);
      const score = Math.max(0.08, Math.min(0.94, base));
      const tier = score > 0.6 ? 'High' : score > 0.4 ? 'Medium' : 'Low';
      const band = score < 0.33 ? 'BAND_1_CRITICAL' : score < 0.45 ? 'BAND_2_SPECIALIZED' : score < 0.62 ? 'BAND_3_MODERATE' : 'BAND_4_ADEQUATE';
      const arch = coastKm < 8 ? 3 : score > 0.62 ? 0 : coastKm > 45 ? 1 : 2;
      const name = prefixes[Math.floor(rnd() * prefixes.length)] + suffixes[Math.floor(rnd() * suffixes.length)];
      const sd = subs[d][Math.floor(rnd() * subs[d].length)];
      id++;
      items.push({
        area_id: `A${String(id).padStart(5, '0')}`,
        pc11_id: String(600000 + id * 7),
        shrid2: `11-29-${d === 'Udupi' ? 588 : d === 'Uttara Kannada' ? 562 : 575}-${String(id).padStart(5, '0')}`,
        village_name: name, name, district_name: d, district: d, subdistrict_name: sd, subdistrict: sd,
        unit_type: rnd() > 0.92 ? 'Town' : 'Village',
        lat, lon, preparedness_score: score, score, preparedness_tier: tier, tier, priority_band: band, band,
        vulnerability_archetype_id: arch, archetype_id: arch, vulnerability_archetype_label: ARCHETYPE[arch].label,
        population: Math.round(400 + rnd() * 9000), dist_coast_km: coastKm,
        dist_nearest_shelter_km: 0.5 + rnd() * 14 + inland * 8, dist_nearest_hospital_km: 1 + rnd() * 22 + inland * 12,
      });
    }
  });
  items.sort((a, b) => a.preparedness_score - b.preparedness_score);
  items.forEach((it, i) => { it.priority_rank = i + 1; it.band_rank = i + 1; });

  const by = (k: string, arr = items) => arr.reduce((m: any, x: any) => ((m[x[k]] = (m[x[k]] || 0) + 1), m), {});
  const mean = (arr: any[], k: string) => arr.reduce((a, x) => a + x[k], 0) / Math.max(arr.length, 1);
  const districts = DISTRICTS.map((d) => {
    const s = items.filter((x) => x.district_name === d);
    return {
      district_name: d, total_units: s.length, mean_preparedness_score: mean(s, 'preparedness_score'),
      min_preparedness_score: Math.min(...s.map((x) => x.preparedness_score)), max_preparedness_score: Math.max(...s.map((x) => x.preparedness_score)),
      band_counts: by('priority_band', s), tier_counts: by('preparedness_tier', s), cluster_counts: by('vulnerability_archetype_id', s),
      mean_dist_coast_km: mean(s, 'dist_coast_km'), mean_dist_nearest_shelter_km: mean(s, 'dist_nearest_shelter_km'),
      mean_dist_nearest_hospital_km: mean(s, 'dist_nearest_hospital_km'), mean_dist_nearest_phc_km: mean(s, 'dist_nearest_hospital_km') * 0.7,
      pct_pucca_road: 0.6 + rnd() * 0.3, pct_drinking_water_secure: 0.55 + rnd() * 0.35,
      total_population: s.reduce((a, x) => a + x.population, 0), total_households: Math.round(s.reduce((a, x) => a + x.population, 0) / 4.3),
      mean_vulnerable_pop_share: 0.18 + rnd() * 0.12, critical_units_count: s.filter((x) => x.priority_band === 'BAND_1_CRITICAL').length,
    };
  });
  const domain = { medical: 0.47, shelters: 0.39, infrastructure: 0.58, early_warning: 0.52, socioeconomic: 0.61 };
  const overview = {
    total_units: items.length, districts_count: 3, assigned_clusters_count: 622, unassigned_clusters_count: 1257,
    priority_band_counts: by('priority_band'), preparedness_tier_counts: by('preparedness_tier'), cluster_counts: by('vulnerability_archetype_id'),
    districts,
    infrastructure_highlights: {
      total_population: items.reduce((a, x) => a + x.population, 0), total_households: Math.round(items.reduce((a, x) => a + x.population, 0) / 4.3),
      mean_dist_coast_km: mean(items, 'dist_coast_km'), mean_dist_nearest_shelter_km: mean(items, 'dist_nearest_shelter_km'),
      mean_dist_nearest_hospital_km: mean(items, 'dist_nearest_hospital_km'), mean_dist_nearest_phc_km: mean(items, 'dist_nearest_hospital_km') * 0.7,
      pucca_road_pct: 0.74, coastal_units_under_5km: items.filter((x) => x.dist_coast_km < 5).length,
    },
    data_disclaimer: 'Demo data shown — connect the FastAPI backend for live Census 2011 / SHRUG v2.1 derived values.',
  };
  const prep = {
    total_units: items.length, tier_distribution: by('preparedness_tier'),
    score_summary: { mean: mean(items, 'preparedness_score'), median: items[Math.floor(items.length / 2)].preparedness_score, min: items[0].preparedness_score, max: items[items.length - 1].preparedness_score, std: 0.14 },
    domain_averages: domain,
    district_averages: Object.fromEntries(districts.map((d) => [d.district_name, { mean_score: d.mean_preparedness_score, ...Object.fromEntries(Object.entries(domain).map(([k, v]) => [k, Math.max(0.1, Math.min(0.9, v + (rnd() - 0.5) * 0.2))])) }])),
    disclaimer: overview.data_disclaimer,
  };
  const bands = {
    total_units: items.length, band_distribution: by('priority_band'),
    trigger_counts: { SHELTER_ACCESS_DEFICIT: 412, HOSPITAL_ISOLATION: 356, COASTAL_EXPOSURE_HIGH: 298, ROAD_CONNECTIVITY_GAP: 241, EARLY_WARNING_GAP: 187, DRINKING_WATER_RISK: 143, VULNERABLE_POP_CONCENTRATION: 129 },
    district_band_crosstab: Object.fromEntries(districts.map((d) => [d.district_name, d.band_counts])), disclaimer: '',
  };
  const clusters = {
    total_units: items.length, assigned_count: 622, unassigned_count: 1257,
    clusters: [0, 1, 2, 3].map((k) => ({
      cluster_id: k, label: ARCHETYPE[k].label, count: items.filter((x) => x.vulnerability_archetype_id === k).length,
      district_distribution: by('district_name', items.filter((x) => x.vulnerability_archetype_id === k)),
      primary_intervention_need: ['Maintain surge capacity & shelter upkeep', 'Mobile medical units & road access', 'Livelihood protection & warning reach', 'Evacuation drills & storm-surge shelters'][k],
      detailed_description: ['Well-connected peri-urban units with dense facilities.', 'Hilly hinterland units far from hospitals.', 'Interior villages with elevated socioeconomic vulnerability.', 'Low-lying coastal villages within storm-surge reach.'][k],
      dimension_means: {},
    })), disclaimer: '',
  };
  const model = {
    model_name: 'village-preparedness-classifier', model_type: 'Multinomial Logistic Regression', framework: 'scikit-learn', features_count: 14, features: [],
    split_info: {}, train_metrics: { accuracy: 0.91, macro_f1: 0.89, weighted_f1: 0.91 }, val_metrics: { accuracy: 0.88, macro_f1: 0.86, weighted_f1: 0.88 },
    test_metrics: { accuracy: 0.87, macro_f1: 0.85, weighted_f1: 0.87 },
    test_confusion_matrix: { classes: ['High', 'Low', 'Medium'], matrix: [[78, 0, 9], [0, 52, 8], [7, 6, 117]] },
    test_classification_report: {}, concordance_rate_overall: 0.89, disclaimer: '',
  };
  return { items, overview, districts, prep, bands, clusters, model };
})();

function demoFor(path: string): any {
  const [p, qs] = path.split('?');
  const q = new URLSearchParams(qs || '');
  const filt = (arr: any[]) =>
    arr.filter((x) =>
      (!q.get('district') || x.district_name === q.get('district')) &&
      (!q.get('tier') || x.preparedness_tier === q.get('tier')) &&
      (!q.get('priority_band') || x.priority_band === q.get('priority_band')) &&
      (!q.get('archetype_id') || String(x.vulnerability_archetype_id) === q.get('archetype_id')) &&
      (!q.get('search') || `${x.village_name} ${x.pc11_id} ${x.shrid2}`.toLowerCase().includes(q.get('search')!.toLowerCase())));
  if (p === '/dashboard/overview') return DEMO.overview;
  if (p === '/analytics/preparedness') return DEMO.prep;
  if (p === '/analytics/priority-bands') return DEMO.bands;
  if (p === '/analytics/clusters') return DEMO.clusters;
  if (p === '/analytics/model-performance') return DEMO.model;
  if (p === '/districts') return { total_districts: 3, districts: DEMO.districts };
  if (p === '/maps/communities') { const pts = filt(DEMO.items); return { total: pts.length, points: pts }; }
  if (p === '/communities') {
    let arr = filt(DEMO.items);
    const sb = q.get('sort_by') || 'priority_rank'; const so = q.get('sort_order') === 'desc' ? -1 : 1;
    arr = [...arr].sort((a, b) => (a[sb] > b[sb] ? so : a[sb] < b[sb] ? -so : 0));
    const page = Number(q.get('page') || 1); const limit = Number(q.get('limit') || 20);
    return { total: arr.length, page, limit, total_pages: Math.max(1, Math.ceil(arr.length / limit)), items: arr.slice((page - 1) * limit, page * limit) };
  }
  if (p.startsWith('/communities/')) {
    const it = DEMO.items.find((x) => x.area_id === decodeURIComponent(p.split('/')[2])) || DEMO.items[0];
    const s = it.preparedness_score;
    const j = (k: number) => Math.max(0.05, Math.min(0.95, s + Math.sin(k * 7.3 + s * 20) * 0.18));
    return {
      ...it, score_medical: j(1), score_shelters: j(2), score_infrastructure: j(3), score_early_warning: j(4), score_socioeconomic: j(5),
      triggers_activated: s < 0.45 ? ['SHELTER_ACCESS_DEFICIT', 'HOSPITAL_ISOLATION', it.dist_coast_km < 8 ? 'COASTAL_EXPOSURE_HIGH' : 'ROAD_CONNECTIVITY_GAP'] : s < 0.62 ? ['EARLY_WARNING_GAP'] : [],
      primary_deficit_rationale: s < 0.45 ? 'Composite deficit driven by shelter distance and isolation from secondary healthcare.' : 'No acute deficit; maintain routine preparedness.',
      actionable_advisories: s < 0.45 ? ['Pre-position a temporary cyclone shelter within 3 km.', 'Deploy a mobile medical unit during IMD orange alerts.', 'Verify siren & SMS early-warning reach to all wards.'] : ['Continue quarterly evacuation drills.', 'Audit shelter stock (water, ORS, generators) pre-monsoon.'],
      predicted_preparedness_tier: it.preparedness_tier, prob_high: it.preparedness_tier === 'High' ? 0.78 : 0.1, prob_medium: it.preparedness_tier === 'Medium' ? 0.74 : 0.14, prob_low: it.preparedness_tier === 'Low' ? 0.81 : 0.08,
      ahp_agreement: 'AGREE', ml_disclaimer: 'ML tier is a surrogate; AHP score has operational precedence.',
      cluster_assignment_status: 'ASSIGNED', archetype_intervention_need: DEMO.clusters.clusters[it.vulnerability_archetype_id].primary_intervention_need,
      households: Math.round(it.population / 4.3), pop_density_sqkm: 300 + s * 900, vulnerable_pop_share: 0.3 - s * 0.15, sc_share: 0.08, st_share: 0.04,
      elevation_mean: 8 + it.dist_coast_km * 6, n_shelters_10km: Math.round(s * 6), dist_nearest_phc_km: it.dist_nearest_hospital_km * 0.6,
      has_phc_or_chc: s > 0.5, road_connectivity_score: s, drinking_water_score: Math.min(0.95, s + 0.1), has_pucca_road: s > 0.35, bus_connectivity: s > 0.4,
      provenance_and_limitations: { census_vintage: 2011, healthcare_source: 'NHP facility registry', coastline_source: 'Natural Earth 10m', shelter_source: 'KSDMA', operational_limitations: 'Demo record.' },
    };
  }
  if (p === '/health') return { status: 'demo' };
  throw new Error('No demo data for ' + p);
}

/* ───────────────────────────── DATA LAYER ───────────────────────────── */

export const cache = new Map<string, any>();
export function clearApiCache() { cache.clear(); }
const demoListeners = new Set<(v: boolean) => void>();
let offlineFallbackDemo = false;
const setOfflineFallbackDemo = (v: boolean) => { if (offlineFallbackDemo !== v) { offlineFallbackDemo = v; demoListeners.forEach((f) => f(v)); } };

async function apiGet<T = any>(path: string): Promise<T> {
  if (cache.has(path)) return cache.get(path);
  try {
    const res = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const j = await res.json(); detail = j.detail || j.message || detail; } catch { /* ignore */ }
      throw new Error(detail);
    }
    const ct = res.headers.get('content-type') || '';
    if (!ct.includes('json')) throw new Error('Backend returned non-JSON response');
    const data = await res.json();
    cache.set(path, data);
    return data;
  } catch (e) {
    if (!USE_DEMO_FALLBACK) throw e;
    setOfflineFallbackDemo(true);
    const d = demoFor(path);
    cache.set(path, d);
    return d;
  }
}

function useApi<T = any>(path: string | null) {
  const [state, setState] = useState<{ data: T | null; loading: boolean; error: string | null }>({ data: path ? cache.get(path) ?? null : null, loading: !!path && !cache.has(path), error: null });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    if (!cache.has(path)) setState((s) => ({ ...s, loading: true, error: null }));
    apiGet<T>(path)
      .then((data) => alive && setState({ data, loading: false, error: null }))
      .catch((e) => alive && setState({ data: null, loading: false, error: e.message || 'Request failed' }));
    return () => { alive = false; };
  }, [path, nonce]);
  const retry = useCallback(() => { if (path) cache.delete(path); setNonce((n) => n + 1); }, [path]);
  return { ...state, retry };
}

function useDemoMode() {
  const [v, setV] = useState(offlineFallbackDemo);
  useEffect(() => { demoListeners.add(setV); return () => { demoListeners.delete(setV); }; }, []);
  return v;
}

async function sendChat(query: string, history: { role: string; content: string }[], communityId?: string) {
  try {
    const res = await fetch(`${API_BASE}/assistant/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: query, query,
        conversation_history: history.map((m) => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content })),
        community_identifier: communityId || undefined,
      }),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const j = await res.json(); detail = j.detail || detail; } catch { /* ignore */ }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return await res.json();
  } catch (e) {
    if (!USE_DEMO_FALLBACK) throw e;
    setOfflineFallbackDemo(true);
    await new Promise((r) => setTimeout(r, 900));
    const crit = DEMO.items.filter((x) => x.priority_band === 'BAND_1_CRITICAL').slice(0, 5);
    return {
      answer:
        `**Demo response** — the assistant backend is offline, so here is a grounded sample answer.\n\n` +
        `The most critical units right now are:\n\n` +
        crit.map((c, i) => `${i + 1}. **${c.village_name}** (${c.district_name}) — score \`${c.preparedness_score.toFixed(2)}\`, shelter ${c.dist_nearest_shelter_km.toFixed(1)} km away`).join('\n') +
        `\n\nRecommended action: pre-position shelters and mobile medical units for Band 1 units before landfall.`,
      tool_executions: [
        { tool_name: 'get_priority_rankings', arguments: { band: 'BAND_1_CRITICAL', limit: 5 }, success: true, execution_time_ms: 42 },
        { tool_name: 'get_community_profile', arguments: { id: crit[0]?.area_id }, success: true, execution_time_ms: 18 },
      ],
      model_name: 'demo-fallback', total_duration_ms: 910,
    };
  }
}

/* ───────────────────────────── STYLES ───────────────────────────── */

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
.cx-root{--glass:rgba(255,255,255,.12);--glass-2:rgba(255,255,255,.18);--line:rgba(255,255,255,.22);--line-soft:rgba(255,255,255,.12);
  --ink:#fff;--ink-2:rgba(255,255,255,.78);--ink-3:rgba(255,255,255,.55);--accent:#9BE7D8;
  position:fixed;inset:0;overflow:hidden;font-family:'Plus Jakarta Sans',Inter,system-ui,sans-serif;color:var(--ink);
  background:#5d7472;display:flex;align-items:center;justify-content:center;letter-spacing:-.01em;-webkit-font-smoothing:antialiased}
.cx-root *{box-sizing:border-box}
.cx-root h1,.cx-root h2,.cx-root h3{color:#fff;font-family:inherit}
.cx-root ::-webkit-scrollbar{width:6px;height:6px}.cx-root ::-webkit-scrollbar-thumb{background:rgba(255,255,255,.25);border-radius:9px}
.cx-root ::-webkit-scrollbar-track{background:transparent}
.cx-outer-bg{position:absolute;inset:-40px;filter:blur(28px) saturate(.8) brightness(.9);opacity:.9;transform:scale(1.06)}
.cx-frame{position:relative;width:min(1480px,calc(100vw - 48px));height:min(900px,calc(100vh - 48px));border-radius:30px;overflow:hidden;
  box-shadow:0 40px 120px -20px rgba(10,25,28,.65),0 0 0 1px rgba(255,255,255,.08) inset;isolation:isolate}
@media (max-width:820px){.cx-frame{width:100vw;height:100vh;border-radius:0}}

/* ── SKY ── */
.cx-sky{position:absolute;inset:0;overflow:hidden;background:linear-gradient(180deg,#182a2d 0%,#34514f 34%,#7f9488 56%,#d8c597 66%,#b9b07a 69%,#3c5a2a 70%)}
.cx-sky-photo{position:absolute;inset:0;background-size:cover;background-position:center}
.cx-cloud-layer{position:absolute;left:-12%;width:124%;will-change:transform}
.cx-cloud-a{top:-8%;height:82%;animation:cxDrift 90s ease-in-out infinite alternate}
.cx-cloud-b{top:-4%;height:70%;opacity:.75;animation:cxDrift2 120s ease-in-out infinite alternate;mix-blend-mode:multiply}
.cx-cloud-layer svg{width:100%;height:100%;display:block}
@keyframes cxDrift{from{transform:translate3d(-2%,0,0) scale(1.02)}to{transform:translate3d(3%,1%,0) scale(1.06)}}
@keyframes cxDrift2{from{transform:translate3d(3%,0,0)}to{transform:translate3d(-4%,-1%,0)}}
.cx-horizon-glow{position:absolute;right:-5%;top:42%;width:60%;height:34%;background:radial-gradient(ellipse at 70% 60%,rgba(255,226,160,.85),rgba(255,210,140,.35) 35%,transparent 70%);filter:blur(10px);mix-blend-mode:screen}
.cx-rain{position:absolute;top:22%;height:50%;background:repeating-linear-gradient(103deg,rgba(220,235,240,.0) 0 7px,rgba(220,235,240,.28) 7px 8px);
  -webkit-mask-image:linear-gradient(180deg,transparent,#000 25%,#000 70%,transparent);mask-image:linear-gradient(180deg,transparent,#000 25%,#000 70%,transparent);
  animation:cxRain .5s linear infinite;filter:blur(.6px)}
.cx-rain.r1{left:58%;width:16%;opacity:.55}.cx-rain.r2{left:74%;width:14%;opacity:.4;animation-duration:.65s}.cx-rain.r3{left:12%;width:10%;opacity:.25;animation-duration:.8s}
@keyframes cxRain{from{background-position:0 0}to{background-position:-18px 80px}}
.cx-bolt{position:absolute;inset:0;pointer-events:none;opacity:0;animation:cxBolt 9s infinite}
.cx-bolt.b2{animation-delay:4.3s;animation-duration:11s}
@keyframes cxBolt{0%,86%,100%{opacity:0}87%{opacity:1}88%{opacity:.2}89.5%{opacity:1}92%{opacity:0}}
.cx-flash{position:absolute;inset:0;background:radial-gradient(ellipse at 45% 35%,rgba(225,240,255,.35),transparent 60%);opacity:0;animation:cxBolt 9s infinite;mix-blend-mode:screen;pointer-events:none}
.cx-treeline{position:absolute;left:0;right:0;top:63.5%;height:8%}
.cx-field{position:absolute;left:-30%;right:-30%;top:69.5%;bottom:-40%;transform-origin:50% 0;transform:perspective(500px) rotateX(62deg);
  background:repeating-linear-gradient(90deg,rgba(18,40,12,.55) 0 3px,rgba(0,0,0,0) 3px 22px),
  linear-gradient(180deg,#6f8a3c 0%,#4f7428 25%,#355a1d 60%,#1f3d12 100%)}
.cx-field-shade{position:absolute;left:0;right:0;top:69%;bottom:0;background:linear-gradient(180deg,rgba(214,200,140,.25),rgba(0,0,0,0) 25%,rgba(5,14,8,.35))}
.cx-vignette{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 40%,transparent 50%,rgba(8,18,20,.45) 100%),linear-gradient(90deg,rgba(10,22,24,.35),transparent 35%,transparent 70%,rgba(10,22,24,.15))}
.cx-dim{position:absolute;inset:0;background:rgba(14,26,28,.0);backdrop-filter:blur(0px);transition:all .6s ease;pointer-events:none}
.cx-dim.on{background:rgba(14,26,28,.45);backdrop-filter:blur(14px) saturate(1.1)}

/* ── GLASS ── */
.cx-glass{background:linear-gradient(160deg,rgba(255,255,255,.22),rgba(255,255,255,.08));border:1px solid var(--line);
  backdrop-filter:blur(22px) saturate(1.3);-webkit-backdrop-filter:blur(22px) saturate(1.3);
  box-shadow:0 1px 0 rgba(255,255,255,.25) inset,0 20px 50px -20px rgba(0,0,0,.35)}
.cx-glass-dark{background:linear-gradient(160deg,rgba(20,38,40,.55),rgba(20,38,40,.35));border:1px solid var(--line-soft);
  backdrop-filter:blur(24px) saturate(1.2);-webkit-backdrop-filter:blur(24px) saturate(1.2)}

/* ── SHELL ── */
.cx-shell{position:absolute;inset:0;display:grid;grid-template-columns:76px 1fr;gap:22px;padding:22px 26px 22px 22px;z-index:2}
.cx-side{border-radius:22px;display:flex;flex-direction:column;align-items:center;padding:14px 0;gap:6px}
.cx-logo{width:44px;height:44px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle at 30% 30%,rgba(255,255,255,.55),rgba(255,255,255,.12));border:1px solid rgba(255,255,255,.4);margin-bottom:18px}
.cx-nav{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;color:var(--ink-2);background:transparent;border:0;cursor:pointer;position:relative;transition:all .25s}
.cx-nav:hover{background:rgba(255,255,255,.14);color:#fff}
.cx-nav.on{background:rgba(255,255,255,.92);color:#20393b;box-shadow:0 8px 20px -6px rgba(0,0,0,.35)}
.cx-nav .tip{position:absolute;left:54px;white-space:nowrap;font-size:12px;font-weight:600;padding:6px 10px;border-radius:9px;background:rgba(15,30,32,.85);color:#fff;opacity:0;transform:translateX(-6px);pointer-events:none;transition:.2s;z-index:40}
.cx-nav:hover .tip{opacity:1;transform:none}
.cx-main{display:flex;flex-direction:column;min-width:0;min-height:0}
.cx-top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:18px}
.cx-hello small{display:block;font-size:12px;color:var(--ink-2);font-weight:500}
.cx-hello b{font-size:15px;font-weight:700}
.cx-iconbtn{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;color:#fff;cursor:pointer;border:1px solid var(--line);
  background:rgba(255,255,255,.14);backdrop-filter:blur(14px);transition:.2s;position:relative}
.cx-iconbtn:hover{background:rgba(255,255,255,.28)}
.cx-dot{position:absolute;top:9px;right:10px;width:8px;height:8px;border-radius:50%;background:#FF6B5E;box-shadow:0 0 0 2px rgba(40,60,60,.6)}
.cx-avatar{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:14px;color:#24403f;background:linear-gradient(135deg,#fff,#cfe9e2);border:2px solid rgba(255,255,255,.7)}
.cx-pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.14);backdrop-filter:blur(10px);color:#fff}
.cx-pill.warn{background:rgba(245,169,74,.22);border-color:rgba(245,169,74,.5)}
.cx-content{flex:1;min-height:0;position:relative}
.cx-view{position:absolute;inset:0;animation:cxIn .55s cubic-bezier(.2,.8,.2,1)}
@keyframes cxIn{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.cx-scroll{overflow:auto;height:100%;padding-right:6px}

/* ── DEMO PRESENTATION DOCK & PROMPTS ── */
.cx-demo-dock{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:12px 18px;border-radius:18px;margin-bottom:14px;background:linear-gradient(135deg,rgba(155,231,216,.18),rgba(20,38,40,.65));border:1px solid rgba(155,231,216,.4);backdrop-filter:blur(20px);animation:cxIn .35s ease;flex-wrap:wrap}
.cx-demo-step-pill{width:28px;height:28px;border-radius:50%;display:grid;place-items:center;font-size:12px;font-weight:700;border:1px solid var(--line);background:rgba(255,255,255,.1);color:#fff;cursor:pointer;transition:.2s}
.cx-demo-step-pill.on{background:#9BE7D8;color:#182e2d;box-shadow:0 0 12px rgba(155,231,216,.6)}
.cx-demo-step-pill:hover:not(.on){background:rgba(255,255,255,.25)}
.cx-prompt-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;margin-top:8px}
.cx-prompt-card{padding:12px 14px;border-radius:14px;background:rgba(255,255,255,.08);border:1px solid var(--line);color:#fff;cursor:pointer;text-align:left;font-family:inherit;font-size:12.5px;line-height:1.45;transition:all .25s;display:flex;flex-direction:column;justify-content:space-between}
.cx-prompt-card:hover{background:rgba(155,231,216,.15);border-color:rgba(155,231,216,.5);transform:translateY(-2px)}

/* ── OVERVIEW HERO ── */
.cx-ov{display:grid;grid-template-columns:1fr 300px;gap:28px;height:100%}
.cx-ov-left{display:flex;flex-direction:column;min-width:0}
.cx-h1{font-size:clamp(34px,4.3vw,58px);line-height:1.04;font-weight:600;letter-spacing:-.035em;margin:14px 0 14px;text-shadow:0 2px 30px rgba(0,0,0,.25)}
.cx-lede{max-width:430px;font-size:13.5px;line-height:1.6;color:var(--ink-2)}
.cx-wave{position:relative;margin-top:auto;padding-bottom:6px}
.cx-wave-vals{display:grid;grid-template-columns:repeat(6,1fr)}
.cx-wave-val{display:flex;align-items:center;justify-content:center;gap:6px;font-size:clamp(20px,2.1vw,28px);font-weight:500;cursor:pointer;opacity:.85;transition:.25s;background:none;border:0;color:#fff;font-family:inherit;letter-spacing:-.02em}
.cx-wave-val:hover{opacity:1}.cx-wave-val.on{opacity:1;font-weight:700}
.cx-wave-val svg{opacity:.85}
.cx-wave-svg{width:100%;height:150px;display:block;overflow:visible}
.cx-wave-labels{display:grid;grid-template-columns:repeat(6,1fr);margin-top:6px}
.cx-wave-labels span{text-align:center;font-size:13px;color:var(--ink-2);font-weight:500}
.cx-wave-labels span.on{color:#fff;font-weight:700}
.cx-ov-right{display:flex;flex-direction:column;gap:14px;min-height:0}
.cx-loc{border-radius:22px;padding:18px 20px}
.cx-loc-big{font-size:64px;font-weight:600;letter-spacing:-.05em;line-height:1;margin:14px 0 14px}
.cx-loc-big{display:flex;align-items:flex-start}.cx-loc-big .u{font-size:20px;font-weight:600;margin:6px 0 0 6px;opacity:.85;letter-spacing:0}
.cx-loc-stats{display:flex;gap:14px;font-size:11.5px;color:var(--ink-2);flex-wrap:wrap}
.cx-loc-stats span{display:inline-flex;align-items:center;gap:5px}
.cx-city{border-radius:20px;padding:14px 18px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;transition:.25s;border:1px solid var(--line)}
.cx-city:hover{transform:translateX(-4px);background:linear-gradient(160deg,rgba(255,255,255,.3),rgba(255,255,255,.12))}
.cx-city small{font-size:11px;color:var(--ink-3);display:block}
.cx-city b{font-size:15px;display:block;margin:3px 0 2px}
.cx-city i{font-style:normal;font-size:12px;color:var(--ink-2)}
.cx-city-val{font-size:32px;font-weight:500;letter-spacing:-.03em;display:flex;align-items:center;gap:6px}

/* ── GENERIC PANELS ── */
.cx-card{border-radius:22px;padding:20px}
.cx-h2{font-size:28px;font-weight:700;letter-spacing:-.03em;margin:0}
.cx-sub{font-size:13px;color:var(--ink-2);margin:4px 0 0}
.cx-label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink-3);font-weight:700}
.cx-grid{display:grid;gap:16px}
.cx-kpi b{display:block;font-size:30px;font-weight:700;letter-spacing:-.03em;margin-top:6px}
.cx-kpi span{font-size:12px;color:var(--ink-2)}
.cx-row{display:flex;align-items:center;gap:10px}
.cx-between{display:flex;align-items:center;justify-content:space-between;gap:12px}
.cx-btn{display:inline-flex;align-items:center;gap:8px;padding:10px 16px;border-radius:12px;font-weight:600;font-size:13px;cursor:pointer;border:1px solid var(--line);background:rgba(255,255,255,.14);color:#fff;font-family:inherit;transition:.2s}
.cx-btn:hover{background:rgba(255,255,255,.26)}
.cx-btn.primary{background:#fff;color:#1e3a3b;border-color:#fff}
.cx-btn.primary:hover{background:#e7f6f2}
.cx-btn:disabled{opacity:.45;cursor:not-allowed}
.cx-input,.cx-select{height:40px;border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.12);color:#fff;padding:0 14px;font-family:inherit;font-size:13px;outline:none;transition:.2s}
.cx-input::placeholder{color:var(--ink-3)}
.cx-input:focus,.cx-select:focus{border-color:rgba(255,255,255,.6);background:rgba(255,255,255,.18)}
.cx-select option{color:#14292a}
.cx-chip{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:11.5px;font-weight:700;white-space:nowrap}
.cx-dotc{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.cx-table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px}
.cx-table th{text-align:left;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);font-weight:700;padding:12px 14px;position:sticky;top:0;background:rgba(30,52,54,.85);backdrop-filter:blur(10px);z-index:1;cursor:pointer;user-select:none}
.cx-table td{padding:12px 14px;border-top:1px solid var(--line-soft)}
.cx-table tbody tr{cursor:pointer;transition:background .15s}
.cx-table tbody tr:hover{background:rgba(255,255,255,.08)}
.cx-bar{height:6px;border-radius:9px;background:rgba(255,255,255,.14);overflow:hidden}
.cx-bar>i{display:block;height:100%;border-radius:9px;transition:width .8s cubic-bezier(.2,.8,.2,1)}
.cx-stack{display:flex;height:10px;border-radius:9px;overflow:hidden;gap:2px}
.cx-stack>i{display:block;height:100%}
.cx-mono{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:12px}
.cx-skel{border-radius:14px;background:linear-gradient(90deg,rgba(255,255,255,.08),rgba(255,255,255,.18),rgba(255,255,255,.08));background-size:200% 100%;animation:cxSk 1.4s infinite}
@keyframes cxSk{from{background-position:200% 0}to{background-position:-200% 0}}
.cx-spin{animation:cxSpin 1s linear infinite}@keyframes cxSpin{to{transform:rotate(360deg)}}
.cx-err{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:40px;text-align:center;color:var(--ink-2)}

/* drawer */
.cx-drawer-bg{position:absolute;inset:0;background:rgba(8,18,20,.35);z-index:30;animation:cxFade .3s}
@keyframes cxFade{from{opacity:0}}
.cx-drawer{position:absolute;top:12px;right:12px;bottom:12px;width:min(480px,calc(100% - 24px));border-radius:24px;z-index:31;display:flex;flex-direction:column;animation:cxSlide .45s cubic-bezier(.2,.8,.2,1)}
@keyframes cxSlide{from{transform:translateX(40px);opacity:0}}

/* map */
.cx-map{border-radius:22px;overflow:hidden;height:100%;position:relative}
.cx-map .leaflet-container{background:#1b2d2f!important;height:100%;width:100%;font-family:inherit}
.cx-map .leaflet-tooltip{background:rgba(16,32,34,.9);color:#fff;border:1px solid rgba(255,255,255,.2);border-radius:10px;font-family:inherit;box-shadow:none}
.cx-map .leaflet-tooltip:before{display:none}
.cx-map .leaflet-control-zoom a{background:rgba(255,255,255,.85);color:#1e3a3b;border:0}
.cx-map .leaflet-control-attribution{background:rgba(0,0,0,.35);color:rgba(255,255,255,.6)}
.cx-map .leaflet-control-attribution a{color:rgba(255,255,255,.8)}

/* chat */
.cx-msg{max-width:78%;padding:14px 16px;border-radius:18px;font-size:14px;line-height:1.6;animation:cxIn .35s}
.cx-msg.user{margin-left:auto;background:#fff;color:#1b3436;border-bottom-right-radius:6px}
.cx-msg.bot{border-bottom-left-radius:6px}
.cx-msg p{margin:0 0 8px}.cx-msg p:last-child{margin:0}
.cx-msg ul,.cx-msg ol{margin:6px 0;padding-left:20px}.cx-msg ul{list-style:disc}.cx-msg ol{list-style:decimal}.cx-msg strong{font-weight:700}
.cx-msg code{font-family:'JetBrains Mono',monospace;font-size:12px;background:rgba(255,255,255,.14);padding:1px 6px;border-radius:6px}
.cx-msg table{border-collapse:collapse;font-size:12.5px;margin:8px 0}.cx-msg td,.cx-msg th{border:1px solid rgba(255,255,255,.2);padding:5px 8px}
.cx-typing span{width:7px;height:7px;border-radius:50%;background:#fff;display:inline-block;margin-right:4px;animation:cxTy 1.2s infinite}
.cx-typing span:nth-child(2){animation-delay:.15s}.cx-typing span:nth-child(3){animation-delay:.3s}
@keyframes cxTy{0%,60%,100%{opacity:.25;transform:translateY(0)}30%{opacity:1;transform:translateY(-4px)}}
.cx-suggest{text-align:left;padding:14px 16px;border-radius:16px;cursor:pointer;font-family:inherit;color:#fff;font-size:13px;line-height:1.45;transition:.2s}
.cx-suggest:hover{transform:translateY(-2px);background:linear-gradient(160deg,rgba(255,255,255,.3),rgba(255,255,255,.12))}

@media (max-width:1100px){.cx-ov{grid-template-columns:1fr}.cx-ov-right{display:grid;grid-template-columns:repeat(2,1fr)}.cx-ov{overflow:auto}}
@media (max-width:820px){.cx-shell{grid-template-columns:1fr;padding:14px;gap:12px}.cx-side{flex-direction:row;justify-content:space-around;order:2;padding:8px}.cx-logo{display:none}.cx-side .cx-spacer{display:none}.cx-main{order:1}.cx-ov-right{grid-template-columns:1fr}.cx-nav .tip{display:none}}
`;

/* ───────────────────────────── STORM SKY ───────────────────────────── */

function treePath(seed: number, base: number, minH: number, maxH: number) {
  const r = mulberry32(seed);
  let d = `M0 80 L0 ${base}`; let x = 0;
  while (x < 1600) {
    const cluster = r() < 0.18;
    const w = cluster ? 10 + r() * 18 : 5 + r() * 9;
    const h = cluster ? maxH * (0.7 + r() * 0.3) : minH + r() * (maxH - minH) * 0.5;
    d += ` Q${(x + w / 2).toFixed(1)} ${(base - h * 2).toFixed(1)} ${(x + w).toFixed(1)} ${base}`;
    x += w * (0.55 + r() * 0.3);
  }
  return d + ' L1600 80 Z';
}
const TREES_BACK = treePath(7, 62, 6, 26);
const TREES_FRONT = treePath(19, 70, 3, 16);

const StormSky: React.FC<{ id: string }> = React.memo(({ id }) => {
  if (BACKGROUND_PHOTO_URL) {
    return (
      <div className="cx-sky">
        <div className="cx-sky-photo" style={{ backgroundImage: `url(${BACKGROUND_PHOTO_URL})` }} />
        <div className="cx-rain r1" /><div className="cx-flash" />
      </div>
    );
  }
  return (
    <div className="cx-sky">
      <div className="cx-horizon-glow" />
      {/* main storm cell */}
      <div className="cx-cloud-layer cx-cloud-a">
        <svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
          <defs>
            <filter id={`${id}-cf`} x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
              <feTurbulence type="fractalNoise" baseFrequency="0.0026 0.0052" numOctaves={6} seed={14} result="n" />
              <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  3.4 0 0 0 -1.32" result="shape" />
              <feDiffuseLighting in="n" surfaceScale={13} diffuseConstant={1.25} lightingColor="#e4efe9" result="lit">
                <feDistantLight azimuth={250} elevation={34} />
              </feDiffuseLighting>
              <feComposite in="lit" in2="shape" operator="in" />
            </filter>
            <radialGradient id={`${id}-cm`} cx="48%" cy="38%" r="62%">
              <stop offset="0" stopColor="#fff" stopOpacity="1" />
              <stop offset=".6" stopColor="#fff" stopOpacity=".9" />
              <stop offset="1" stopColor="#fff" stopOpacity="0" />
            </radialGradient>
            <mask id={`${id}-m`}><rect width="1600" height="900" fill={`url(#${id}-cm)`} /></mask>
          </defs>
          <g mask={`url(#${id}-m)`}>
            <rect width="1600" height="900" filter={`url(#${id}-cf)`} />
          </g>
        </svg>
      </div>
      {/* dark under-belly */}
      <div className="cx-cloud-layer cx-cloud-b">
        <svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
          <defs>
            <filter id={`${id}-df`} x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
              <feTurbulence type="fractalNoise" baseFrequency="0.004 0.009" numOctaves={5} seed={3} result="n" />
              <feColorMatrix in="n" type="matrix" values="0 0 0 0 0.10  0 0 0 0 0.19  0 0 0 0 0.20  3 0 0 0 -1.25" />
            </filter>
            <linearGradient id={`${id}-dm`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#fff" stopOpacity=".95" />
              <stop offset=".75" stopColor="#fff" stopOpacity=".7" />
              <stop offset="1" stopColor="#fff" stopOpacity="0" />
            </linearGradient>
            <mask id={`${id}-m2`}><rect width="1600" height="900" fill={`url(#${id}-dm)`} /></mask>
          </defs>
          <g mask={`url(#${id}-m2)`}><rect width="1600" height="900" filter={`url(#${id}-df)`} /></g>
        </svg>
      </div>
      <div className="cx-rain r1" /><div className="cx-rain r2" /><div className="cx-rain r3" />
      {/* lightning */}
      <svg className="cx-bolt" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
        <defs><filter id={`${id}-g`}><feGaussianBlur stdDeviation="4" /></filter></defs>
        {[0, 1].map((k) => (
          <g key={k} filter={k === 0 ? `url(#${id}-g)` : undefined} stroke={k === 0 ? '#bfe3ff' : '#fff'} strokeWidth={k === 0 ? 7 : 2} fill="none" strokeLinejoin="round">
            <path d="M720 300 L700 380 L728 392 L690 470 L716 478 L676 560 L702 568 L690 630" />
            <path d="M716 478 L752 520 L746 560" strokeWidth={k === 0 ? 4 : 1.2} />
            <path d="M700 380 L660 420" strokeWidth={k === 0 ? 4 : 1} />
          </g>
        ))}
      </svg>
      <svg className="cx-bolt b2" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
        <g stroke="#e6f3ff" strokeWidth={1.6} fill="none">
          <path d="M1120 330 L1104 410 L1126 420 L1100 500 L1118 510 L1092 600" />
          <path d="M1118 510 L1150 560" strokeWidth={1} />
        </g>
      </svg>
      <div className="cx-flash" />
      {/* tree line */}
      <svg className="cx-treeline" viewBox="0 0 1600 80" preserveAspectRatio="none">
        <path fill="#3b5a35" opacity={0.7} d={TREES_BACK} />
        <path fill="#22401f" d={TREES_FRONT} />
      </svg>
      <div className="cx-field" />
      <div className="cx-field-shade" />
      <div className="cx-vignette" />
    </div>
  );
});

/* ───────────────────────────── PRIMITIVES ───────────────────────────── */

const BandChip: React.FC<{ band?: string }> = ({ band }) => {
  const b = BAND[band || ''] || { hex: '#aaa', short: band || '—' };
  return <span className="cx-chip" style={{ background: `${b.hex}26`, color: b.hex, border: `1px solid ${b.hex}55` }}><span className="cx-dotc" style={{ background: b.hex }} />{b.short}</span>;
};
const TierChip: React.FC<{ tier?: string }> = ({ tier }) => {
  const c = TIER[tier || ''] || '#ccc';
  return <span className="cx-chip" style={{ background: `${c}22`, color: c, border: `1px solid ${c}55` }}>{tier || '—'}</span>;
};
const Skel: React.FC<{ h?: number; w?: string | number; r?: number; style?: React.CSSProperties }> = ({ h = 16, w = '100%', r, style }) => (
  <div className="cx-skel" style={{ height: h, width: w, borderRadius: r, ...style }} />
);
const ErrorBox: React.FC<{ msg: string; onRetry?: () => void }> = ({ msg, onRetry }) => (
  <div className="cx-err">
    <AlertTriangle size={30} color="#F5A94A" />
    <div style={{ fontWeight: 700, color: '#fff' }}>Couldn't load this data</div>
    <div style={{ fontSize: 13, maxWidth: 360 }}>{msg}</div>
    {onRetry && <button className="cx-btn" onClick={onRetry}>Retry</button>}
  </div>
);
const ScoreBar: React.FC<{ v: number; color?: string }> = ({ v, color }) => {
  const norm = Number(v) > 1 ? Number(v) / 100 : Number(v);
  const clamped = Math.max(0, Math.min(1, Number.isNaN(norm) ? 0 : norm));
  return (
    <div className="cx-bar"><i style={{ width: `${clamped * 100}%`, background: color || scoreColor(v) }} /></div>
  );
};
function scoreColor(v: number) {
  const val = Number(v) <= 1.0 && Number(v) > 0 ? Number(v) * 100 : Number(v) || 0;
  return val < 40 ? BAND.BAND_1_CRITICAL.hex : val < 50 ? BAND.BAND_2_SPECIALIZED.hex : val < 60 ? BAND.BAND_3_MODERATE.hex : BAND.BAND_4_ADEQUATE.hex;
}
function scoreIcon(v: number, size = 18) {
  const val = Number(v) <= 1.0 && Number(v) > 0 ? Number(v) * 100 : Number(v) || 0;
  if (val < 40) return <CloudLightning size={size} />;
  if (val < 50) return <CloudRain size={size} />;
  if (val < 60) return <Cloudy size={size} />;
  return <Sun size={size} />;
}
function scoreWord(v: number) {
  const val = Number(v) <= 1.0 && Number(v) > 0 ? Number(v) * 100 : Number(v) || 0;
  return val < 40 ? 'Critical' : val < 50 ? 'Vulnerable' : val < 60 ? 'Moderate' : 'Adequate';
}

const ViewHeader: React.FC<{ eyebrow: string; title: string; sub?: string; right?: React.ReactNode }> = ({ eyebrow, title, sub, right }) => (
  <div className="cx-between" style={{ marginBottom: 18, flexWrap: 'wrap' }}>
    <div>
      <span className="cx-pill">{eyebrow}</span>
      <h2 className="cx-h2" style={{ marginTop: 10 }}>{title}</h2>
      {sub && <p className="cx-sub">{sub}</p>}
    </div>
    {right}
  </div>
);

const chartTip = {
  contentStyle: { background: 'rgba(16,32,34,.92)', border: '1px solid rgba(255,255,255,.2)', borderRadius: 12, color: '#fff', fontFamily: 'Plus Jakarta Sans', fontSize: 12 },
  itemStyle: { color: '#fff' }, labelStyle: { color: 'rgba(255,255,255,.7)' }, cursor: { fill: 'rgba(255,255,255,.06)' },
};

/* ───────────────────────────── WAVE (hero chart) ───────────────────────────── */

function smoothPath(pts: [number, number][]) {
  if (pts.length < 2) return '';
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${c1x},${c1y} ${c2x},${c2y} ${p2[0]},${p2[1]}`;
  }
  return d;
}

const HeroWave: React.FC<{ items: { key: string; label: string; value: number }[]; active: number; onPick: (i: number) => void }> = ({ items, active, onPick }) => {
  const W = 1000, H = 150, n = items.length;
  const xs = items.map((_, i) => (W / n) * (i + 0.5));
  const vmin = Math.min(...items.map((x) => x.value)), vmax = Math.max(...items.map((x) => x.value));
  const ys = items.map((x) => 40 + (1 - (x.value - vmin) / Math.max(vmax - vmin, 0.01)) * 70);
  const pts: [number, number][] = [[-40, ys[0] + 12], ...xs.map((x, i) => [x, ys[i]] as [number, number]), [W + 40, ys[n - 1] + 12]];
  const d = smoothPath(pts);
  const ax = xs[active], ay = ys[active];
  return (
    <div className="cx-wave">
      <div className="cx-wave-vals">
        {items.map((it, i) => (
          <button key={it.key} className={cx('cx-wave-val', i === active && 'on')} onClick={() => onPick(i)}>
            {it.value.toFixed(2)}{scoreIcon(it.value, 17)}
          </button>
        ))}
      </div>
      <svg className="cx-wave-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="cxWaveStroke" x1="0" x2="1">
            <stop offset="0" stopColor="#fff" stopOpacity=".15" />
            <stop offset=".5" stopColor="#fff" stopOpacity=".95" />
            <stop offset="1" stopColor="#fff" stopOpacity=".15" />
          </linearGradient>
          <linearGradient id="cxBeam" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#fff" stopOpacity="0" />
            <stop offset=".5" stopColor="#fff" stopOpacity=".9" />
            <stop offset="1" stopColor="#fff" stopOpacity="0" />
          </linearGradient>
          <radialGradient id="cxFlare"><stop offset="0" stopColor="#fff" /><stop offset=".25" stopColor="#fff" stopOpacity=".7" /><stop offset="1" stopColor="#fff" stopOpacity="0" /></radialGradient>
        </defs>
        <path d={d} fill="none" stroke="rgba(255,255,255,.25)" strokeWidth={6} style={{ filter: 'blur(4px)' }} />
        <path d={d} fill="none" stroke="url(#cxWaveStroke)" strokeWidth={2} />
        <g style={{ transition: 'transform .5s cubic-bezier(.2,.8,.2,1)', transform: `translate(${ax}px,0)` }}>
          <rect x={-1} y={-30} width={2} height={H + 30} fill="url(#cxBeam)" />
          <g style={{ transition: 'transform .5s cubic-bezier(.2,.8,.2,1)', transform: `translate(0,${ay}px)` }}>
            <circle r={26} fill="url(#cxFlare)" opacity={0.75} />
            <rect x={-60} y={-1} width={120} height={2} fill="url(#cxBeam)" transform="rotate(90)" opacity={0} />
            <rect x={-55} y={-0.75} width={110} height={1.5} fill="#fff" opacity={0.6} style={{ filter: 'blur(1px)' }} />
            <circle r={5} fill="#fff" />
          </g>
        </g>
      </svg>
      <div className="cx-wave-labels">
        {items.map((it, i) => <span key={it.key} className={i === active ? 'on' : ''} onClick={() => onPick(i)} style={{ cursor: 'pointer' }}>{it.label}</span>)}
      </div>
    </div>
  );
};

/* ───────────────────────────── VIEWS ───────────────────────────── */

const OverviewView: React.FC<{ go: (v: ViewKey, opts?: any) => void }> = ({ go }) => {
  const ov = useApi<any>('/dashboard/overview');
  const prep = useApi<any>('/analytics/preparedness');
  const [active, setActive] = useState<number | null>(null);

  const waveItems = useMemo(() => {
    const dom = prep.data?.domain_averages || {};
    const arr = Object.keys(DOMAIN_META).map((k) => ({ key: k, label: DOMAIN_META[k].label, value: Number(dom[k] ?? 0) }));
    const overall = Number(prep.data?.score_summary?.mean ?? 0);
    arr.splice(3, 0, { key: 'overall', label: 'Overall', value: overall });
    return arr;
  }, [prep.data]);

  const weakest = useMemo(() => {
    if (!prep.data) return 0;
    let wi = 0; waveItems.forEach((x, i) => { if (x.key !== 'overall' && x.value < waveItems[wi].value) wi = i; });
    return wi;
  }, [waveItems, prep.data]);
  const act = active ?? weakest;

  if (ov.error) return <ErrorBox msg={ov.error} onRetry={ov.retry} />;
  const d = ov.data;
  const bc = d?.priority_band_counts || {};
  const crit = bc.BAND_1_CRITICAL || 0;
  const hi = d?.infrastructure_highlights || {};
  const meanScore = prep.data?.score_summary?.mean ?? (d ? d.districts.reduce((a: number, x: any) => a + x.mean_preparedness_score * x.total_units, 0) / Math.max(d.total_units, 1) : 0);
  const sel = waveItems[act];
  const headline = crit > 0 ? ['Preparedness Priorities', 'Coastal Vulnerability Watch'] : ['Coastal Readiness', 'Baseline Maintained'];

  return (
    <div className="cx-ov">
      <div className="cx-ov-left">
        <div><span className="cx-pill"><Zap size={13} /> Preparedness Overview</span></div>
        {d ? (
          <>
            <h1 className="cx-h1">{headline[0]}<br />{headline[1]}</h1>
            <p className="cx-lede">
              {fmt(d.total_units)} administrative units across {d.districts_count} coastal districts. Mean AHP preparedness sits at{' '}
              <b>{Number(meanScore).toFixed(2)}</b>, with <b>{fmt(crit)}</b> units in Band 1 critical and{' '}
              {fmt(hi.coastal_units_under_5km)} villages within 5 km of the Arabian Sea. Weakest domain:{' '}
              <b>{waveItems[weakest]?.label}</b> ({waveItems[weakest]?.value.toFixed(2)}).
            </p>
            <div className="cx-row" style={{ marginTop: 18, gap: 10, flexWrap: 'wrap' }}>
              <button className="cx-btn primary" onClick={() => go('communities', { band: 'BAND_1_CRITICAL' })}>View critical units <ArrowRight size={15} /></button>
              <button className="cx-btn" onClick={() => go('assistant')}><Sparkles size={15} /> Ask the assistant</button>
            </div>
          </>
        ) : (
          <div style={{ marginTop: 18 }}><Skel h={56} w="70%" /><Skel h={56} w="55%" style={{ marginTop: 10 }} /><Skel h={14} w="40%" style={{ marginTop: 18 }} /><Skel h={14} w="36%" style={{ marginTop: 8 }} /></div>
        )}
        {prep.data ? (
          <>
            <div style={{ marginTop: 'auto' }} />
            <div className="cx-row" style={{ justifyContent: 'center', marginBottom: 6, fontSize: 12.5, color: 'rgba(255,255,255,.8)' }}>
              {sel && <span key={sel.key} style={{ animation: 'cxIn .4s' }}>{sel.label} domain · regional mean <b style={{ color: '#fff' }}>{sel.value.toFixed(2)}</b> · {scoreWord(sel.value)} conditions</span>}
            </div>
            <HeroWave items={waveItems} active={act} onPick={setActive} />
          </>
        ) : <div style={{ marginTop: 'auto' }}><Skel h={150} /></div>}
      </div>

      <div className="cx-ov-right">
        <div className="cx-loc cx-glass">
          <div className="cx-row" style={{ fontWeight: 700, fontSize: 14 }}><MapPin size={16} /> Coastal Karnataka</div>
          {d ? (
            <>
              <div className="cx-loc-big">{Number(meanScore).toFixed(2)}<span className="u">AHP</span></div>
              <div className="cx-loc-stats">
                <span><Building2 size={13} />{fmt(d.total_units)}</span>
                <span><AlertTriangle size={13} />{fmt(crit)}</span>
                <span><Home size={13} />{fmt(hi.mean_dist_nearest_shelter_km, 1)} km</span>
                <span><Waves size={13} />{fmt(hi.mean_dist_coast_km, 1)} km</span>
              </div>
            </>
          ) : <><Skel h={60} w={150} style={{ margin: '14px 0' }} /><Skel h={12} /></>}
        </div>
        {(d?.districts || [null, null, null]).map((dist: any, i: number) => (
          <div key={dist?.district_name || i} className="cx-city cx-glass" onClick={() => dist && go('districts', { district: dist.district_name })}>
            {dist ? (
              <>
                <div>
                  <small>Karnataka</small>
                  <b>{dist.district_name}</b>
                  <i>{fmt(dist.total_units)} units · {fmt(dist.critical_units_count)} critical</i>
                </div>
                <div className="cx-city-val">{Number(dist.mean_preparedness_score).toFixed(2)}{scoreIcon(dist.mean_preparedness_score, 18)}</div>
              </>
            ) : <div style={{ width: '100%' }}><Skel h={12} w="40%" /><Skel h={16} w="70%" style={{ marginTop: 8 }} /><Skel h={12} w="50%" style={{ marginTop: 8 }} /></div>}
          </div>
        ))}
        <button className="cx-btn" style={{ justifyContent: 'center', marginTop: 'auto' }} onClick={() => go('map')}><MapIcon size={15} /> Open preparedness map</button>
      </div>
    </div>
  );
};

/* ── MAP ── */
const MapFly: React.FC<{ district: string }> = ({ district }) => {
  const map = useMap();
  useEffect(() => { const v = DISTRICT_VIEW[district] || DISTRICT_VIEW['']; map.flyTo(v.c, v.z, { duration: 1.1 }); }, [district, map]);
  return null;
};

const MapView: React.FC<{ openCommunity: (id: string) => void }> = ({ openCommunity }) => {
  const [district, setDistrict] = useState('');
  const [band, setBand] = useState('');
  const qs = new URLSearchParams({ ...(district && { district }), ...(band && { priority_band: band }) }).toString();
  const { data, loading, error, retry } = useApi<any>(`/maps/communities${qs ? `?${qs}` : ''}`);
  const pts: any[] = data?.points || [];
  const [hover, setHover] = useState<any>(null);
  const counts = useMemo(() => pts.reduce((m: any, p: any) => ((m[p.band] = (m[p.band] || 0) + 1), m), {}), [pts]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 290px', gap: 18, height: '100%' }}>
      <div className="cx-map cx-glass">
        <MapContainer center={DISTRICT_VIEW[''].c} zoom={8} preferCanvas zoomControl style={{ height: '100%' }}>
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; OpenStreetMap &copy; CARTO' />
          <MapFly district={district} />
          {pts.map((p) => {
            const c = BAND[p.band]?.hex || '#ccc';
            return (
              <CircleMarker key={p.area_id} center={[p.lat, p.lon]} radius={p.band === 'BAND_1_CRITICAL' ? 5.5 : 4}
                pathOptions={{ color: c, fillColor: c, fillOpacity: 0.75, weight: 1, opacity: 0.9 }}
                eventHandlers={{ click: () => openCommunity(p.area_id), mouseover: () => setHover(p) }}>
                <LTooltip direction="top" offset={[0, -4]}>
                  <div style={{ fontWeight: 700 }}>{p.name}</div>
                  <div style={{ fontSize: 11, opacity: 0.8 }}>{p.district} · score {Number(p.score).toFixed(2)}</div>
                </LTooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>
        {loading && <div style={{ position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 500 }} className="cx-pill"><Loader2 size={14} className="cx-spin" /> Loading units…</div>}
        {error && <div style={{ position: 'absolute', inset: 0, zIndex: 500, display: 'grid', placeItems: 'center', background: 'rgba(14,26,28,.6)' }}><ErrorBox msg={error} onRetry={retry} /></div>}
      </div>
      <div className="cx-scroll" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="cx-card cx-glass">
          <div className="cx-label">Community Preparedness Map</div>
          <div style={{ fontSize: 34, fontWeight: 700, letterSpacing: '-.03em', margin: '6px 0 2px' }}>{fmt(pts.length)}</div>
          <div style={{ fontSize: 12.5, color: 'rgba(255,255,255,.75)' }}>units plotted</div>
          <div style={{ marginTop: 16 }} className="cx-label">District</div>
          <div className="cx-row" style={{ flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {['', ...DISTRICTS].map((d) => (
              <button key={d || 'all'} className={cx('cx-chip')} onClick={() => setDistrict(d)}
                style={{ cursor: 'pointer', border: '1px solid rgba(255,255,255,.3)', background: district === d ? '#fff' : 'rgba(255,255,255,.1)', color: district === d ? '#1e3a3b' : '#fff', fontFamily: 'inherit' }}>
                {d || 'All'}
              </button>
            ))}
          </div>
          <div style={{ marginTop: 16 }} className="cx-label">Priority band</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
            {Object.entries(BAND).map(([k, b]) => (
              <button key={k} onClick={() => setBand(band === k ? '' : k)} className="cx-between"
                style={{ cursor: 'pointer', padding: '9px 12px', borderRadius: 12, border: `1px solid ${band === k ? b.hex : 'rgba(255,255,255,.15)'}`, background: band === k ? `${b.hex}26` : 'rgba(255,255,255,.06)', color: '#fff', fontFamily: 'inherit', fontSize: 13 }}>
                <span className="cx-row" style={{ gap: 8 }}><span className="cx-dotc" style={{ background: b.hex, boxShadow: `0 0 10px ${b.hex}` }} />{b.short}</span>
                <b>{band && band !== k ? '—' : fmt(counts[k] || 0)}</b>
              </button>
            ))}
          </div>
        </div>
        <div className="cx-card cx-glass" style={{ minHeight: 150 }}>
          <div className="cx-label">Hovered unit</div>
          {hover ? (
            <div style={{ marginTop: 10, animation: 'cxIn .3s' }} key={hover.area_id}>
              <div style={{ fontWeight: 700, fontSize: 17 }}>{hover.name}</div>
              <div style={{ fontSize: 12, color: 'rgba(255,255,255,.7)' }}>{hover.subdistrict ? `${hover.subdistrict}, ` : ''}{hover.district}</div>
              <div className="cx-row" style={{ marginTop: 10, gap: 6 }}><BandChip band={hover.band} /><TierChip tier={hover.tier} /></div>
              <div style={{ marginTop: 12 }}><ScoreBar v={hover.score} /></div>
              <div className="cx-between" style={{ fontSize: 12, marginTop: 8, color: 'rgba(255,255,255,.8)' }}>
                <span>Score {Number(hover.score).toFixed(2)}</span><span>Rank #{fmt(hover.priority_rank)}</span>
              </div>
              <button className="cx-btn primary" style={{ marginTop: 14, width: '100%', justifyContent: 'center' }} onClick={() => openCommunity(hover.area_id)}>Open dossier <ArrowUpRight size={14} /></button>
            </div>
          ) : <div style={{ fontSize: 13, color: 'rgba(255,255,255,.6)', marginTop: 10 }}>Hover any point for a quick read. Click to open the full dossier.</div>}
        </div>
      </div>
    </div>
  );
};

/* ── COMMUNITIES ── */
const CommunitiesView: React.FC<{ init?: any; openCommunity: (id: string) => void }> = ({ init, openCommunity }) => {
  const [search, setSearch] = useState('');
  const [debounced, setDebounced] = useState('');
  const [district, setDistrict] = useState(init?.district || '');
  const [band, setBand] = useState(init?.band || '');
  const [tier, setTier] = useState('');
  const [sort, setSort] = useState<{ by: string; order: 'asc' | 'desc' }>({ by: 'priority_rank', order: 'asc' });
  const [page, setPage] = useState(1);
  useEffect(() => { const t = setTimeout(() => setDebounced(search), 300); return () => clearTimeout(t); }, [search]);
  useEffect(() => setPage(1), [debounced, district, band, tier, sort]);
  const qs = new URLSearchParams({
    ...(debounced && { search: debounced }), ...(district && { district }), ...(band && { priority_band: band }), ...(tier && { tier }),
    sort_by: sort.by, sort_order: sort.order, page: String(page), limit: '25',
  }).toString();
  const { data, loading, error, retry } = useApi<any>(`/communities?${qs}`);
  const th = (key: string, label: string) => (
    <th onClick={() => setSort((s) => ({ by: key, order: s.by === key && s.order === 'asc' ? 'desc' : 'asc' }))}>
      {label}{sort.by === key ? (sort.order === 'asc' ? ' ↑' : ' ↓') : ''}
    </th>
  );
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <ViewHeader eyebrow="Communities" title="Unit Priority Register" sub="Deterministic AHP ranking of every surveyed village and town. Click a row to inspect."
        right={<div className="cx-pill"><Filter size={13} /> {fmt(data?.total)} matching units</div>} />
      <div className="cx-row" style={{ gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: '1 1 240px' }}>
          <Search size={15} style={{ position: 'absolute', left: 13, top: 12, opacity: 0.7 }} />
          <input className="cx-input" style={{ width: '100%', paddingLeft: 36 }} placeholder="Search village, PC11 ID or SHRID…" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <select className="cx-select" value={district} onChange={(e) => setDistrict(e.target.value)}>
          <option value="">All districts</option>{DISTRICTS.map((d) => <option key={d}>{d}</option>)}
        </select>
        <select className="cx-select" value={band} onChange={(e) => setBand(e.target.value)}>
          <option value="">All bands</option>{Object.entries(BAND).map(([k, b]) => <option key={k} value={k}>{b.label}</option>)}
        </select>
        <select className="cx-select" value={tier} onChange={(e) => setTier(e.target.value)}>
          <option value="">All tiers</option>{['High', 'Medium', 'Low'].map((t) => <option key={t}>{t}</option>)}
        </select>
      </div>
      <div className="cx-glass" style={{ borderRadius: 20, flex: 1, minHeight: 0, overflow: 'auto', position: 'relative' }}>
        {error ? <ErrorBox msg={error} onRetry={retry} /> : (
          <table className="cx-table">
            <thead><tr>{th('priority_rank', 'Rank')}{th('village_name', 'Unit')}<th>District</th>{th('preparedness_score', 'Score')}<th>Band</th><th>Tier</th>{th('dist_coast_km', 'Coast')}{th('population', 'Pop.')}</tr></thead>
            <tbody>
              {loading && !data ? Array.from({ length: 10 }).map((_, i) => <tr key={i}><td colSpan={8}><Skel h={18} /></td></tr>) :
                (data?.items || []).map((r: any) => (
                  <tr key={r.area_id} onClick={() => openCommunity(r.area_id)}>
                    <td className="cx-mono">#{fmt(r.priority_rank)}</td>
                    <td><div style={{ fontWeight: 700 }}>{r.village_name}</div><div style={{ fontSize: 11.5, color: 'rgba(255,255,255,.6)' }}>{r.subdistrict_name || r.unit_type}</div></td>
                    <td>{r.district_name}</td>
                    <td style={{ minWidth: 120 }}><div className="cx-row" style={{ gap: 8 }}><b className="cx-mono" style={{ fontSize: 13 }}>{Number(r.preparedness_score).toFixed(2)}</b><div style={{ flex: 1 }}><ScoreBar v={r.preparedness_score} /></div></div></td>
                    <td><BandChip band={r.priority_band} /></td>
                    <td><TierChip tier={r.preparedness_tier} /></td>
                    <td>{fmt(r.dist_coast_km, 1)} km</td>
                    <td>{fmt(r.population)}</td>
                  </tr>
                ))}
              {data && data.items?.length === 0 && <tr><td colSpan={8} style={{ textAlign: 'center', padding: 40, color: 'rgba(255,255,255,.6)' }}>No units match these filters.</td></tr>}
            </tbody>
          </table>
        )}
        {loading && data && <div style={{ position: 'absolute', top: 10, right: 12 }}><Loader2 size={16} className="cx-spin" /></div>}
      </div>
      <div className="cx-between" style={{ marginTop: 12 }}>
        <span style={{ fontSize: 12.5, color: 'rgba(255,255,255,.75)' }}>Page {data?.page || page} of {data?.total_pages || 1}</span>
        <div className="cx-row">
          <button className="cx-btn" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}><ChevronLeft size={15} /> Prev</button>
          <button className="cx-btn" disabled={!data || page >= data.total_pages} onClick={() => setPage((p) => p + 1)}>Next <ChevronRight size={15} /></button>
        </div>
      </div>
    </div>
  );
};

/* ── COMMUNITY DRAWER ── */
const CommunityDrawer: React.FC<{ id: string; onClose: () => void; onAsk: (id: string, name: string) => void }> = ({ id, onClose, onAsk }) => {
  const { data: c, loading, error, retry } = useApi<any>(`/communities/${encodeURIComponent(id)}`);
  useEffect(() => { const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h); }, [onClose]);
  const domains = c ? [
    ['Medical', c.score_medical, HeartPulse], ['Shelters', c.score_shelters, Home], ['Infrastructure', c.score_infrastructure, Wrench],
    ['Early warning', c.score_early_warning, Radio], ['Socio-economic', c.score_socioeconomic, Users],
  ] as [string, number, React.ElementType][] : [];
  return (
    <>
      <div className="cx-drawer-bg" onClick={onClose} />
      <aside className="cx-drawer cx-glass-dark" role="dialog" aria-label="Community dossier">
        <div className="cx-between" style={{ padding: '18px 20px 8px' }}>
          <span className="cx-label">Community dossier</span>
          <button className="cx-iconbtn" style={{ width: 34, height: 34 }} onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="cx-scroll" style={{ padding: '0 20px 20px' }}>
          {error ? <ErrorBox msg={error} onRetry={retry} /> : loading || !c ? (
            <div style={{ display: 'grid', gap: 12 }}><Skel h={28} w="60%" /><Skel h={14} w="40%" /><Skel h={120} /><Skel h={160} /></div>
          ) : (
            <div style={{ animation: 'cxIn .4s' }}>
              <h3 style={{ fontSize: 26, fontWeight: 700, margin: '4px 0 2px', letterSpacing: '-.02em' }}>{c.village_name}</h3>
              <div style={{ fontSize: 12.5, color: 'rgba(255,255,255,.7)' }}>{c.subdistrict_name ? `${c.subdistrict_name}, ` : ''}{c.district_name} · {c.unit_type}</div>
              <div className="cx-mono" style={{ fontSize: 11, color: 'rgba(255,255,255,.5)', marginTop: 4 }}>{c.shrid2} · PC11 {c.pc11_id}</div>
              <div className="cx-row" style={{ gap: 6, marginTop: 12, flexWrap: 'wrap' }}><BandChip band={c.priority_band} /><TierChip tier={c.preparedness_tier} />
                {c.vulnerability_archetype_label && <span className="cx-chip" style={{ background: 'rgba(255,255,255,.12)', border: '1px solid rgba(255,255,255,.2)' }}>{c.vulnerability_archetype_label}</span>}
                {c.tauktae_affected && <span className="cx-chip" style={{ background: '#FF6B5E33', color: '#FF6B5E' }}>Historic Cyclone Tauktae Impact (2021)</span>}
              </div>

              <div className="cx-glass" style={{ borderRadius: 18, padding: 16, marginTop: 16, display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 18, alignItems: 'center' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 44, fontWeight: 700, letterSpacing: '-.04em', lineHeight: 1 }}>{Number(c.preparedness_score).toFixed(2)}</div>
                  <div style={{ fontSize: 11, color: 'rgba(255,255,255,.7)', marginTop: 4 }}>AHP score · #{fmt(c.priority_rank)}</div>
                </div>
                <div style={{ display: 'grid', gap: 7 }}>
                  {domains.map(([l, v, I]) => (
                    <div key={l} style={{ display: 'grid', gridTemplateColumns: '18px 92px 1fr 34px', alignItems: 'center', gap: 6, fontSize: 12 }}>
                      <I size={13} opacity={0.8} /><span style={{ color: 'rgba(255,255,255,.8)' }}>{l}</span><ScoreBar v={Number(v)} /><span className="cx-mono" style={{ textAlign: 'right' }}>{Number(v).toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {c.primary_deficit_rationale && (
                <div style={{ marginTop: 16 }}>
                  <div className="cx-label">Primary deficit</div>
                  <p style={{ fontSize: 13.5, lineHeight: 1.55, margin: '6px 0 0', color: 'rgba(255,255,255,.88)' }}>{c.primary_deficit_rationale}</p>
                </div>
              )}
              {c.triggers_activated?.length > 0 && (
                <div style={{ marginTop: 14, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {c.triggers_activated.map((t: string) => <span key={t} className="cx-chip" style={{ background: '#F5A94A22', color: '#F5C27F', border: '1px solid #F5A94A55' }}><AlertTriangle size={11} />{titleCase(t.toLowerCase())}</span>)}
                </div>
              )}
              {c.actionable_advisories?.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div className="cx-label">Actionable advisories</div>
                  <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
                    {c.actionable_advisories.map((a: string, i: number) => (
                      <div key={i} className="cx-row" style={{ alignItems: 'flex-start', gap: 10, fontSize: 13, padding: '10px 12px', borderRadius: 12, background: 'rgba(255,255,255,.07)' }}>
                        <ShieldCheck size={15} style={{ flex: 'none', marginTop: 2, color: '#9BE7D8' }} />{a}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginTop: 16 }}>
                {[
                  ['Coast', `${fmt(c.dist_coast_km, 1)} km`, Waves], ['Shelter', `${fmt(c.dist_nearest_shelter_km, 1)} km`, Home], ['Hospital', `${fmt(c.dist_nearest_hospital_km, 1)} km`, HeartPulse],
                  ['Population', fmt(c.population), Users], ['Elevation', `${fmt(c.elevation_mean, 0)} m`, Navigation], ['Shelters ≤10km', fmt(c.n_shelters_10km), Building2],
                ].map(([l, v, I]: any) => (
                  <div key={l} style={{ padding: 12, borderRadius: 14, background: 'rgba(255,255,255,.07)' }}>
                    <I size={14} opacity={0.75} /><div style={{ fontSize: 15, fontWeight: 700, marginTop: 6 }}>{v}</div><div style={{ fontSize: 11, color: 'rgba(255,255,255,.6)' }}>{l}</div>
                  </div>
                ))}
              </div>

              {(c.prob_high != null || c.predicted_preparedness_tier) && (
                <div style={{ marginTop: 16 }}>
                  <div className="cx-between"><span className="cx-label">ML surrogate</span><span style={{ fontSize: 11.5, color: 'rgba(255,255,255,.7)' }}>Predicted: <b style={{ color: TIER[c.predicted_preparedness_tier] }}>{c.predicted_preparedness_tier}</b>{c.ahp_agreement ? ` · ${c.ahp_agreement}` : ''}</span></div>
                  <div className="cx-stack" style={{ marginTop: 8, height: 12 }}>
                    {(['High', 'Medium', 'Low'] as const).map((t) => <i key={t} title={`${t}: ${pct(c[`prob_${t.toLowerCase()}`])}`} style={{ flex: Number(c[`prob_${t.toLowerCase()}`] || 0), background: TIER[t] }} />)}
                  </div>
                  <div className="cx-row" style={{ gap: 12, fontSize: 11, marginTop: 6, color: 'rgba(255,255,255,.7)' }}>
                    {(['High', 'Medium', 'Low'] as const).map((t) => <span key={t}><span className="cx-dotc" style={{ background: TIER[t], marginRight: 4 }} />{t} {pct(c[`prob_${t.toLowerCase()}`])}</span>)}
                  </div>
                  <p style={{ fontSize: 11, color: 'rgba(255,255,255,.6)', marginTop: 8 }}>{c.ml_disclaimer || 'AHP composite scores maintain absolute operational precedence. Logistic regression class probabilities are uncalibrated surrogate alignments, not live cyclone forecast probabilities.'}</p>
                </div>
              )}
              <button className="cx-btn primary" style={{ width: '100%', justifyContent: 'center', marginTop: 18 }} onClick={() => onAsk(c.area_id, c.village_name)}>
                <Sparkles size={15} /> Ask assistant about {c.village_name}
              </button>
            </div>
          )}
        </div>
      </aside>
    </>
  );
};

/* ── DISTRICTS ── */
const Ring: React.FC<{ v: number; size?: number }> = ({ v, size = 110 }) => {
  const norm = Number(v) > 1 ? Number(v) / 100 : Number(v);
  const clamped = Math.max(0, Math.min(1, Number.isNaN(norm) ? 0 : norm));
  const r = size / 2 - 8, C = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
      <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,.15)" strokeWidth={8} fill="none" />
      <circle cx={size / 2} cy={size / 2} r={r} stroke={scoreColor(v)} strokeWidth={8} fill="none" strokeLinecap="round"
        strokeDasharray={C} strokeDashoffset={C * (1 - clamped)} style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(.2,.8,.2,1)', filter: `drop-shadow(0 0 6px ${scoreColor(v)})` }} />
    </svg>
  );
};

const DistrictsView: React.FC<{ focus?: string; go: (v: ViewKey, o?: any) => void }> = ({ focus, go }) => {
  const { data, loading, error, retry } = useApi<any>('/districts');
  const prep = useApi<any>('/analytics/preparedness');
  if (error) return <ErrorBox msg={error} onRetry={retry} />;
  const list: any[] = data?.districts || [];
  return (
    <div className="cx-scroll">
      <ViewHeader eyebrow="Districts" title="District Intelligence" sub="Comparative preparedness profiles for Dakshina Kannada, Udupi and Uttara Kannada." />
      <div className="cx-grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))' }}>
        {loading && !data ? [0, 1, 2].map((i) => <div key={i} className="cx-card cx-glass"><Skel h={260} /></div>) :
          list.map((d) => {
            const total = sumVals(d.band_counts);
            const dom = prep.data?.district_averages?.[d.district_name] || {};
            return (
              <div key={d.district_name} className="cx-card cx-glass" style={{ outline: focus === d.district_name ? '2px solid rgba(255,255,255,.7)' : 'none', animation: 'cxIn .5s' }}>
                <div className="cx-between" style={{ alignItems: 'flex-start' }}>
                  <div>
                    <div className="cx-label">Karnataka</div>
                    <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-.02em', marginTop: 4 }}>{d.district_name}</div>
                    <div style={{ fontSize: 12.5, color: 'rgba(255,255,255,.7)', marginTop: 2 }}>{fmt(d.total_units)} units · pop. {fmt(d.total_population)}</div>
                  </div>
                  <div style={{ position: 'relative', width: 96, height: 96 }}>
                    <Ring v={d.mean_preparedness_score} size={96} />
                    <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', fontWeight: 700, fontSize: 20 }}>{Number(d.mean_preparedness_score).toFixed(2)}</div>
                  </div>
                </div>
                <div className="cx-label" style={{ marginTop: 14 }}>Priority bands</div>
                <div className="cx-stack" style={{ marginTop: 8 }}>
                  {Object.keys(BAND).map((k) => <i key={k} title={`${BAND[k].label}: ${d.band_counts?.[k] || 0}`} style={{ flex: d.band_counts?.[k] || 0, background: BAND[k].hex }} />)}
                </div>
                <div className="cx-row" style={{ flexWrap: 'wrap', gap: 10, marginTop: 8, fontSize: 11.5, color: 'rgba(255,255,255,.75)' }}>
                  {Object.keys(BAND).map((k) => <span key={k}><span className="cx-dotc" style={{ background: BAND[k].hex, marginRight: 5 }} />{BAND[k].short} {total ? Math.round(((d.band_counts?.[k] || 0) / total) * 100) : 0}%</span>)}
                </div>
                {Object.keys(dom).length > 0 && (
                  <div style={{ display: 'grid', gap: 7, marginTop: 14 }}>
                    {Object.keys(DOMAIN_META).map((k) => (
                      <div key={k} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 34px', gap: 8, alignItems: 'center', fontSize: 12 }}>
                        <span style={{ color: 'rgba(255,255,255,.75)' }}>{DOMAIN_META[k].label}</span><ScoreBar v={Number(dom[k] || 0)} /><span className="cx-mono" style={{ textAlign: 'right' }}>{Number(dom[k] || 0).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginTop: 14 }}>
                  {[['Shelter', d.mean_dist_nearest_shelter_km, Home], ['Hospital', d.mean_dist_nearest_hospital_km, HeartPulse], ['Coast', d.mean_dist_coast_km, Waves]].map(([l, v, I]: any) => (
                    <div key={l} style={{ padding: 10, borderRadius: 12, background: 'rgba(255,255,255,.08)' }}>
                      <I size={13} opacity={0.7} /><div style={{ fontWeight: 700, marginTop: 4 }}>{fmt(v, 1)} km</div><div style={{ fontSize: 10.5, color: 'rgba(255,255,255,.6)' }}>avg. to {l.toLowerCase()}</div>
                    </div>
                  ))}
                </div>
                <div className="cx-row" style={{ gap: 8, marginTop: 14 }}>
                  <button className="cx-btn" style={{ flex: 1, justifyContent: 'center' }} onClick={() => go('communities', { district: d.district_name })}>Units</button>
                  <button className="cx-btn primary" style={{ flex: 1, justifyContent: 'center' }} onClick={() => go('communities', { district: d.district_name, band: 'BAND_1_CRITICAL' })}>{fmt(d.critical_units_count ?? d.band_counts?.BAND_1_CRITICAL)} critical <ArrowRight size={14} /></button>
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
};

/* ── ANALYTICS ── */
const AnalyticsView: React.FC = () => {
  const prep = useApi<any>('/analytics/preparedness');
  const bands = useApi<any>('/analytics/priority-bands');
  const clusters = useApi<any>('/analytics/clusters');
  const model = useApi<any>('/analytics/model-performance');

  const tierData = Object.entries(prep.data?.tier_distribution || {}).map(([k, v]) => ({ name: k, value: Number(v), fill: TIER[k] }));
  const bandData = Object.keys(BAND).map((k) => ({ name: BAND[k].short, value: Number(bands.data?.band_distribution?.[k] || 0), fill: BAND[k].hex }));
  const radarData = Object.keys(DOMAIN_META).map((k) => ({ domain: DOMAIN_META[k].label, ...Object.fromEntries(Object.entries(prep.data?.district_averages || {}).map(([d, v]: any) => [d, Number(v[k] || 0)])) }));
  const triggers = Object.entries(bands.data?.trigger_counts || {}).sort((a: any, b: any) => b[1] - a[1]).slice(0, 7).map(([k, v]) => ({ name: titleCase(k.toLowerCase()), value: Number(v) }));
  const districtColors = ['#9BE7D8', '#F5A94A', '#B794F6'];
  const cm = model.data?.test_confusion_matrix;
  const cmMax = cm ? Math.max(...cm.matrix.flat()) : 1;

  return (
    <div className="cx-scroll">
      <ViewHeader eyebrow="Analytics" title="Preparedness Analytics" sub="Distribution of AHP scores, priority bands, vulnerability archetypes and ML concordance." />
      <div className="cx-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)', marginBottom: 16 }}>
        {[
          ['Mean score', prep.data?.score_summary?.mean, Gauge, 2], ['Median score', prep.data?.score_summary?.median, Activity, 2],
          ['Test accuracy', model.data?.test_metrics?.accuracy, ShieldCheck, -1], ['ML ↔ AHP concordance', model.data?.concordance_rate_overall, Crosshair, -1],
        ].map(([l, v, I, d]: any) => (
          <div key={l} className="cx-card cx-glass cx-kpi">
            <div className="cx-between"><span>{l}</span><I size={16} opacity={0.7} /></div>
            {v == null ? <Skel h={30} w="60%" style={{ marginTop: 8 }} /> : <b>{d === -1 ? pct(v, 1) : Number(v).toFixed(d)}</b>}
          </div>
        ))}
      </div>
      <div className="cx-grid" style={{ gridTemplateColumns: '1fr 1.4fr', marginBottom: 16 }}>
        <div className="cx-card cx-glass">
          <div className="cx-label">Preparedness tiers</div>
          <div style={{ height: 230, position: 'relative' }}>
            {prep.data ? (
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={tierData} dataKey="value" nameKey="name" innerRadius={62} outerRadius={92} paddingAngle={3} stroke="none">
                    {tierData.map((t) => <Cell key={t.name} fill={t.fill} />)}
                  </Pie>
                  <Tooltip {...chartTip} />
                </PieChart>
              </ResponsiveContainer>
            ) : <Skel h={220} />}
            {prep.data && <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none', textAlign: 'center' }}><div><div style={{ fontSize: 26, fontWeight: 700 }}>{fmt(prep.data.total_units)}</div><div style={{ fontSize: 11, opacity: 0.7 }}>units</div></div></div>}
          </div>
          <div className="cx-row" style={{ justifyContent: 'center', gap: 14, fontSize: 12 }}>
            {tierData.map((t) => <span key={t.name}><span className="cx-dotc" style={{ background: t.fill, marginRight: 5 }} />{t.name} {fmt(t.value)}</span>)}
          </div>
        </div>
        <div className="cx-card cx-glass">
          <div className="cx-label">Domain profile by district</div>
          <div style={{ height: 260 }}>
            {prep.data ? (
              <ResponsiveContainer>
                <RadarChart data={radarData} outerRadius="72%">
                  <PolarGrid stroke="rgba(255,255,255,.18)" />
                  <PolarAngleAxis dataKey="domain" tick={{ fill: 'rgba(255,255,255,.8)', fontSize: 12 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                  {Object.keys(prep.data.district_averages || {}).map((d, i) => (
                    <Radar key={d} name={d} dataKey={d} stroke={districtColors[i % 3]} fill={districtColors[i % 3]} fillOpacity={0.15} strokeWidth={2} />
                  ))}
                  <Tooltip {...chartTip} />
                </RadarChart>
              </ResponsiveContainer>
            ) : <Skel h={250} />}
          </div>
          <div className="cx-row" style={{ justifyContent: 'center', gap: 14, fontSize: 12 }}>
            {Object.keys(prep.data?.district_averages || {}).map((d, i) => <span key={d}><span className="cx-dotc" style={{ background: districtColors[i % 3], marginRight: 5 }} />{d}</span>)}
          </div>
        </div>
      </div>
      <div className="cx-grid" style={{ gridTemplateColumns: '1fr 1.4fr', marginBottom: 16 }}>
        <div className="cx-card cx-glass">
          <div className="cx-label">Priority bands</div>
          <div style={{ height: 240, marginTop: 10 }}>
            {bands.data ? (
              <ResponsiveContainer>
                <BarChart data={bandData} margin={{ left: -18 }}>
                  <CartesianGrid vertical={false} stroke="rgba(255,255,255,.1)" />
                  <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,.75)', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: 'rgba(255,255,255,.55)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip {...chartTip} />
                  <Bar dataKey="value" radius={[10, 10, 4, 4]}>{bandData.map((b) => <Cell key={b.name} fill={b.fill} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <Skel h={230} />}
          </div>
        </div>
        <div className="cx-card cx-glass">
          <div className="cx-label">Most frequent deficit triggers</div>
          <div style={{ display: 'grid', gap: 10, marginTop: 14 }}>
            {bands.data ? triggers.map((t) => (
              <div key={t.name} style={{ display: 'grid', gridTemplateColumns: '210px 1fr 50px', gap: 10, alignItems: 'center', fontSize: 12.5 }}>
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.name}</span>
                <div className="cx-bar" style={{ height: 8 }}><i style={{ width: `${(t.value / (triggers[0]?.value || 1)) * 100}%`, background: 'linear-gradient(90deg,#F5A94A,#FF6B5E)' }} /></div>
                <span className="cx-mono" style={{ textAlign: 'right' }}>{fmt(t.value)}</span>
              </div>
            )) : <Skel h={200} />}
          </div>
        </div>
      </div>
      <div className="cx-label" style={{ margin: '6px 0 10px' }}>Vulnerability archetypes (K-Means, k=4)</div>
      <div className="cx-grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', marginBottom: 16 }}>
        {(clusters.data?.clusters || [0, 1, 2, 3].map(() => null)).map((c: any, i: number) => (
          <div key={c?.cluster_id ?? i} className="cx-card cx-glass">
            {c ? (
              <>
                <div className="cx-between"><span className="cx-dotc" style={{ width: 12, height: 12, background: ARCHETYPE[c.cluster_id]?.hex, boxShadow: `0 0 12px ${ARCHETYPE[c.cluster_id]?.hex}` }} /><b style={{ fontSize: 24 }}>{fmt(c.count)}</b></div>
                <div style={{ fontWeight: 700, marginTop: 10, fontSize: 15 }}>{c.label}</div>
                <div style={{ fontSize: 12, color: 'rgba(255,255,255,.7)', marginTop: 6, lineHeight: 1.5 }}>{c.detailed_description}</div>
                <div className="cx-row" style={{ gap: 6, marginTop: 10, fontSize: 12, color: '#9BE7D8', alignItems: 'flex-start' }}><Wrench size={13} style={{ flex: 'none', marginTop: 2 }} />{c.primary_intervention_need}</div>
              </>
            ) : <Skel h={130} />}
          </div>
        ))}
        {clusters.data?.unassigned_count > 0 && (
          <div className="cx-card cx-glass">
            <div className="cx-between">
              <span className="cx-dotc" style={{ width: 12, height: 12, background: '#7f9488', boxShadow: '0 0 12px #7f9488' }} />
              <b style={{ fontSize: 24 }}>{fmt(clusters.data.unassigned_count)}</b>
            </div>
            <div style={{ fontWeight: 700, marginTop: 10, fontSize: 15 }}>Unassigned (Uttara Kannada Fallback)</div>
            <div style={{ fontSize: 12, color: 'rgba(255,255,255,.7)', marginTop: 6, lineHeight: 1.5 }}>
              Administrative units operating under deterministic AHP scoring without cluster assignment.
            </div>
            <div className="cx-row" style={{ gap: 6, marginTop: 10, fontSize: 12, color: '#9BE7D8', alignItems: 'flex-start' }}>
              <ShieldCheck size={13} style={{ flex: 'none', marginTop: 2 }} />
              Deterministic AHP baseline applies for all emergency resource allocation.
            </div>
          </div>
        )}
      </div>
      <div className="cx-grid" style={{ gridTemplateColumns: '1.2fr 1fr', marginBottom: 8 }}>
        <div className="cx-card cx-glass">
          <div className="cx-label">Surrogate model · {model.data?.model_type || '—'}</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginTop: 14 }}>
            {(['train', 'val', 'test'] as const).map((s) => (
              <div key={s} style={{ padding: 14, borderRadius: 14, background: 'rgba(255,255,255,.08)' }}>
                <div className="cx-label">{s}</div>
                {model.data ? (
                  <div style={{ display: 'grid', gap: 6, marginTop: 8, fontSize: 12.5 }}>
                    <div className="cx-between"><span>Accuracy</span><b>{pct(model.data[`${s}_metrics`]?.accuracy, 1)}</b></div>
                    <div className="cx-between"><span>Macro F1</span><b>{pct(model.data[`${s}_metrics`]?.macro_f1, 1)}</b></div>
                    <div className="cx-between"><span>Weighted F1</span><b>{pct(model.data[`${s}_metrics`]?.weighted_f1, 1)}</b></div>
                  </div>
                ) : <Skel h={60} style={{ marginTop: 8 }} />}
              </div>
            ))}
          </div>
          <p style={{ fontSize: 11.5, color: 'rgba(255,255,255,.55)', marginTop: 12 }}>Deterministic AHP composite score maintains operational precedence. Logistic regression serves strictly as an empirical tier surrogate (uncalibrated class alignments, not live cyclone forecast probabilities).</p>
        </div>
        <div className="cx-card cx-glass">
          <div className="cx-label">Test confusion matrix</div>
          {cm ? (
            <div style={{ display: 'grid', gridTemplateColumns: `70px repeat(${cm.classes.length},1fr)`, gap: 6, marginTop: 14, fontSize: 12 }}>
              <span />{cm.classes.map((c: string) => <span key={c} style={{ textAlign: 'center', color: 'rgba(255,255,255,.7)' }}>{c}</span>)}
              {cm.matrix.map((row: number[], i: number) => (
                <React.Fragment key={i}>
                  <span style={{ alignSelf: 'center', color: 'rgba(255,255,255,.7)' }}>{cm.classes[i]}</span>
                  {row.map((v, j) => (
                    <div key={j} style={{ height: 46, borderRadius: 10, display: 'grid', placeItems: 'center', fontWeight: 700, background: i === j ? `rgba(155,231,216,${0.15 + (v / cmMax) * 0.6})` : `rgba(255,107,94,${0.06 + (v / cmMax) * 0.6})` }}>{v}</div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          ) : <Skel h={170} style={{ marginTop: 14 }} />}
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,.55)', marginTop: 10 }}>Rows: actual · Columns: predicted</div>
        </div>
      </div>
      {(prep.data?.disclaimer) && <p style={{ fontSize: 11.5, color: 'rgba(255,255,255,.55)', margin: '10px 0 4px' }}>{prep.data.disclaimer}</p>}
    </div>
  );
};

/* ── ASSISTANT ── */
type Msg = { role: 'user' | 'assistant'; content: string; tools?: any[]; meta?: string; note?: string; error?: boolean };
const SUGGESTIONS = [
  'Which 5 villages in Udupi need cyclone shelters most urgently?',
  'Compare preparedness across the three coastal districts.',
  'What interventions does the High-Exposure Coastal Belt archetype need?',
  'List Band 1 critical units within 5 km of the coast in Dakshina Kannada.',
];

export const DEMO_SAMPLE_PROMPTS = [
  'Explain why Tallur is classified as Band 4.',
  'Compare Dakshina Kannada and Udupi.',
  'Explain the four vulnerability archetypes.',
  'Identify Band 1 communities using the actual dataset.',
  'Explain how AHP preparedness tiers differ from operational priority bands.',
  'Explain preparedness measures based on dataset-backed vulnerabilities.',
];

const AssistantView: React.FC<{
  context?: { id: string; name: string } | null;
  clearContext: () => void;
  demoMode?: boolean;
  clearKey?: number;
}> = ({ context, clearContext, demoMode, clearKey }) => {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (clearKey) setMsgs([]);
  }, [clearKey]);

  useEffect(() => { const el = listRef.current; if (el && typeof el.scrollTo === 'function') el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }); }, [msgs, busy]);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    const history = msgs.filter((m) => !m.error).map((m) => ({ role: m.role, content: m.content }));
    setMsgs((m) => [...m, { role: 'user', content: q }]);
    setInput('');
    setBusy(true);
    try {
      const r = await sendChat(q, history, context?.id);
      const tools = r.tool_executions || r.tool_records || (r.tools_executed || []).map((t: string) => ({ tool_name: t }));
      const ms = r.total_duration_ms ?? r.duration_ms;
      const metaParts = [r.model_name, ms != null ? `${Math.round(ms)} ms` : null, r.is_fallback ? 'Fallback mode' : null].filter(Boolean);
      setMsgs((m) => [...m, { role: 'assistant', content: r.answer || '_No answer returned._', tools, meta: metaParts.join(' · '), note: r.scientific_precedence_note }]);
    } catch (e: any) {
      setMsgs((m) => [...m, { role: 'assistant', content: `**Request failed:** ${e.message || 'Unknown error'}`, error: true }]);
    } finally { setBusy(false); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', maxWidth: 960, margin: '0 auto' }}>
      <ViewHeader eyebrow="AI Assistant" title="Decision Support Assistant" sub="Closed-world, tool-grounded answers over the preparedness dataset — no free-form speculation."
        right={msgs.length > 0 ? <button className="cx-btn" onClick={() => setMsgs([])}>New chat</button> : undefined} />
      <div ref={listRef} className="cx-scroll" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14, paddingBottom: 10 }}>
        {msgs.length === 0 && (
          <div style={{ margin: 'auto 0', animation: 'cxIn .5s' }}>
            <div style={{ textAlign: 'center', marginBottom: 18 }}>
              <div className="cx-logo" style={{ margin: '0 auto 12px', width: 56, height: 56 }}><Bot size={26} /></div>
              <div style={{ fontSize: 20, fontWeight: 700 }}>How can I help the control room today?</div>
              <div style={{ fontSize: 12.5, color: 'rgba(255,255,255,.7)', marginTop: 4 }}>Ask about rankings, districts, archetypes or a specific village.</div>
            </div>

            <div style={{ marginBottom: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, fontWeight: 700, color: '#9BE7D8', marginBottom: 8 }}>
                <Sparkles size={14} /> Sample Presentation Interactions (Live API Grounding)
              </div>
              <div className="cx-prompt-grid">
                {DEMO_SAMPLE_PROMPTS.map((p) => (
                  <button key={p} className="cx-prompt-card cx-glass" onClick={() => send(p)}>
                    <span style={{ fontWeight: 600 }}>{p}</span>
                    <span style={{ fontSize: 11, opacity: 0.65, marginTop: 8, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <ArrowRight size={12} /> Run deterministic query
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,.6)', marginBottom: 8 }}>General Inquiries:</div>
              <div className="cx-grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))' }}>
                {SUGGESTIONS.map((s) => <button key={s} className="cx-suggest cx-glass" onClick={() => send(s)}><Sparkles size={14} style={{ marginBottom: 6, opacity: 0.8 }} /><div>{s}</div></button>)}
              </div>
            </div>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={cx('cx-msg', m.role === 'user' ? 'user' : 'bot cx-glass')} style={m.error ? { borderColor: '#FF6B5E88' } : undefined}>
            {m.role === 'assistant' ? <ReactMarkdown>{m.content}</ReactMarkdown> : m.content}
            {m.tools && m.tools.length > 0 && (
              <div className="cx-row" style={{ flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
                {m.tools.map((t: any, j: number) => (
                  <span key={j} className="cx-chip" title={t.arguments ? JSON.stringify(t.arguments) : ''} style={{ background: 'rgba(155,231,216,.15)', color: '#9BE7D8', border: '1px solid rgba(155,231,216,.35)', fontWeight: 600 }}>
                    <Wrench size={11} />{t.tool_name}{t.execution_time_ms != null ? ` · ${Math.round(t.execution_time_ms)}ms` : ''}
                  </span>
                ))}
              </div>
            )}
            {m.meta && <div style={{ fontSize: 10.5, opacity: 0.55, marginTop: 8 }}>{m.meta}</div>}
            {m.meta && m.meta.includes('Fallback mode') && (
              <div className="cx-pill warn" style={{ background: 'rgba(245,169,74,.15)', borderColor: '#F5A94A', color: '#F5A94A', marginTop: 8, fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <AlertTriangle size={12} /> Qwen/Ollama local engine unavailable · Serving deterministic tool-grounded briefing
              </div>
            )}
            {m.note && <div style={{ fontSize: 11, color: 'rgba(255,255,255,.55)', marginTop: 8, fontStyle: 'italic', borderLeft: '2px solid rgba(255,255,255,.2)', paddingLeft: 8 }}>{m.note}</div>}
          </div>
        ))}
        {busy && <div className="cx-msg bot cx-glass cx-typing" style={{ width: 80 }}><span /><span /><span /></div>}
      </div>

      {demoMode && msgs.length > 0 && (
        <div className="cx-row" style={{ flexWrap: 'nowrap', overflowX: 'auto', gap: 6, marginBottom: 8, paddingBottom: 4 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: '#9BE7D8', whiteSpace: 'nowrap' }}>Demo prompts:</span>
          {DEMO_SAMPLE_PROMPTS.map((p) => (
            <button
              key={p}
              className="cx-chip"
              style={{ background: 'rgba(155,231,216,.12)', borderColor: 'rgba(155,231,216,.3)', color: '#fff', whiteSpace: 'nowrap', cursor: 'pointer', fontSize: 11 }}
              onClick={() => send(p)}
            >
              {p.slice(0, 36)}…
            </button>
          ))}
        </div>
      )}

      {context && (
        <div className="cx-row" style={{ marginBottom: 8 }}>
          <span className="cx-pill"><MapPin size={12} /> Context: {context.name}<X size={13} style={{ cursor: 'pointer', marginLeft: 4 }} onClick={clearContext} /></span>
        </div>
      )}
      <form className="cx-glass" style={{ borderRadius: 18, padding: 8, display: 'flex', gap: 8 }} onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input className="cx-input" style={{ flex: 1, border: 0, background: 'transparent', height: 44 }} placeholder={context ? `Ask about ${context.name}…` : 'Ask the preparedness assistant…'} value={input} onChange={(e) => setInput(e.target.value)} disabled={busy} />
        <button className="cx-btn primary" type="submit" disabled={busy || !input.trim()} style={{ borderRadius: 13 }}>{busy ? <Loader2 size={16} className="cx-spin" /> : <Send size={16} />}</button>
      </form>
    </div>
  );
};

/* ───────────────────────────── SHELL ───────────────────────────── */

type ViewKey = 'overview' | 'map' | 'communities' | 'districts' | 'analytics' | 'assistant';
const NAV: { key: ViewKey; icon: React.ElementType; label: string }[] = [
  { key: 'overview', icon: LayoutGrid, label: 'Overview' },
  { key: 'map', icon: MapIcon, label: 'Preparedness Map' },
  { key: 'communities', icon: Building2, label: 'Communities' },
  { key: 'districts', icon: Landmark, label: 'Districts' },
  { key: 'analytics', icon: BarChart3, label: 'Analytics' },
  { key: 'assistant', icon: Bot, label: 'AI Assistant' },
];

export interface DemoStep {
  step: number;
  view: ViewKey;
  title: string;
  subtitle: string;
  summary: string;
  drawerId?: string;
  opts?: any;
}

export const DEMO_STEPS: DemoStep[] = [
  {
    step: 1,
    view: 'overview',
    title: 'Command Center',
    subtitle: 'Macroscopic Regional Preparedness Posture',
    summary: 'Executive overview across 1,879 administrative units in Dakshina Kannada, Udupi, and Uttara Kannada. Grounded in deterministic AHP composite indices with 4 operational priority bands.',
  },
  {
    step: 2,
    view: 'map',
    title: 'Community Preparedness Map',
    subtitle: 'Interactive Geospatial Preparedness Intelligence',
    summary: 'Spatial visualization of coastal revenue units categorized by operational priority band and proximity to coastline, highlighting high-exposure clusters.',
  },
  {
    step: 3,
    view: 'communities',
    title: 'Community Explorer',
    subtitle: 'Granular Administrative Unit Register & Search',
    summary: 'Multi-criteria filtering and full-text search across all 1,879 villages and Census Towns with real-time indicators and priority rankings.',
  },
  {
    step: 4,
    view: 'communities',
    title: 'Community Dossier (Tallur)',
    subtitle: 'Deep-Dive Administrative Dossier Inspection',
    summary: 'Detailed inspection of Tallur (PC11: 608691, Udupi): Authoritative AHP score 74.34, High preparedness tier, Priority Band 4 (Adequate), with 5 domain indicator breakdowns and historical Tauktae impact.',
    drawerId: '11-29-569-05522-608691',
  },
  {
    step: 5,
    view: 'districts',
    title: 'District Intelligence',
    subtitle: 'Cross-District Comparative Analytics',
    summary: 'Comparative analysis across Dakshina Kannada, Udupi, and Uttara Kannada demonstrating stark resource differentials and vulnerability concentrations.',
  },
  {
    step: 6,
    view: 'analytics',
    title: 'Analytics & Models',
    subtitle: 'Scientific Safeguards & Model Quality Gates',
    summary: 'K-Means (k=4) vulnerability archetypes (622 coastal units) with 1,257 unassigned Uttara Kannada units, plus surrogate Logistic Regression confusion matrix and concordance metrics.',
  },
  {
    step: 7,
    view: 'assistant',
    title: 'AI Decision Support Assistant',
    subtitle: 'Deterministic Tool-Grounded Conversational Intelligence',
    summary: 'Closed-world conversational assistant backed by Qwen and 7 deterministic AHP tools. Refusal gates prevent hallucinations, and deterministic fallback guarantees uptime.',
  },
];

export const App: React.FC = () => {
  const [view, setView] = useState<ViewKey>(() => {
    const h = (typeof window !== 'undefined' ? window.location.hash.replace('#/', '') : '') as ViewKey;
    return NAV.some((n) => n.key === h) ? h : 'overview';
  });
  const [opts, setOpts] = useState<any>({});
  const [drawer, setDrawer] = useState<string | null>(null);
  const [chatCtx, setChatCtx] = useState<{ id: string; name: string } | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [q, setQ] = useState('');
  const [demoMode, setDemoMode] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const sp = new URLSearchParams(window.location.search);
      return sp.get('demo') === 'true' || window.location.hash.includes('demo');
    }
    return false;
  });
  const [demoStep, setDemoStep] = useState<number>(0);
  const [chatResetKey, setChatResetKey] = useState<number>(0);
  const isDemo = useDemoMode();
  const health = useApi<any>('/health');

  const go = useCallback((v: ViewKey, o: any = {}) => { setOpts(o); setView(v); }, []);

  const goToDemoStep = useCallback((stepIndex: number) => {
    const s = DEMO_STEPS[stepIndex];
    if (!s) return;
    setDemoStep(stepIndex);
    setOpts(s.opts || {});
    setView(s.view);
    if (s.drawerId) {
      setDrawer(s.drawerId);
    } else {
      setDrawer(null);
    }
  }, []);

  const toggleDemoMode = useCallback(() => {
    if (demoMode) {
      setDemoMode(false);
      setDrawer(null);
      go('overview');
    } else {
      setDemoMode(true);
      goToDemoStep(0);
    }
  }, [demoMode, go, goToDemoStep]);

  const resetDemoSession = useCallback(() => {
    setChatResetKey((k) => k + 1);
    setChatCtx(null);
    setOpts({});
    goToDemoStep(0);
  }, [goToDemoStep]);

  const toggleFullscreen = () => {
    try {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen?.().catch(() => {});
      } else {
        document.exitFullscreen?.().catch(() => {});
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => { try { if (window.location.hash !== `#/${view}` && !demoMode) window.history.replaceState(null, '', `#/${view}`); } catch { /* ignore */ } }, [view, demoMode]);
  useEffect(() => {
    const h = () => { const k = window.location.hash.replace('#/', '') as ViewKey; if (NAV.some((n) => n.key === k)) { setOpts({}); setView(k); } };
    window.addEventListener('hashchange', h); return () => window.removeEventListener('hashchange', h);
  }, []);

  const ov = useApi<any>('/dashboard/overview');
  const critical = ov.data?.priority_band_counts?.BAND_1_CRITICAL;

  const searchRes = useApi<any>(q.trim().length >= 2 ? `/communities?search=${encodeURIComponent(q.trim())}&limit=6` : null);

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setSearchOpen(true); } if (e.key === 'Escape') setSearchOpen(false); };
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h);
  }, []);

  const content = (() => {
    switch (view) {
      case 'overview': return <OverviewView go={go} />;
      case 'map': return <MapView openCommunity={setDrawer} />;
      case 'communities': return <CommunitiesView key={JSON.stringify(opts)} init={opts} openCommunity={setDrawer} />;
      case 'districts': return <DistrictsView focus={opts.district} go={go} />;
      case 'analytics': return <AnalyticsView />;
      case 'assistant': return <AssistantView context={chatCtx} clearContext={() => setChatCtx(null)} demoMode={demoMode} clearKey={chatResetKey} />;
    }
  })();

  return (
    <div className="cx-root">
      <style>{CSS}</style>
      <div className="cx-outer-bg"><StormSky id="sky-o" /></div>
      <div className="cx-frame">
        <StormSky id="sky-i" />
        <div className={cx('cx-dim', view !== 'overview' && 'on')} />
        <div className="cx-shell">
          <nav className="cx-side cx-glass" aria-label="Primary">
            <div className="cx-logo" title="Cyclone Intel"><Wind size={22} /></div>
            {NAV.map((n) => (
              <button key={n.key} className={cx('cx-nav', view === n.key && 'on')} onClick={() => go(n.key)} aria-label={n.label}>
                <n.icon size={19} /><span className="tip">{n.label}</span>
              </button>
            ))}
            <div className="cx-spacer" style={{ flex: 1 }} />
            <button className="cx-nav" aria-label="Settings" onClick={() => alert('Settings — wire to your settings route')}><Settings size={19} /><span className="tip">Settings</span></button>
            <button className="cx-nav" aria-label="Sign out"><LogOut size={19} /><span className="tip">Sign out</span></button>
          </nav>

          <main className="cx-main">
            <header className="cx-top">
              <div className="cx-hello"><small>Welcome</small><b>{OFFICER_NAME}</b></div>
              <div className="cx-row" style={{ gap: 10 }}>
                <button
                  className={cx('cx-btn', demoMode && 'primary')}
                  onClick={toggleDemoMode}
                  style={{ height: 36, padding: '0 12px', gap: 6, fontSize: 12.5 }}
                  aria-label="Toggle Presentation Demo Mode"
                >
                  <Sparkles size={15} />
                  <span>{demoMode ? 'Exit Demo' : 'Demo Mode'}</span>
                </button>
                {isDemo && <span className="cx-pill warn" title="Backend unreachable — showing generated demo data"><Droplets size={13} /> Demo data</span>}
                {!isDemo && health.data && <span className="cx-pill"><span className="cx-dotc" style={{ background: '#5FD39A', boxShadow: '0 0 8px #5FD39A' }} /> Live</span>}
                <button className="cx-iconbtn" aria-label="Ask assistant" onClick={() => go('assistant')}><Plus size={18} /></button>
                <button className="cx-iconbtn" aria-label="Search (Ctrl+K)" onClick={() => setSearchOpen(true)}><Search size={17} /></button>
                <button className="cx-iconbtn" aria-label="Alerts" onClick={() => go('communities', { band: 'BAND_1_CRITICAL' })}>
                  <Bell size={17} />{critical > 0 && <span className="cx-dot" />}
                </button>
                <div className="cx-avatar">DC</div>
              </div>
            </header>

            {demoMode && (
              <div className="cx-demo-dock cx-glass" role="region" aria-label="Presentation Demo Controls">
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: '1 1 320px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="cx-pill" style={{ background: 'rgba(155,231,216,.2)', borderColor: '#9BE7D8', color: '#9BE7D8', fontWeight: 700 }}>
                      <Sparkles size={13} /> Demo Walkthrough
                    </span>
                    <span className="cx-pill" style={{ fontSize: 11, opacity: 0.85 }}>
                      Step {demoStep + 1} of {DEMO_STEPS.length}
                    </span>
                  </div>
                  <div>
                    <b style={{ fontSize: 14, color: '#fff' }}>{DEMO_STEPS[demoStep].title}</b>
                    <div style={{ fontSize: 11.5, color: 'rgba(255,255,255,.7)', marginTop: 2 }}>{DEMO_STEPS[demoStep].summary}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {DEMO_STEPS.map((s, idx) => (
                      <button
                        key={s.step}
                        className={cx('cx-demo-step-pill', demoStep === idx && 'on')}
                        onClick={() => goToDemoStep(idx)}
                        title={`${s.step}. ${s.title}`}
                        aria-label={`Jump to step ${s.step}: ${s.title}`}
                      >
                        {s.step}
                      </button>
                    ))}
                  </div>

                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="cx-btn"
                      disabled={demoStep === 0}
                      onClick={() => goToDemoStep(demoStep - 1)}
                      style={{ height: 32, padding: '0 10px', fontSize: 12 }}
                    >
                      <ChevronLeft size={14} /> Prev
                    </button>
                    <button
                      className="cx-btn primary"
                      disabled={demoStep === DEMO_STEPS.length - 1}
                      onClick={() => goToDemoStep(demoStep + 1)}
                      style={{ height: 32, padding: '0 12px', fontSize: 12 }}
                    >
                      Next <ChevronRight size={14} />
                    </button>
                  </div>

                  <div style={{ display: 'flex', gap: 6, borderLeft: '1px solid var(--line-soft)', paddingLeft: 8 }}>
                    <button
                      className="cx-btn"
                      onClick={resetDemoSession}
                      title="Reset demo session to Step 1 and clear state"
                      style={{ height: 32, padding: '0 10px', fontSize: 12 }}
                    >
                      Reset Session
                    </button>
                    <button
                      className="cx-btn"
                      onClick={toggleFullscreen}
                      title="Toggle Fullscreen"
                      style={{ height: 32, width: 32, padding: 0 }}
                    >
                      <Crosshair size={14} />
                    </button>
                    <button
                      className="cx-btn"
                      onClick={toggleDemoMode}
                      aria-label="Exit presentation mode"
                      title="Exit presentation mode"
                      style={{ height: 32, padding: '0 10px', fontSize: 12, borderColor: '#FF6B5E88', color: '#FF9488' }}
                    >
                      Exit Demo
                    </button>
                  </div>
                </div>

                <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,.08)', paddingTop: 6, marginTop: 4, fontSize: 11, color: 'rgba(255,255,255,.6)' }}>
                  <span>📌 <b>Data Source:</b> Census 2011 &amp; Spatial Baseline · Deterministic AHP Baseline (Non-live Meteorological Simulation)</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {health.data ? (
                      <span style={{ color: '#9BE7D8' }}>● Live API (1,879 units)</span>
                    ) : (
                      <span style={{ color: '#F5A94A' }}>⚠️ API Offline</span>
                    )}
                  </span>
                </div>
                {(!health.data && !health.loading) && (
                  <div style={{ width: '100%', background: 'rgba(255,107,94,0.15)', border: '1px solid #FF6B5E', borderRadius: 8, padding: '6px 12px', marginTop: 6, fontSize: 12, color: '#FF9488', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <AlertTriangle size={14} />
                    <span><b>Backend Health Warning:</b> API backend is unreachable. Presentation Demo Mode requires a live API at <code>{API_BASE}</code>.</span>
                  </div>
                )}
              </div>
            )}

            <div className="cx-content">
              <div className="cx-view" key={view + JSON.stringify(opts)}>{content}</div>
            </div>
          </main>
        </div>

        {drawer && <CommunityDrawer id={drawer} onClose={() => setDrawer(null)} onAsk={(id, name) => { setChatCtx({ id, name }); setDrawer(null); go('assistant'); }} />}

        {searchOpen && (
          <>
            <div className="cx-drawer-bg" onClick={() => setSearchOpen(false)} />
            <div className="cx-glass-dark" style={{ position: 'absolute', top: '14%', left: '50%', transform: 'translateX(-50%)', width: 'min(560px,92%)', borderRadius: 22, zIndex: 32, padding: 10, animation: 'cxIn .3s' }}>
              <div style={{ position: 'relative' }}>
                <Search size={17} style={{ position: 'absolute', left: 14, top: 14, opacity: 0.7 }} />
                <input autoFocus className="cx-input" style={{ width: '100%', height: 46, paddingLeft: 42, fontSize: 15 }} placeholder="Search any village, PC11 ID or SHRID…" value={q} onChange={(e) => setQ(e.target.value)} />
              </div>
              <div style={{ marginTop: 6, maxHeight: 340, overflow: 'auto' }}>
                {q.trim().length < 2 ? <div style={{ padding: 14, fontSize: 12.5, color: 'rgba(255,255,255,.6)' }}>Type at least 2 characters · <span className="cx-mono">Ctrl/⌘ K</span> to open anywhere</div> :
                  searchRes.loading ? <div style={{ padding: 14 }}><Loader2 size={16} className="cx-spin" /></div> :
                    (searchRes.data?.items || []).length === 0 ? <div style={{ padding: 14, fontSize: 13, color: 'rgba(255,255,255,.6)' }}>No matches.</div> :
                      searchRes.data.items.map((r: any) => (
                        <button key={r.area_id} onClick={() => { setDrawer(r.area_id); setSearchOpen(false); setQ(''); }} className="cx-between"
                          style={{ width: '100%', padding: '11px 12px', borderRadius: 12, border: 0, background: 'transparent', color: '#fff', cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left' }}
                          onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,.1)')} onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}>
                          <span><b style={{ fontSize: 14 }}>{r.village_name}</b><span style={{ fontSize: 12, opacity: 0.65, marginLeft: 8 }}>{r.district_name}</span></span>
                          <BandChip band={r.priority_band} />
                        </button>
                      ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default App;
