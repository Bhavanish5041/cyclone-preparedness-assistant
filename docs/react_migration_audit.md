# React Migration Architecture Audit & Feature-Parity Specification
**Project:** Cyclone Preparedness Intelligence Platform  
**Location:** `/home/bhavanish/cyclone-preparedness-assistant`  
**Phase:** Phase 0 (Pre-Migration Architecture Audit)  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

This document establishes the authoritative pre-migration baseline and architectural audit for migrating the **Cyclone Preparedness Intelligence Platform** from its existing Streamlit-based presentation layer to a modern, production-grade **React + TypeScript + Vite + Tailwind CSS** frontend backed by a modular **FastAPI** REST API service layer.

### Core Migration Principles & Non-Negotiables
1. **Preservation of Scientific Integrity**: Deterministic Analytical Hierarchy Process (AHP) composite scores and tiers (`High`, `Medium`, `Low`) remain absolute and authoritative for structural capacity. Operational Priority Bands (`Band 1: Critical`, `Band 2: Specialized`, `Band 3: Moderate`, `Band 4: Adequate`) remain authoritative for emergency resource allocation. The frontend shall never compute these calculations independently.
2. **Surrogate Model Guardrails**: The Multinomial Logistic Regression model is strictly an empirical surrogate approximating AHP-derived tiers, trained on 26 engineered indicators across 1,879 units. It does NOT forecast cyclone landfall probabilities, storm surges, structural failure, or human casualties. Never allow an ML prediction to downgrade an authoritative AHP `Low` classification.
3. **Vulnerability Archetype Boundaries**: K-Means clustering partitions 622 administrative units across Dakshina Kannada (369) and Udupi (253) into four descriptive vulnerability archetypes. Uttara Kannada (1,257 units) is unassigned to K-Means clusters and uses an independent deterministic multi-trigger sector-deficit fallback engine.
4. **Zero Destruction / Safe Coexistence**: The existing Streamlit platform in `src/assistant/app.py` and its supporting views in `src/assistant/views/` will remain completely intact, functional, and unaffected throughout and following the migration.
5. **No Blind Code or Artifact Deletion**: All existing model artifacts (`models/logistic_regression_tier.joblib`, `models/logistic_regression_metadata.json`, `models/label_encoder.joblib`) and processed datasets (`data/processed/unified_village_preparedness_v2.parquet`, `data/processed/coastal_vulnerability_archetypes.parquet`) are preserved without alteration.

---

## 2. Existing Repository Architecture

### 2.1 File System Topology
```text
cyclone-preparedness-assistant/
├── .env / .env.example
├── Dockerfile.api                  # FastAPI production container
├── Dockerfile.dashboard            # Streamlit dashboard container
├── Dockerfile.render               # Render free-tier unified container
├── requirements.txt                # Pinned Python dependencies
├── data/
│   └── processed/
│       ├── coastal_vulnerability_archetypes.parquet # 622 units (GeoDataFrame)
│       └── unified_village_preparedness_v2.parquet  # 1,879 units (DataFrame)
├── models/
│   ├── label_encoder.joblib
│   ├── logistic_regression_metadata.json            # 26 features, test metrics, confusion matrix
│   ├── logistic_regression_tier.joblib              # Pipeline (RobustScaler + LogisticRegression)
│   └── render_artifact/
├── src/
│   ├── api/                        # Existing FastAPI application & schemas
│   │   ├── main.py                 # FastAPI endpoints (health, units, scenarios, model, chat)
│   │   └── schemas.py              # Pydantic v2 validation models
│   ├── assistant/                  # Streamlit application & decision services
│   │   ├── app.py                  # Main Streamlit multi-page entry point
│   │   ├── api_client.py           # Dual-mode client (Direct in-process & HTTP mode)
│   │   ├── decision_support.py     # DecisionSupportService analytical engine
│   │   ├── charts.py               # Altair & Matplotlib chart generators
│   │   ├── maps.py                 # Folium & Leaflet map builders
│   │   ├── theme.py                # CSS styles and HTML UI helpers
│   │   ├── llm/                    # Conversational AI orchestration
│   │   │   ├── client.py           # OpenRouter & Gemini HTTP clients
│   │   │   ├── history.py          # Message history compaction & management
│   │   │   ├── orchestrator.py     # Multi-turn tool execution orchestrator
│   │   │   ├── prompts.py          # System instructions & response templates
│   │   │   └── schemas.py          # Chat request/response schemas
│   │   ├── tools/                  # 7 Deterministic tool implementations
│   │   │   ├── base.py             # BaseTool and ToolResult abstractions
│   │   │   ├── registry.py         # ToolRegistry with Gemini schema sanitizer
│   │   │   ├── community.py        # lookup_community, analyze_infrastructure_and_exposure
│   │   │   ├── decision.py         # explain_preparedness_assessment, get_intervention_recommendations
│   │   │   ├── regional.py         # compare_districts, filter_and_rank_communities
│   │   │   └── archetypes.py       # explain_vulnerability_archetype
│   │   └── views/                  # Streamlit page view renderers
│   │       ├── overview.py
│   │       ├── gis_explorer.py
│   │       ├── community_profile.py
│   │       ├── scenario_analysis.py
│   │       ├── archetypes.py
│   │       ├── reports.py
│   │       ├── chat_assistant.py
│   │       └── ai_readiness.py
│   ├── features/                   # Feature engineering & dictionary
│   ├── geo/                        # Geographic boundaries & utilities
│   ├── integration/                # Integration & decision matrix logic
│   │   └── decision_matrix.py      # compute_priority_bands_and_ranking
│   ├── models/                     # Model training & loading utilities
│   ├── monitoring/                 # Prometheus & Evidently metrics
│   └── tracking/                   # Weights & Biases telemetry
├── tests/                          # 35 test suites (437 passing tests)
└── docs/                           # Architecture documentation & logs
```

---

## 3. Dataset Characteristics & Authoritative Baselines

| Dimension | Total Units | Dakshina Kannada | Udupi | Uttara Kannada | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Administrative Units** | **1,879** | 369 (19.6%) | 253 (13.5%) | 1,257 (66.9%) | Revenue villages and census towns |
| **Geospatial Coordinates** | **1,879** | 369 | 253 | 1,257 | 100% have valid `lat` and `lon` |
| **AHP Preparedness Tiers** | | | | | |
| - High (> 60.0) | **596** (31.7%) | 168 | 134 | 294 | Authoritative structural capacity |
| - Medium (40.0 - 60.0) | **1,128** (60.0%) | 196 | 114 | 818 | Authoritative structural capacity |
| - Low (< 40.0) | **155** (8.2%) | 5 | 5 | 145 | Authoritative structural capacity |
| **Operational Priority Bands** | | | | | Assigned deterministically via rules |
| - Band 1: Critical Priority | **155** (8.2%) | 5 | 5 | 145 | Exact match with AHP Low |
| - Band 2: Specialized Readiness | **829** (44.1%) | 144 | 76 | 609 | Medium AHP + sector deficit trigger |
| - Band 3: Moderate Priority | **299** (15.9%) | 52 | 38 | 209 | Medium AHP without deficit trigger |
| - Band 4: Adequate Baseline | **596** (31.7%) | 168 | 134 | 294 | Exact match with AHP High |
| **Vulnerability Archetypes** | | | | | |
| - Cluster 0: Urban/Peri-Urban Corridor | **130** | 77 | 53 | 0 (NA) | Well-resourced, high infrastructure |
| - Cluster 1: Remote Foothill Communities | **82** | 49 | 33 | 0 (NA) | Healthcare-isolated, high elevation |
| - Cluster 2: Vulnerable Rural Interior | **161** | 98 | 63 | 0 (NA) | Socioeconomically vulnerable |
| - Cluster 3: High-Exposure Coastal Belt | **249** | 145 | 104 | 0 (NA) | <5 km to coastline, surge-exposed |
| - Unassigned Archetype (UK Fallback) | **1,257** | 0 | 0 | 1,257 | Independent deficit trigger engine |

---

## 4. Analytical Services & Model Artifacts

### 4.1 Decision Support Engine (`src.assistant.decision_support.DecisionSupportService`)
- Provides singleton data management over `unified_village_preparedness_v2.parquet` and `coastal_vulnerability_archetypes.parquet`.
- Enforces deterministic ranking via `compute_priority_bands_and_ranking`:
  1. Priority Band (Band 1 through Band 4)
  2. Preparedness Score ascending (lowest preparedness first)
  3. Minimum sector score `min(score_shelters, score_medical)` ascending
  4. Distance to coast `dist_coast_km` ascending
  5. Deterministic tie-breaker on `shrid2`
- Evaluates individual unit profiles (`evaluate_decision_unit`) including:
  - 5 AHP Domain Scores: Medical, Shelters, Civil Infrastructure, Early Warning, Socioeconomic.
  - Granular raw indicators: distance to hospital/PHC, tap water, pucca roads, SC/ST share, population density.
  - Surrogate model classification probabilities and margin.
  - Rule-based operational directives and action recommendations.

### 4.2 Machine Learning Surrogate (`models/logistic_regression_tier.joblib`)
- Architecture: scikit-learn `Pipeline(RobustScaler(quantile_range=[25.0, 75.0]), LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=500, random_state=42))`.
- Strict 26-feature input schema verified in `models/logistic_regression_metadata.json`.
- Performance metrics on hold-out test set (N=282):
  - Accuracy: **94.33%**
  - Macro F1: **90.60%**
  - Weighted F1: **94.22%**
  - Confusion Matrix:
    - High: 85 True Positive, 0 Low, 5 Medium
    - Low: 17 True Positive, 0 High, 6 Medium
    - Medium: 164 True Positive, 3 High, 2 Low

### 4.3 Deterministic Tool Registry (`src.assistant.tools.registry.ToolRegistry`)
Centralized registry containing 7 deterministic analytical tools:
1. `lookup_community`: Single community metadata and identifier search.
2. `analyze_infrastructure_and_exposure`: Sector indicators, healthcare isolation, coastal risk.
3. `explain_preparedness_assessment`: Authoritative AHP breakdown, priority band, model surrogate alignment.
4. `get_intervention_recommendations`: Grounded operational directives based on deficit triggers.
5. `compare_districts`: Comparative cross-district analytics and indicator distributions.
6. `filter_and_rank_communities`: Multi-criteria querying, sorting, and pagination across all 1,879 units.
7. `explain_vulnerability_archetype`: Descriptive profiles and Uttara Kannada fallback documentation.

---

## 5. Streamlit Page-to-Feature Mapping & Target React Routes

| Streamlit Page (`src/assistant/views/`) | Core Features | Target React Page / Route |
| :--- | :--- | :--- |
| **Overview** (`overview.py`) | - Regional 1,879 unit KPIs (Total, Bands 1-4)<br>- District cross-tabulation table & stacked bars<br>- Preparedness Tier vs Priority Band distribution<br>- Vulnerability archetype breakdown<br>- Scientific vintage governance banner | **Main Dashboard** (`/`) |
| **Regional GIS Explorer** (`gis_explorer.py`) | - Interactive Leaflet/MapLibre map with 1,879 unit markers<br>- Multi-criteria filters (District, Band, Tier, Archetype)<br>- Community search with quick fly-to<br>- Layer controls and color-coded priority markers<br>- Interactive marker popup and selection side drawer | **Geospatial Intelligence** (`/map`) |
| **Community Profiles** (`community_profile.py`) | - Search & autocomplete by Name, PC11 ID, or Taluk<br>- Quick-pick selector for representative communities<br>- Comprehensive Dossier: AHP Score, Priority Band, Rank<br>- 5 Sector indicators & raw infrastructure proxies<br>- Demographic profile & vulnerability archetype<br>- ML Surrogate prediction probabilities & disclaimer<br>- Grounded operational response advisories | **Community Explorer & Detail** (`/communities`, `/communities/:id`) |
| **Scenario Analysis** (`scenario_analysis.py`) | - MCDA weight customization sliders (Health, Demo, Infra, Coast)<br>- Population scale toggle<br>- Dynamic re-ranking & distribution statistics<br>- Baseline vs Simulated policy comparison<br>- Spearman rank correlation & rank shift metrics<br>- Rank displacement scatter plot | **District & Policy Intelligence** (`/districts`, `/policy-simulator`) |
| **Vulnerability Archetypes** (`archetypes.py`) | - 4 K-Means cluster deep-dive cards<br>- 622 DK/Udupi assigned vs 1,257 UK fallback accounting<br>- Radar / multi-bar dimension deficit comparison<br>- Intervention guidance by archetype | **Analytics & Model Intelligence** (`/analytics`) |
| **Reports and Export** (`reports.py`) | - Regional executive summary dossier<br>- Priority action lists exportable to CSV/JSON<br>- District briefing profiles<br>- Single community printable dossier | **Reports & Export** (`/reports`) |
| **Cyclone Assistant** (`chat_assistant.py`) | - Multi-turn conversational interface<br>- Suggested inquiry prompts<br>- Live tool execution telemetry (timing, inputs, status)<br>- Authoritative ground truth reconciliation block<br>- Provider status indicator (Gemini / OpenRouter / Deterministic Fallback) | **CycloneLM Assistant** (`/assistant`) |
| **AI Readiness** (`ai_readiness.py`) | - Architectural specification and RAG blueprint<br>- Governance guardrails distinguishing generation from calculation | Integrated into Assistant and Architecture docs |

---

## 6. Target Architecture: FastAPI Backend & React Frontend

### 6.1 Backend Target Structure (`api/`)
```text
api/
├── main.py                     # FastAPI app factory, CORS, lifespan, routes mounting
├── dependencies.py             # Dependency injection (DecisionSupportService, ModelContainer, Orchestrator)
├── routers/
│   ├── health.py               # GET /api/health
│   ├── dashboard.py            # GET /api/dashboard/overview
│   ├── districts.py            # GET /api/districts, GET /api/districts/{id}
│   ├── communities.py          # GET /api/communities, GET /api/communities/{id}
│   ├── maps.py                 # GET /api/maps/communities (GeoJSON / spatial records)
│   ├── analytics.py            # GET /api/analytics/{preparedness,priority-bands,clusters,model-performance}
│   ├── simulation.py           # POST /api/simulation/prioritize, POST /api/simulation/compare
│   └── assistant.py            # POST /api/assistant/chat, GET /api/assistant/tools
└── schemas/                    # Pydantic v2 schemas validating all inputs/outputs
```

### 6.2 Frontend Target Structure (`frontend/`)
```text
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
├── src/
│   ├── App.tsx                 # Root router and QueryClient provider
│   ├── main.tsx                # DOM entry point
│   ├── index.css               # Tailwind CSS directives & theme variables
│   ├── components/             # Reusable atomic UI components
│   │   ├── Badge.tsx           # Semantic badges for Tiers, Bands, Archetypes
│   │   ├── Button.tsx
│   │   ├── Card.tsx            # Clean enterprise cards
│   │   ├── DisclaimerBanner.tsx # Scientific governance and vintage disclaimers
│   │   ├── ErrorState.tsx
│   │   ├── KpiCard.tsx         # Standardized KPI metric cards
│   │   ├── LoadingSpinner.tsx
│   │   ├── Modal.tsx
│   │   └── Navbar.tsx & Sidebar.tsx # Navigation layouts
│   ├── layouts/
│   │   └── MainLayout.tsx      # Sidebar + Header + Content shell
│   ├── pages/
│   │   ├── DashboardPage.tsx   # Command-center regional overview
│   │   ├── MapPage.tsx         # Geospatial intelligence interactive map
│   │   ├── CommunitiesPage.tsx # Searchable, filterable community table
│   │   ├── CommunityDetailPage.tsx # Deep-dive dossier for single unit
│   │   ├── DistrictsPage.tsx   # Cross-district comparative intelligence
│   │   ├── AnalyticsPage.tsx   # ML surrogate performance & archetype analytics
│   │   ├── AssistantPage.tsx   # CycloneLM conversational assistant
│   │   └── NotFoundPage.tsx
│   ├── hooks/                  # Custom React hooks (useCommunities, useChat, etc.)
│   ├── services/               # Typed API client functions using fetch / axios
│   ├── types/                  # TypeScript interface definitions (matching Pydantic schemas)
│   └── utils/                  # Formatting, color scales, number formatters
└── tests/                      # Vitest / React Testing Library unit tests
```

---

## 7. Feature-Parity Verification Matrix

| ID | Feature Requirement | Current Streamlit State | Target React Implementation | Verification Strategy |
| :---: | :--- | :--- | :--- | :--- |
| **F-01** | Total unit accounting (1,879) | Reconciled across views | Dashboard KPI & Summary Card | Check total === 1,879 |
| **F-02** | Priority Band distribution (155, 829, 299, 596) | Rendered in Overview | Dashboard KPI cards & Bar charts | Verify exact count match |
| **F-03** | K-Means Archetypes (622 assigned, 1,257 fallback) | Rendered in Archetypes | Analytics & Community Dossier | Verify cluster distribution |
| **F-04** | Interactive Geospatial Map (1,879 markers) | Folium iframe | Leaflet / MapLibre with marker clustering & color coding | Verify all 1,879 coordinates plotted |
| **F-05** | Map filtering (District, Band, Tier) | Dropdowns + rerender | Client/Server reactive filtering | Test filter combinations |
| **F-06** | Community Search & Autocomplete | Selectbox with 1,879 options | Debounced input + API autocomplete | Search "Tallur", "608720", "Mudbhatkal" |
| **F-07** | Community Deep-Dive Dossier | 4-tab Streamlit view | Dedicated detail route `/communities/:id` | Compare indicators with Streamlit |
| **F-08** | Surrogate ML class probabilities & margin | Displayed with caveat | Calibrated probability progress bars + disclaimer | Check prob_high, prob_med, prob_low |
| **F-09** | AHP Precedence rule safeguard | Injected warning box | Prominent alert banner if AHP Low != ML | Test AHP Low unit cannot be downgraded |
| **F-10** | Dynamic Policy Simulation (MCDA weights) | Slider re-ranking | Interactive sliders + reactive score graph | Test weight recomputation against backend |
| **F-11** | Cross-district comparison metrics | Markdown cross-tabulation | Recharts comparative grouped bar charts | Compare DK vs Udupi vs UK averages |
| **F-12** | Model Performance & Confusion Matrix | Hardcoded in view / JSON | Recharts confusion matrix heatmap & metrics | Compare against `logistic_regression_metadata.json` |
| **F-13** | Conversational Assistant with 7 Tools | Chat UI + tool telemetry | Modern chat UI with tool execution badges & markdown | Test query with tool grounding & fallback |
| **F-14** | Authoritative Ground Truth Audit | Reconciliation block in answer | Reconciliation callout in chat message | Verify AHP score & Band cited in answer |
| **F-15** | Data Export to CSV | Pandas `to_csv` download button | Direct CSV generation / download | Export priority action list |
| **F-16** | Zero Fabricated Data Guardrail | Enforced across codebase | Enforced via strict schema validation | Verify no mock alerts or live weather |

---

## 8. Migration Phase Roadmap

- **Phase 0 (This Phase)**: Architecture audit, baseline inventory, feature-parity checklist. Deliverable: `docs/react_migration_audit.md`.
- **Phase 1**: FastAPI Modular Service Layer (`api/`), Pydantic schemas, REST endpoints, endpoint tests matching authoritative baseline values.
- **Phase 2**: Frontend Foundation (`frontend/`), React 18/19, TypeScript, Vite, Tailwind CSS, React Router, TanStack Query, layout & shared components.
- **Phase 3**: Feature Migration:
  - Phase 3A: Dashboard & Executive Overview
  - Phase 3B: Geospatial Intelligence (Leaflet Map)
  - Phase 3C: Community Explorer & Detail Dossier
  - Phase 3D: District Intelligence & Comparative Analytics
  - Phase 3E: Analytics, Vulnerability Archetypes, and Model Diagnostics
  - Phase 3F: Conversational CycloneLM Assistant
- **Phase 4**: Testing, Integration & Parity Verification (Pytest, Vitest, Playwright).
- **Phase 5**: Production Readiness & Deployment Documentation.

*Audit complete and verified. Authorized to proceed to Phase 1: Backend API.*
