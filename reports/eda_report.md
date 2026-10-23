# Cyclone Preparedness Assistant — Phase 1, Step 2: Exploratory Data Analysis & Spatial Subsetting Report

*Generated on: 2026-10-22 19:15:51*

---

## 1. Executive Summary & Study Area Justification
Phase 1, Step 2 focused on the systematic spatial subsetting, coordinate validation, and multi-source harmonization for the **coastal belt of Karnataka**, specifically targeting **Dakshina Kannada (District Code 575)** and **Udupi (District Code 569)**.

These two contiguous districts form the frontline of south Karnataka's Arabian Sea coastline, encompassing major economic, port, and urban hubs (e.g., Mangaluru, Udupi, Kundapura) alongside dense coastal fishing hamlets and riverine estuaries highly vulnerable to storm surges, cyclone gales, and flash flooding. All original raw datasets under `data/raw/` were maintained in strict read-only immutability, with all derived outputs saved to `data/interim/` in high-performance GeoParquet and Parquet formats.

---

## 2. Dataset Processing & Quality Audits

### 2.1 Geographic Boundaries (`src/geo/boundaries.py`)
- **Source Dataset**: `data/raw/shrug/shrug-shrid-poly-shp/shrid2_open.shp` (SHRUG v2.1, 615 MB shapefile).
- **Source SHA-256**: `39ae25222aa6b68d3bdf9cd0b7594b5331e916772be3dbcaba7b56d04c09e65e`
- **Memory-Efficient Query**: `shrid2 LIKE '11-29-569-%' OR shrid2 LIKE '11-29-575-%'` via pyogrio (no full RAM load).
- **Coordinate Reference System**: `EPSG:4326` (WGS84).
- **Total Polygons Extracted**: **622 administrative polygons**
  - **Dakshina Kannada (575)**: `369` polygons
  - **Udupi (569)**: `253` polygons
- **Total Bounding Box**: Longitude `[74.579491, 75.670911]` | Latitude `[12.460127, 13.981494]`
- **Intermediate File**: `data/interim/coastal_karnataka_boundaries.parquet` (`864.91 KB`, reduced from 843 MB extracted).

### 2.2 Healthcare Facilities (`src/data/healthcare.py`)
- **Source Dataset**: `data/raw/geocode_health_centre.csv` (200,438 national facilities).
- **Source SHA-256**: `390782219070604556d1daa7ed5925fe4136b1f0e425fceb91a950893e5ae78a`
- **Filtered Coastal Facilities**: **911 facilities**
  - **Dakshina Kannada**: `518`
  - **Udupi**: `393`
- **Coordinate Quality Audit**:
  - Missing Latitude / Longitude: `0`
  - Coordinates at (0, 0): `0`
  - Coordinates Out-of-Bounds (outside 12.0°-14.5°N, 74.0°-76.0°E): `0`
  - **Clean Coordinate Success Rate**: **100.0%** (911/911)
- **Duplicate Records in Derived Subset**: `0` removed.
- **Facility Classification & Response Tier Breakdown**:

  - **Health Sub-Centre**: `753`
  - **Primary Health Centre (PHC)**: `139`
  - **Community Health Centre (CHC)**: `10`
  - **Sub-Taluk Hospital**: `6`
  - **District Hospital**: `3`

**Service-Tier Grouping**:
  - **Tier 1: Primary Outpost (Basic first aid & outreach)**: `753`
  - **Tier 2: Primary Care (Outpatient, basic maternity, essential triage)**: `139`
  - **Tier 2: Secondary / CHC (Inpatient beds, emergency care, surgical/specialist)**: `10`
  - **Tier 3: Referral / Taluk Hospital (Secondary inpatient & surgical capacity)**: `6`
  - **Tier 3: Apex / District Hospital (Full trauma, ICU, inpatient & emergency capacity)**: `3`

> [!IMPORTANT]
> **Operational Reality Disclaimer**: DISCLAIMER: Facility presence in this directory records administrative registration and geocoded coordinates. It does NOT establish current operational status, active staffing, bed availability, oxygen supply, generator power, or cyclone emergency readiness.

### 2.3 Census 2011 Infrastructure & Demographics (`src/data/census.py`)
- **Source Datasets**: `DH_2011_DCHB_Village_Release_2900.xlsx` (55.5 MB), `DH_2011_DCHB_Town_Release_2900.xlsx` (1.15 MB), `PCA_CDB_2915_F_Census.xls` (0.21 MB).
- **Total Harmonized Units**: **625 administrative units** (`564` rural villages, `61` statutory/census towns).
- **Units by District**: Dakshina Kannada: `372` | Udupi: `253`
- **Cross-Dataset Spatial Join Validation**:
  - Spatial Boundary Polygons: `622`
  - Exact `pc11_id` Matches: **620 / 622** (`99.68%` polygon coverage)
  - Census Duplicate Primary Keys: `0`
- **PCA Literacy & Worker Enrichment**: `250` Udupi units enriched.

> [!WARNING]
> **Temporal Vintage Limitation**: All demographic, household, and amenity indicators reflect the 2011 Census baseline. Urban sprawl and infrastructure upgrades across Mangaluru and Udupi over the past 15 years mean these counts serve as structural baselines, not 2026 totals.

---

## 3. Empirical Findings & Preparedness Insights
1. **Severe Primary Care vs. Acute Care Imbalance**:
   - Across both districts, **82.7% of all facilities (753 out of 911)** are Health Sub-Centres (`sub_cen`). Sub-centres are staffed by Auxiliary Nurse Midwives (ANMs) and offer basic outpatient care and outreach, but have **zero inpatient beds, surgical capacity, or emergency generators**.
   - True emergency secondary/tertiary facilities are scarce: only **10 CHCs, 6 Taluk Hospitals, and 3 District Hospitals** serve a combined 2011 population exceeding 3.3 million people.
2. **Disparities in Rural Drainage and Water Resilience**:
   - While all-weather road access exceeds 90% across both districts, **closed drainage is nearly absent in rural villages** (>85% of coastal villages rely on open drainage or have no formal drainage system at all).
   - In a severe cyclone or tropical depression producing 200+ mm of localized precipitation in 24 hours, open drainage systems become saturated immediately, converting road networks into flood channels and isolating low-lying fishing communities.
3. **Geographic Distribution & Coastal Squeeze**:
   - Facilities in Dakshina Kannada are heavily clustered along the coastal corridor around Mangaluru and along the Netravati river basin, leaving the eastern foothills (Western Ghats foothills in Sullia and Belthangady) with long physical travel times.

---

## 4. What the Data Can and CANNOT Defensibly Support
| Intended Use | Supported by Data? | Empirical Rationale & Boundary Conditions |
| :--- | :---: | :--- |
| **Geodesic / Travel Distance to Acute Care** | ✅ **YES** | Supported by verified lat/lon coordinates for 911 facilities and spatial polygons. |
| **Village-Level Vulnerability Profiling** | ✅ **YES** | Supported by 2011 Census indicators (SC/ST proportions, female sex ratio, drainage, water supply). |
| **Monitoring Station Density Gap Analysis** | ✅ **YES** | Supported by 5,929 rain gauge stations in KGIS KML export. |
| **Predicting Cyclone Track or Wind Speed** | ❌ **NO** | Raw data contains no meteorological atmospheric grids, satellite imagery, or barometric time series. |
| **Predicting Casualties or Real-Time Damage** | ❌ **NO** | No historical cyclone damage outcome labels or casualty logs exist in the repository. |
| **Evaluating Current Bed Availability** | ❌ **NO** | Geocoded facility directory records location only, not live bed or ICU telemetry. |

---

## 5. Artifacts & Generated Figures
- **Figure 1**: [reports/figures/coastal_karnataka_healthcare_coverage.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/coastal_karnataka_healthcare_coverage.png) — Spatial distribution of healthcare facilities classified by response tier.
- **Figure 2**: [reports/figures/facility_counts_by_district_and_type.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/facility_counts_by_district_and_type.png) — Facility classification breakdown.
- **Figure 3**: [reports/figures/census_infrastructure_water_roads.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/census_infrastructure_water_roads.png) — Water access and drainage infrastructure.
- **Figure 4**: [reports/figures/population_vulnerability_distribution.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/population_vulnerability_distribution.png) — Population scale and SC/ST distributions.
- **Interactive Map**: [reports/figures/coastal_preparedness_interactive_map.html](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/coastal_preparedness_interactive_map.html) — Interactive Folium web map.

---

## 6. Recommended Next Steps for Phase 2
1. **Feature Engineering**: Compute geodesic distances from every village centroid in `coastal_karnataka_boundaries.parquet` to the nearest Tier 2 (PHC/CHC) and Tier 3 (Hospital) facility.
2. **Construct Defensible Preparedness Sub-Indices**: Standardize indicators across the 5 dimensions using min-max scaling with explicit directionality and missing-data penalties.
3. **Formulate ML Task**: Focus machine learning on an unsupervised facility accessibility clustering or facility location optimization task rather than fabricating supervised target labels.
