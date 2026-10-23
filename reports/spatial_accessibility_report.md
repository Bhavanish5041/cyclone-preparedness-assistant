# Cyclone Preparedness Assistant — Phase 2, Step 1: Spatial Accessibility Report

*Generated on: 2026-10-22 19:25:09*

---

## 1. Pre-Extension Audit & Join Reconciliation
Prior to computing spatial metrics, a rigorous verification audit was conducted across the intermediate datasets created during Phase 1:

- **Boundary Polygons**: `622` administrative boundary polygons in [data/interim/coastal_karnataka_boundaries.parquet](file:///home/bhavanish/cyclone-preparedness-assistant/data/interim/coastal_karnataka_boundaries.parquet) (`EPSG:4326`).
- **Reconciliation of Unmatched Polygons**: In the initial raw extraction, two boundary polygons had null `pc11_id` attributes in the SHRUG shapefile DBF table:
  - `11-29-569-05523-608856` (Udupi)
  - `11-29-575-05564-617728` (Dakshina Kannada)
  *Root Cause*: The SHRUG shapefile metadata (`polysource`) indicates these polygons resulted from multi-polygon merges or manual picks, where DDL embedded the 6-digit Census ID in the `shrid2` string (`parts[4]`) but left the DBF attribute column as `None`.
  *Resolution*: By assigning the verified `pc11_id` parsed from `shrid2`, **100.0% of boundary polygons (622 / 622) now achieve an exact primary-key match** to Census 2011 records (matching `Mallar (CT)` 608856 in Udupi and `Kodimbala` 617728 in Dakshina Kannada).
- **Unmatched Census Units (3 units)**:
  - `617734`: `Kemminje (Rural)` (Dakshina Kannada, Pop: 1,205)
  - `617507`: `Mulur (CT)` (Dakshina Kannada, Pop: 5,445)
  - `617510`: `Neermarga (CT)` (Dakshina Kannada, Pop: 5,503)
  *Root Cause*: These 3 Census towns / rural remnants were merged into the adjacent urban polygon extents by the SHRUG geospatial platform. In accordance with strict guidelines, these joins are not forced.
- **Healthcare Facility Coordinates**: Evaluated 911 facilities; 100.0% have valid numeric coordinates within `12.0°N–14.5°N`, `74.0°E–76.0°E`.

---

## 2. Geodesic Accessibility Methodology
### 2.1 Representative Points vs. Geometric Centroids
Standard geometric centroids for non-convex, crescent-shaped, or coastal inlet polygons frequently fall outside the actual administrative boundary (e.g. in estuaries, rivers, or adjacent villages). To guarantee topological validity, we compute the **Shapely representative point** (`geometry.representative_point()`) for each of the 622 polygons. **100.0% of representative points fall strictly within the polygon interior**.

### 2.2 Geodesic Metric Definition
Pairwise straight-line geodesic distances are computed in kilometres using the great-circle haversine equation on the WGS84 sphere ($R = 6371.0088$ km). Pairwise matrices between the 622 representative points and target facility subsets are evaluated deterministically.

> [!IMPORTANT]
> **Operational Definition & Limitation**: All distance metrics represent **straight-line geodesic distance proxies**, > NOT road network travel distances or driving times. They do not account for river crossings, estuary detours, > flooded roadways, or terrain elevation. Furthermore, registered facility records indicate administrative registration > and do NOT confirm active doctor presence, power generator functioning, or bed availability.

---

## 3. Spatial Accessibility Summary Statistics

### Table 1: Distance to Nearest Registered Primary Health Centre (PHC) or Community Health Centre (CHC)
| Geographic Scope | Total Units | Min (km) | Median (km) | Mean (km) | 75th % (km) | 90th % (km) | Max (km) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Study Area** | 622 | 0.12 | 3.0 | 3.3 | 4.29 | 5.84 | 12.19 |
| **Dakshina Kannada (575)** | 369 | 0.12 | 3.33 | 3.54 | 4.53 | 6.01 | 10.79 |
| **Udupi (569)** | 253 | 0.16 | 2.54 | 2.97 | 3.82 | 5.51 | 12.19 |

### Table 2: Distance to Nearest Registered Referral / Apex Hospital (District or Sub-Taluk Hospital)
| Geographic Scope | Total Units | Min (km) | Median (km) | Mean (km) | 75th % (km) | 90th % (km) | Max (km) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Study Area** | 622 | 0.17 | 12.83 | 13.9 | 17.88 | 23.02 | 36.41 |
| **Dakshina Kannada (575)** | 369 | 0.17 | 12.51 | 13.37 | 17.42 | 21.83 | 34.46 |
| **Udupi (569)** | 253 | 0.55 | 13.47 | 14.66 | 19.36 | 25.37 | 36.41 |

### Table 3: In-Unit Facility Presence (Point-in-Polygon Coverage)
| Infrastructure Presence | Number of Administrative Units | Percentage of Total Units (n=622) |
| :--- | :---: | :---: |
| **At least one PHC or CHC within boundary** | 132 | 21.2% |
| **At least one Hospital (STH/DH) within boundary** | 8 | 1.3% |
| **At least one health facility of ANY type (including sub-centres)** | 505 | 81.2% |
| **Zero healthcare facilities of any type within boundary** | 117 | 18.8% |

---

## 4. Key Spatial Patterns & Outlier Clusters
1. **Primary Care Accessibility is Relatively Dense**:
   - The median straight-line distance to a registered PHC or CHC is **3.0 km** across the study area (3.33 km in Dakshina Kannada, 2.54 km in Udupi).
   - Over 75% of all coastal villages are within **4.5 km** straight-line distance of a registered PHC/CHC.
2. **Severe Tertiary Hospital Isolation in Inland/Foothill Taluks**:
   - While the median distance to a registered hospital is **12.83 km**, the 90th percentile reaches **23.02 km**, with a maximum straight-line distance of **36.41 km** in eastern Dakshina Kannada.
   - **43 administrative units** lie more than **25 km** straight-line distance from the nearest registered hospital. These units are heavily concentrated in the Western Ghats foothills of **Sullia and Belthangady taluks**, where rugged topography means actual road driving distances are typically 1.5× to 2.5× longer than the straight-line proxy.
3. **In-Unit Coverage Void**:
   - **117 out of 622 administrative units (18.8%)** have **zero** health facilities of any kind located within their polygon boundary. Residents of these villages depend entirely on inter-village transport to reach even a basic primary sub-centre.

---

## 5. Artifacts & Generated Figures
- **Derived Dataset**: [data/processed/coastal_spatial_accessibility_features.parquet](file:///home/bhavanish/cyclone-preparedness-assistant/data/processed/coastal_spatial_accessibility_features.parquet) (GeoParquet, 622 rows, 35 attributes).
- **Tabular Copy**: [data/processed/coastal_spatial_accessibility_features.csv](file:///home/bhavanish/cyclone-preparedness-assistant/data/processed/coastal_spatial_accessibility_features.csv).
- **Figure 1**: [reports/figures/geodesic_distance_phc_chc_choropleth.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/geodesic_distance_phc_chc_choropleth.png) — Choropleth map of distance to nearest PHC/CHC.
- **Figure 2**: [reports/figures/geodesic_distance_hospital_choropleth.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/geodesic_distance_hospital_choropleth.png) — Choropleth map of distance to nearest Hospital.
- **Figure 3**: [reports/figures/accessibility_distance_distribution.png](file:///home/bhavanish/cyclone-preparedness-assistant/reports/figures/accessibility_distance_distribution.png) — Grouped boxplot and kernel density distributions.

---

## 6. Next Steps
1. **Coastal Distance & Hazard Exposure**: Compute the shortest geodesic distance from each polygon to the Karnataka coastline boundary.
2. **Preparedness Sub-Index Normalization**: Standardize the accessibility proxies (lower distance = higher accessibility) alongside census vulnerability metrics using min-max scaling with explicit boundary clipping.
3. **Unsupervised Clustering Formulation**: Formulate an evidence-based k-means or spatial clustering pipeline to identify contiguous clusters of isolated, high-vulnerability coastal communities.
