# Cyclone Preparedness Assistant — Raw Data Inventory and Quality Report

*Generated on: 2026-10-22 16:15:39*

---

## 1. Executive Summary
The raw data repository under `data/raw/` contains **20 discovered files** totaling **1295.19 MB (~1.26 GB)**. The inventory was executed with memory-safe chunking, header inspections, and shapefile/KML metadata extraction without loading large geographic layers into RAM.

### Summary Statistics
- **Total files cataloged**: `20`
- **Total raw data size**: `1295.19 MB` (1.26 GB)
- **Exact duplicate file pairs detected**: `1`
- **Empty / invalid files detected**: `1`
- **Format breakdown**: `BIB`: 1, `CPG`: 1, `CSV`: 3, `DBF`: 1, `KML`: 2, `MD`: 1, `PDF`: 3, `PRJ`: 1, `SHP`: 1, `SHX`: 1, `XLS`: 1, `XLSX`: 2, `ZIP`: 2

---

## 2. Complete Asset Manifest
| Relative Path | Format | Size (MB) | Status | Records / Features | CRS | Suitable? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `data/raw/70578d25-7ede-426c-8e8e-5c679274a55a.kml` | KML | 3.31 | `valid` | 5929 | EPSG:4326 | ✅ Yes |
| `data/raw/7bfe0453-27a3-4013-a09d-3f56e2d3ab15.kml` | KML | 0.01 | `valid` | 26 | EPSG:4326 | ✅ Yes |
| `data/raw/DH_2011_2921-Dakshina_Kannada.zip` | ZIP | 0.70 | `valid` | 62 | - | ✅ Yes |
| `data/raw/DH_2011_2921_PART_A_DCHB_DAKSHINA_KANNADA.pdf` | PDF | 10.20 | `valid` | 362 | - | ✅ Yes |
| `data/raw/DH_2011_DCHB_Town_Release_2900.xlsx` | XLSX | 1.15 | `valid` | - | - | ✅ Yes |
| `data/raw/DH_2011_DCHB_Village_Release_2900.xlsx` | XLSX | 55.47 | `valid` | - | - | ✅ Yes |
| `data/raw/Hospital_with_discipline_jul_15.csv` | CSV | 0.44 | `valid` | 1049 | - | ✅ Yes |
| `data/raw/IntegratedCoastalVulnerabilityAssessment.pdf` | PDF | 1.02 | `valid` | 13 | - | ✅ Yes |
| `data/raw/PCA_CDB_2915_F_Census.xls` | XLS | 0.21 | `valid` | - | - | ✅ Yes |
| `data/raw/REPORT-8-OF-2025---ENGLISH-0694389fd547e10.69426873.pdf` | PDF | 0.00 | `empty_or_invalid` | None | - | 🚫 Empty (0B) |
| `data/raw/geocode_health_centre (1).csv` | CSV | 20.00 | `valid` | 200438 | - | ⚠️ Duplicate |
| `data/raw/geocode_health_centre.csv` | CSV | 20.00 | `valid` | 200438 | - | ✅ Yes |
| `data/raw/shrug/shrug-shrid-poly-shp/README.md` | MD | 0.00 | `sidecar_or_doc` | - | - | ❌ No |
| `data/raw/shrug/shrug-shrid-poly-shp/open_poly.bib` | BIB | 0.00 | `sidecar_or_doc` | - | - | ❌ No |
| `data/raw/shrug/shrug-shrid-poly-shp/shrid2_open.cpg` | CPG | 0.00 | `sidecar_or_doc` | - | - | ❌ No |
| `data/raw/shrug/shrug-shrid-poly-shp/shrid2_open.dbf` | DBF | 223.17 | `sidecar_or_doc` | - | - | ❌ No |
| `data/raw/shrug/shrug-shrid-poly-shp/shrid2_open.prj` | PRJ | 0.00 | `sidecar_or_doc` | - | - | ❌ No |
| `data/raw/shrug/shrug-shrid-poly-shp/shrid2_open.shp` | SHP | 615.36 | `valid` | 595438 | EPSG:4326 | ✅ Yes |
| `data/raw/shrug/shrug-shrid-poly-shp/shrid2_open.shx` | SHX | 4.54 | `sidecar_or_doc` | - | - | ❌ No |
| `data/raw/shrug-shrid-poly-shp.zip` | ZIP | 339.61 | `valid` | 8 | - | ✅ Yes |

---

## 3. Critical Data Findings & Quality Assessment

### 3.1 Exact Duplicate Files Detected
- **SHA-256 Digest**: `390782219070604556d1daa7ed5925fe4136b1f0e425fceb91a950893e5ae78a`
  - Primary file: `data/raw/geocode_health_centre.csv`
  - Redundant copy: `data/raw/geocode_health_centre (1).csv` (Identical size and content)

> [!NOTE]
> Raw copies remain strictly immutable under `data/raw/`. However, the downstream ingestion pipeline > must exclusively reference the primary file `data/raw/geocode_health_centre.csv` and omit the redundant copy.

### 3.2 Empty and Corrupt Files
- **File**: `data/raw/REPORT-8-OF-2025---ENGLISH-0694389fd547e10.69426873.pdf`
  - **Size**: `0 bytes`
  - **SHA-256**: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - **Diagnosis**: Truncated download or unwritten export. Marked `invalid/corrupted`.
  - **Action**: Exclude from all data processing. If the 2025 CAG report is needed, it must be re-downloaded.

### 3.3 Encoding and Formatting Irregularities
- **`Hospital_with_discipline_jul_15.csv`**:
  - **Issue**: Standard UTF-8 decoding fails on byte `0x96` (position 14984, representing an en-dash `–` in Windows-1252 / CP1252).
  - **Remediation**: The ingestion pipeline must use `encoding='latin1'` or `encoding='cp1252'`.
  - **Spatial limitation**: Contains institution names, cities, and PIN codes, but **lacks latitude/longitude coordinates**. Cannot be used as a standalone GIS point layer without external postal geocoding.

### 3.4 Shapefile Storage and Archive Redundancy
- **`data/raw/shrug-shrid-poly-shp.zip`** (339.61 MB) and **`data/raw/shrug/shrug-shrid-poly-shp/`** (843.08 MB extracted):
  - **Issue**: The extracted directory contains the exact contents of the ZIP archive (`shrid2_open.shp`, `shrid2_open.shx`, `shrid2_open.dbf`, `shrid2_open.prj`).
  - **Git LFS Consideration**: Tracking both an 843 MB extracted shapefile and a 340 MB ZIP in Git LFS wastes GitHub LFS storage quotas (2 GB limit on standard accounts).
  - **Recommendation**: Raw archives should remain stored locally. Only filtered coastal subsets (Karnataka coastal taluks) should be exported to `data/processed/` and tracked.

---

## 4. Dataset Provenance and Preparedness Framework Suitability
| Asset Name | Source Organization | Vintage | Geographic Coverage | Candidate Preparedness Dimension |
| :--- | :--- | :---: | :--- | :--- |
| `geocode_health_centre.csv` | MoHFW / National Health Portal | ~2015-2019 | All-India (12,192 in Karnataka; 518 in Dakshina Kannada) | **Healthcare Accessibility** (PHC, CHC, hospital locations) |
| `Hospital_with_discipline_jul_15.csv` | MoHFW | July 2015 | All-India major hospitals (1,049 facilities) | **Healthcare Capacity** (Specialties, ICU, trauma reference) |
| `DH_2011_DCHB_Town_Release_2900.xlsx` | ORGI (Census of India) | 2011 | Karnataka State (347 towns) | **Urban Infrastructure & Demographics** (beds, drainage, fire service) |
| `DH_2011_DCHB_Village_Release_2900.xlsx` | ORGI (Census of India) | 2011 | Karnataka State (29,340 villages) | **Rural Vulnerability** (distances to PHC, roads, power, water) |
| `PCA_CDB_2915_F_Census.xls` | ORGI (Census of India) | 2011 | Udupi District (264 CD blocks/villages) | **Demographic Vulnerability** (SC/ST, literacy, worker categories) |
| `DH_2011_2921-Dakshina_Kannada.zip` | ORGI (Census of India) | 2011 | Dakshina Kannada District (62 tables) | **District Baseline** (Dakshina Kannada census tables) |
| `DH_2011_2921_PART_A_DCHB_DAKSHINA_KANNADA.pdf` | Directorate of Census Operations | 2011 | Dakshina Kannada District (362 pages) | **Context & Physiography** (qualitative baseline and administrative boundaries) |
| `IntegratedCoastalVulnerabilityAssessment.pdf` | Disaster Advances (Naik & Kunte) | July 2023 | Karnataka Coastline (10 coastal talukas) | **Coastal Hazard & Methodological Framework** (physical & socio-economic weights) |
| `70578d25-7ede-426c-8e8e-5c679274a55a.kml` | Karnataka GIS (KGIS) / KSRSAC | Current | Karnataka State (5,929 stations) | **Emergency Infrastructure** (Telemetric rain gauge network density) |
| `7bfe0453-27a3-4013-a09d-3f56e2d3ab15.kml` | Karnataka GIS (KGIS) / KSRSAC | Current | Karnataka State (26 stations) | **Flood Monitoring Gap** (Sparse water-level sensor coverage) |
| `shrid2_open.shp` (SHRUG v2.1) | Development Data Lab (DDL) | 2024 / Census 2011 | All-India (595,438 boundaries) | **Spatial Unit of Analysis** (Village/town boundary polygons for choropleths) |
| `REPORT-8-OF-2025...pdf` | CAG (presumed) | 2025 | Unknown | **Unusable** (0 bytes / corrupt file) |

---

## 5. Candidate Preparedness Framework Alignment
Based on the empirical evidence discovered across the raw datasets, the following **5-dimension preparedness framework** is grounded in the available data:
1. **Healthcare Accessibility Dimension**:
   - Supported by `geocode_health_centre.csv` (coordinates for 518 facilities in Dakshina Kannada, 12,192 in Karnataka) and `DH_2011_DCHB_Village_Release_2900.xlsx` (reported road distance to nearest PHC, CHC, and hospital).
   - *Indicators*: Travel time / geodesic distance to nearest acute health facility, ratio of population to primary health centres.
2. **Population Vulnerability Dimension**:
   - Supported by `PCA_CDB_2915_F_Census.xls` and `DH_2011_DCHB_Town/Village_Release_2900.xlsx`.
   - *Indicators*: Proportion of vulnerable demographic groups (elderly/child, SC/ST, agricultural laborers, female literacy rate, housing density).
   - *Crucial Caveat*: Census data reflects 2011 baseline; must be explicitly marked with vintage and freshness warnings.
3. **Emergency Infrastructure Dimension**:
   - Supported by `70578d25...kml` (telemetric rain gauges) and `DH_2011_DCHB_Town_Release_2900.xlsx` (fire fighting services, electricity connections, pucca road length).
   - *Indicators*: Proximity to all-weather pucca roads, presence of local fire services, rain gauge sensor density per taluk.
4. **Coastal Hazard Exposure Dimension**:
   - Supported by the peer-reviewed methodology in `IntegratedCoastalVulnerabilityAssessment.pdf` (Naik & Kunte, 2023).
   - *Indicators*: Physical vulnerability scores (regional elevation, coastal slope, historical shoreline change, tidal range).
5. **Data Freshness and Uncertainty Dimension**:
   - *Indicators*: Explicit confidence score penalizing outdated or missing metrics, alerting decision-makers when a score relies heavily on 2011 census figures.

---

## 6. Actionable Next Steps
1. **Immutability Enforcement**: Maintain all files under `data/raw/` in read-only condition.
2. **Interim Filtering Pipeline**: Build a spatial extraction script in `src/geo/` to extract only Karnataka coastal districts (Dakshina Kannada, Udupi, Uttara Kannada) from `shrid2_open.shp` and save to `data/interim/coastal_karnataka_bounds.parquet`.
3. **Facility Cleaning**: Parse `geocode_health_centre.csv`, filter to coastal districts, validate coordinates, and store clean GeoJSON/Parquet in `data/interim/`.
4. **Town & Village Directory ETL**: Extract relevant healthcare, infrastructure, and demographic columns from `DH_2011_DCHB_Town_Release_2900.xlsx` and `DH_2011_DCHB_Village_Release_2900.xlsx`.
5. **Git LFS Management**: Keep large raw archives (1.3 GB) local; track only code, tests, documentation, manifests, and small processed summaries in Git.
