"""Data provenance registry and metadata catalog for raw datasets.

Maintains verified information regarding source organizations, licensing,
temporal vintage, geographic boundaries, intended project use, and documented limitations.
Adheres strictly to evidence-based documentation: unknown or unverified attributes
are explicitly flagged as 'unknown' or 'needs verification' rather than assumed.
"""

from typing import Any, Dict

KNOWN_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "geocode_health_centre.csv": {
        "original_filename": "geocode_health_centre.csv",
        "source_org": "Ministry of Health and Family Welfare (MoHFW) / National Health Mission (NHM), Govt. of India",
        "source_url": "https://data.gov.in (Open Government Data Platform India)",
        "download_date": "unknown",
        "license": "Government Open Data License - India (GODL) (needs verification of exact snapshot terms)",
        "geographic_coverage": "All India (200,438 facilities across all States/UTs; 12,192 in Karnataka; 518 in Dakshina Kannada)",
        "temporal_coverage": "approx. 2015-2019 (needs verification of release vintage)",
        "known_limitations": (
            "Exact geocoding accuracy varies across rural facilities; some entries have missing or zero coordinates; "
            "facility operational status may have changed; identical duplicate copy exists in raw directory."
        ),
        "intended_use": (
            "Primary spatial layer for healthcare accessibility (PHCs, CHCs, Sub-centres, Hospitals), "
            "distance calculations, and emergency healthcare coverage modeling."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Facility Name, Subdistrict Name, District Name, Latitude, Longitude",
        "duplicate_candidate_of": None,
    },
    "geocode_health_centre (1).csv": {
        "original_filename": "geocode_health_centre (1).csv",
        "source_org": "Ministry of Health and Family Welfare (MoHFW) / NHM, Govt. of India",
        "source_url": "https://data.gov.in (Open Government Data Platform India)",
        "download_date": "unknown",
        "license": "Government Open Data License - India (GODL)",
        "geographic_coverage": "All India (200,438 facilities)",
        "temporal_coverage": "approx. 2015-2019",
        "known_limitations": (
            "Byte-for-byte SHA-256 identical duplicate of geocode_health_centre.csv. "
            "Must be preserved in raw data for immutability but excluded from downstream ingestion."
        ),
        "intended_use": "Retained as raw duplicate record; excluded from analysis pipeline in favor of geocode_health_centre.csv.",
        "suitable_for_analysis": False,
        "primary_key_or_identifiers": "Facility Name, Subdistrict Name, District Name, Latitude, Longitude",
        "duplicate_candidate_of": "geocode_health_centre.csv",
    },
    "Hospital_with_discipline_jul_15.csv": {
        "original_filename": "Hospital_with_discipline_jul_15.csv",
        "source_org": "Ministry of Health and Family Welfare (MoHFW) / National Health Portal",
        "source_url": "https://data.gov.in",
        "download_date": "unknown",
        "license": "Government Open Data License - India (GODL) (needs verification)",
        "geographic_coverage": "All India directory of major hospitals (1,049 facilities nationwide)",
        "temporal_coverage": "July 2015",
        "known_limitations": (
            "Encoded in Windows-1252/latin1 (fails UTF-8 decode on byte 0x96); lacks geographic coordinates "
            "(no latitude/longitude); covers only a small curated subset of 1,049 major institutions nationally; "
            "over 10 years old; requires postal code or address geocoding to be spatially joined."
        ),
        "intended_use": (
            "Qualitative reference for medical specialties (trauma, cardiology, ICU beds, services) "
            "and hospital classification; not usable directly for spatial distance analysis without geocoding."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Hospital / Private, City, State, Area Pin Code",
        "duplicate_candidate_of": None,
    },
    "DH_2011_DCHB_Town_Release_2900.xlsx": {
        "original_filename": "DH_2011_DCHB_Town_Release_2900.xlsx",
        "source_org": "Office of the Registrar General & Census Commissioner, India (ORGI), Ministry of Home Affairs",
        "source_url": "https://censusindia.gov.in (District Census Handbook)",
        "download_date": "unknown",
        "license": "Census of India Terms of Use (Government Data) (needs verification)",
        "geographic_coverage": "Karnataka State (State code 2900, all 347 statutory and census towns)",
        "temporal_coverage": "2011 Census",
        "known_limitations": (
            "Census data is from 2011 (15 years old); town boundaries, population figures, and facility counts "
            "do not reflect 2026 ground reality; wide table (429 columns) requires selective column extraction."
        ),
        "intended_use": (
            "Baseline urban infrastructure indicators (hospital beds, doctors in position, fire fighting services, "
            "water storage capacity, electrification, drainage system, demographic totals)."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Town Code, District Code, Town Name",
        "duplicate_candidate_of": None,
    },
    "DH_2011_DCHB_Village_Release_2900.xlsx": {
        "original_filename": "DH_2011_DCHB_Village_Release_2900.xlsx",
        "source_org": "Office of the Registrar General & Census Commissioner, India (ORGI), Ministry of Home Affairs",
        "source_url": "https://censusindia.gov.in (District Census Handbook)",
        "download_date": "unknown",
        "license": "Census of India Terms of Use (Government Data) (needs verification)",
        "geographic_coverage": "Karnataka State (State code 2900, all 29,340 villages)",
        "temporal_coverage": "2011 Census",
        "known_limitations": (
            "Census data is from 2011; large spreadsheet (55.5 MB, 396 columns, 29,340 rows); distances to nearest "
            "medical and emergency amenities reflect 2011 network; requires streaming/read-only loading to prevent memory bloat."
        ),
        "intended_use": (
            "Baseline rural vulnerability indicators: distance to nearest PHC/CHC/hospital, all-weather road access, "
            "power availability, drinking water sources, demographic composition."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Village Code, Sub District Code, District Code, Village Name",
        "duplicate_candidate_of": None,
    },
    "PCA_CDB_2915_F_Census.xls": {
        "original_filename": "PCA_CDB_2915_F_Census.xls",
        "source_org": "Office of the Registrar General & Census Commissioner, India (ORGI)",
        "source_url": "https://censusindia.gov.in (Primary Census Abstract)",
        "download_date": "unknown",
        "license": "Census of India Terms of Use (needs verification)",
        "geographic_coverage": "Udupi District (District code 2915 / 569, Karnataka coastal district)",
        "temporal_coverage": "2011 Census",
        "known_limitations": (
            "Census 2011 vintage; covers Udupi district exclusively (CD block and village/town aggregate); "
            "does not cover Dakshina Kannada or other districts directly."
        ),
        "intended_use": (
            "Demographic baseline (population, SC/ST, literacy, worker categories, sex ratio) for coastal Udupi "
            "to cross-validate rural vulnerability indicators."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Town/Village_Code, CD Block_Code, District_Code",
        "duplicate_candidate_of": None,
    },
    "DH_2011_2921-Dakshina_Kannada.zip": {
        "original_filename": "DH_2011_2921-Dakshina_Kannada.zip",
        "source_org": "Office of the Registrar General & Census Commissioner, India (ORGI)",
        "source_url": "https://censusindia.gov.in",
        "download_date": "unknown",
        "license": "Census of India Terms of Use (needs verification)",
        "geographic_coverage": "Dakshina Kannada District (District code 2921, Karnataka)",
        "temporal_coverage": "2011 Census",
        "known_limitations": (
            "Compressed collection of 62 individual Excel files (T01-T45, Town Statements, Appendices); "
            "2011 vintage; redundant with state-level release tables if not selectively utilized."
        ),
        "intended_use": (
            "Granular district tables for Dakshina Kannada when specific cross-tabulations (e.g. amenities, "
            "slums, detailed commodity manufacturing) are required."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Table specific (Village/Town codes)",
        "duplicate_candidate_of": None,
    },
    "DH_2011_2921_PART_A_DCHB_DAKSHINA_KANNADA.pdf": {
        "original_filename": "DH_2011_2921_PART_A_DCHB_DAKSHINA_KANNADA.pdf",
        "source_org": "Directorate of Census Operations, Karnataka / ORGI",
        "source_url": "https://censusindia.gov.in",
        "download_date": "unknown",
        "license": "Government Publication / Census of India",
        "geographic_coverage": "Dakshina Kannada District, Karnataka",
        "temporal_coverage": "2011 Census (compiled 2016)",
        "known_limitations": (
            "362-page unstructured narrative and tabular PDF; automated parsing is fragile compared to native spreadsheets; "
            "reflects 2011 administrative boundaries (e.g. before subsequent taluk re-organizations)."
        ),
        "intended_use": (
            "Definitive administrative, geographic, physiographic, and hazard background reference for Dakshina Kannada."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Document chapters and village directories",
        "duplicate_candidate_of": None,
    },
    "IntegratedCoastalVulnerabilityAssessment.pdf": {
        "original_filename": "IntegratedCoastalVulnerabilityAssessment.pdf",
        "source_org": "Disaster Advances journal (Authors: Deepa D. Naik & Pravin D. Kunte; Symbiosis & CSIR-NIO)",
        "source_url": "Disaster Advances Vol. 16 (7), July 2023 (ISSN 0974-262X)",
        "download_date": "unknown",
        "license": "Published academic article; Disaster Advances copyright (fair academic use for research reference)",
        "geographic_coverage": "Karnataka Coastline (10 coastal talukas across Dakshina Kannada, Udupi, and Uttara Kannada)",
        "temporal_coverage": "Published July 2023 (integrating multi-year physical and demographic records)",
        "known_limitations": (
            "Academic research paper; indices and findings are synthesized at taluka level rather than village level; "
            "formulae and weights must be extracted from document text."
        ),
        "intended_use": (
            "Provides scientific ground-truth and peer-reviewed methodology for coastal vulnerability dimensions, "
            "physical variables (shoreline change, sea level rise, wave height, coastal slope), and defensible indicator weights."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "Taluk names (e.g. Mangalore, Bantval, Udupi, Kundapura, Karwar, etc.)",
        "duplicate_candidate_of": None,
    },
    "70578d25-7ede-426c-8e8e-5c679274a55a.kml": {
        "original_filename": "70578d25-7ede-426c-8e8e-5c679274a55a.kml",
        "source_org": "Karnataka Geographic Information System (KGIS) / Karnataka State Remote Sensing Applications Centre (KSRSAC)",
        "source_url": "https://kgis.ksrsac.in (portal export)",
        "download_date": "unknown",
        "license": "Government of Karnataka / KGIS Portal Terms (needs verification)",
        "geographic_coverage": "Karnataka State (5,929 telemetric rain gauge stations across all districts)",
        "temporal_coverage": "Current monitoring station network (needs verification of snapshot date)",
        "known_limitations": (
            "Contains sensor location coordinates and station identifiers only; does NOT contain historical or real-time "
            "precipitation observations or time-series data; raw UUID filename lacks semantic meaning."
        ),
        "intended_use": (
            "Spatial mapping and spatial density evaluation of hydrometeorological monitoring infrastructure "
            "in cyclone-prone coastal zones."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "OBJECTID, KGISCode, KGISTM_RainGauge_LocationID, geometry",
        "duplicate_candidate_of": None,
    },
    "7bfe0453-27a3-4013-a09d-3f56e2d3ab15.kml": {
        "original_filename": "7bfe0453-27a3-4013-a09d-3f56e2d3ab15.kml",
        "source_org": "Karnataka Geographic Information System (KGIS) / KSRSAC / Water Resources Department",
        "source_url": "https://kgis.ksrsac.in (portal export)",
        "download_date": "unknown",
        "license": "Government of Karnataka / KGIS Portal Terms (needs verification)",
        "geographic_coverage": "Karnataka State (26 water level sensor stations, predominantly urban/Bengaluru)",
        "temporal_coverage": "Current monitoring station network (needs verification of snapshot date)",
        "known_limitations": (
            "Very sparse dataset (only 26 stations statewide); stations are located primarily in Bengaluru urban waterways, "
            "not along coastal storm surge / river mouth locations; does not contain sensor telemetry readings."
        ),
        "intended_use": (
            "Evaluated as candidate monitoring infrastructure; documented as an infrastructure gap along the coastal belt."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "OBJECTID, KGISWLS_StationID, KGISCode, geometry",
        "duplicate_candidate_of": None,
    },
    "shrug-shrid-poly-shp.zip": {
        "original_filename": "shrug-shrid-poly-shp.zip",
        "source_org": "Development Data Lab (DDL) (Paul Novosad, Sam Asher, et al.)",
        "source_url": "https://devdatalab.org/shrug",
        "download_date": "unknown (SHRUG v2.1)",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "geographic_coverage": "All India (595,438 village and town boundary polygons matching 2011 Census)",
        "temporal_coverage": "2011 Census boundaries (v2.1 released 2024)",
        "known_limitations": (
            "Very large compressed archive (340 MB); extracted shapefile is already present under data/raw/shrug/; "
            "tracking both in version control wastes GitHub storage; all-India coverage must be filtered to coastal Karnataka "
            "to prevent massive memory usage."
        ),
        "intended_use": (
            "Archive source for SHRUG geographic boundary polygons; redundant with extracted shrug/ directory."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "shrid2, pc11_id",
        "duplicate_candidate_of": None,
    },
    "shrid2_open.shp": {
        "original_filename": "shrid2_open.shp",
        "source_org": "Development Data Lab (DDL)",
        "source_url": "https://devdatalab.org/shrug",
        "download_date": "unknown (SHRUG v2.1)",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "geographic_coverage": "All India (595,438 village and town boundary polygons)",
        "temporal_coverage": "2011 Census boundaries",
        "known_limitations": (
            "Extremely large file (615 MB .shp, 234 MB .dbf); loading full dataset into memory crashes lightweight environments; "
            "must be read using spatial bounding box or attribute filters (pc11_state_id == '29') to subset Karnataka."
        ),
        "intended_use": (
            "Primary geographic boundary layer for joining Census 2011 indicators (via pc11_id) and rendering "
            "village/town choropleth maps in coastal Karnataka."
        ),
        "suitable_for_analysis": True,
        "primary_key_or_identifiers": "shrid2, pc11_id, geometry",
        "duplicate_candidate_of": None,
    },
    "REPORT-8-OF-2025---ENGLISH-0694389fd547e10.69426873.pdf": {
        "original_filename": "REPORT-8-OF-2025---ENGLISH-0694389fd547e10.69426873.pdf",
        "source_org": "Comptroller and Auditor General of India (CAG) (suggested by report naming convention)",
        "source_url": "unknown (needs verification)",
        "download_date": "unknown",
        "license": "needs verification",
        "geographic_coverage": "unknown",
        "temporal_coverage": "2025",
        "known_limitations": (
            "File size is exactly 0 bytes (empty/truncated file). Completely unreadable. "
            "Must be marked invalid/corrupt in all data manifests and excluded from processing."
        ),
        "intended_use": "None in current state. Requires re-download if the report is relevant.",
        "suitable_for_analysis": False,
        "primary_key_or_identifiers": "none",
        "duplicate_candidate_of": None,
    },
}


def get_provenance_info(filename_or_path: str) -> Dict[str, Any]:
    """Retrieve verified provenance information for a given file name or path.

    Falls back to a standardized 'needs verification' dict if the file is unknown.
    """
    import os

    basename = os.path.basename(filename_or_path)
    if basename in KNOWN_PROVENANCE:
        return dict(KNOWN_PROVENANCE[basename])

    # Check partial / case-insensitive match
    for key, val in KNOWN_PROVENANCE.items():
        if key.lower() == basename.lower():
            return dict(val)

    return {
        "original_filename": basename,
        "source_org": "unknown (needs verification)",
        "source_url": "unknown (needs verification)",
        "download_date": "unknown",
        "license": "unknown (needs verification)",
        "geographic_coverage": "unknown",
        "temporal_coverage": "unknown",
        "known_limitations": "Provenance not yet verified",
        "intended_use": "Evaluation pending inspection",
        "suitable_for_analysis": False,
        "primary_key_or_identifiers": "unknown",
        "duplicate_candidate_of": None,
    }
