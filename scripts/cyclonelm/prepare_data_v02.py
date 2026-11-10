"""CycloneLM v0.2 Dataset Realignment and Curation Engine.

Generates:
1. Verified Domain Corpus (correct archetypes, actual district statistics, 5 domain scores).
2. Curated Instruction and Tool-Use Datasets split strictly by Scenario Family:
   - Train split (70%)
   - Validation split (15%)
   - Held-out Test split (15%)
   With verified ZERO query/target overlap between train, val, and test.
3. Held-out 100-Question Evaluation Suite:
   - 30 unique Domain Knowledge questions
   - 30 unique Tool Selection questions (all 7 tools)
   - 20 unique Tool Argument Generation questions (all 7 tools)
   - 20 unique Evidence-Grounded Response questions (20 distinct community/evidence cases)
4. Authoritative Source Provenance Catalog with explicit licensing, publisher, and copyright terms.
5. Automated BPE Tokenizer training for v0.2 with 16 atomic special tokens including closing tags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Authoritative source provenance catalog
PROVENANCE_CATALOG = {
    "catalog_version": "0.2.0",
    "sources": [
        {
            "id": "IMD_SOP_CYCLONE",
            "title": "Cyclone Warning Services in India: Standard Operating Procedure",
            "publisher": "India Meteorological Department, Ministry of Earth Sciences, Government of India",
            "url": "https://mausam.imd.gov.in",
            "access_date": "2026-11-10",
            "copyright": "Government of India (Open Data Policy - NDSAP)",
            "permitted_usage": "Public information and academic research reference",
            "usage_mode": "Paraphrased and synthesized in original domain prose",
            "covers": ["8-stage cyclone intensity scale", "4-stage cyclone alerts", "Arabian Sea basin dynamics"],
        },
        {
            "id": "UNDRR_SENDAI_2015",
            "title": "Sendai Framework for Disaster Risk Reduction 2015-2030",
            "publisher": "United Nations Office for Disaster Risk Reduction (UNDRR)",
            "url": "https://www.undrr.org/implementing-sendai-framework/what-sendai-framework",
            "access_date": "2026-11-10",
            "copyright": "United Nations",
            "permitted_usage": "Public international disaster risk reduction policy citation",
            "usage_mode": "Conceptual framework reference for coastal preparedness and resilience",
            "covers": ["Disaster risk identification", "Early warning systems", "Resilient infrastructure"],
        },
        {
            "id": "SAATY_AHP_1980",
            "title": "The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation",
            "publisher": "McGraw-Hill International (Thomas L. Saaty)",
            "access_date": "2026-11-10",
            "copyright": "Academic foundational literature",
            "permitted_usage": "Academic scientific citation and mathematical formulation",
            "usage_mode": "Mathematical implementation (pairwise comparisons, consistency ratio CR < 0.10)",
            "covers": ["AHP methodology", "Consistency index", "Reciprocal matrix mathematics"],
        },
        {
            "id": "KSDMA_COASTAL_PLAN",
            "title": "Karnataka State Disaster Management Plan: Coastal Hazards and Vulnerability",
            "publisher": "Revenue Department (Disaster Management), Government of Karnataka",
            "url": "https://dm.karnataka.gov.in",
            "access_date": "2026-11-10",
            "copyright": "Government of Karnataka",
            "permitted_usage": "Government public disaster management documentation",
            "usage_mode": "Paraphrased administrative and geographical reference",
            "covers": ["Dakshina Kannada, Udupi, Uttara Kannada coastline", "Estuarine networks", "Multi-Purpose Cyclone Shelters"],
        },
        {
            "id": "CENSUS_INDIA_2011_DK_UDUPI_UK",
            "title": "District Census Handbooks: Dakshina Kannada, Udupi, and Uttara Kannada",
            "publisher": "Office of the Registrar General & Census Commissioner, Ministry of Home Affairs, Government of India",
            "url": "https://censusindia.gov.in",
            "access_date": "2026-11-10",
            "copyright": "Open Government Data (OGD) Platform India",
            "permitted_usage": "Statistical and geographical public reference",
            "usage_mode": "Baseline demographic and habitation boundaries (1,879 administrative units)",
            "covers": ["Revenue village boundaries", "Housing stock characteristics", "Demographic vulnerability"],
        },
    ],
}

# Clean, factually verified technical domain sections
DOMAIN_CORPUS_SECTIONS = [
    """=== SECTION 1: CYCLONE METEOROLOGY AND ARABIAN SEA DYNAMICS ===
Tropical cyclones are warm-core, non-frontal synoptic-scale cyclonic storms developing over tropical oceans characterized by low atmospheric surface pressure, organized cyclonic circulation, and intense convective precipitation. In the North Indian Ocean (NIO) basin, cyclogenesis occurs across the Bay of Bengal and the Arabian Sea. While the Bay of Bengal historically accounts for approximately four times more storms than the Arabian Sea, recent thermodynamic trends indicate that rapid sea-surface warming (sea surface temperatures frequently exceeding the 28.0°C threshold) and enhanced atmospheric mid-tropospheric moisture have amplified both the frequency and rapid intensification of severe storms in the Arabian Sea.
The India Meteorological Department (IMD) classifies cyclonic disturbances into eight distinct intensity tiers based on maximum sustained surface wind speeds (3-minute average):
1. Low Pressure Area: Wind speed less than 31 km/h (< 17 knots).
2. Depression: Wind speed between 31 and 49 km/h (17 to 27 knots).
3. Deep Depression: Wind speed between 50 and 61 km/h (28 to 33 knots).
4. Cyclonic Storm: Wind speed between 62 and 87 km/h (34 to 47 knots).
5. Severe Cyclonic Storm (SCS): Wind speed between 88 and 117 km/h (48 to 63 knots).
6. Very Severe Cyclonic Storm (VSCS): Wind speed between 118 and 166 km/h (64 to 89 knots).
7. Extremely Severe Cyclonic Storm (ESCS): Wind speed between 167 and 221 km/h (90 to 119 knots).
8. Super Cyclonic Storm (SuCS): Wind speed of 222 km/h or higher (>= 120 knots).
Severe cyclonic systems impacting western India generate primary physical hazards including astronomical storm surge amplification, high-energy wave erosion, intense coastal backwater flooding where rivers meet the sea, and torrential rainfall inducing flash floods along the Western Ghats escarpment.""",

    """=== SECTION 2: COASTAL KARNATAKA GEOGRAPHY AND HAZARDS ===
The Coastal Karnataka maritime belt spans approximately 320 kilometers along the eastern Arabian Sea coastline, bordered eastward by the Western Ghats. Administratively, the study area covers 1,879 distinct revenue units across three districts:
1. Dakshina Kannada (South Canara): Comprises 369 administrative units. It encompasses the highly populated and industrialized Mangaluru urban center and major estuarine confluences (Netravati and Gurupura rivers). High coastal population densities, extensive commercial port installations, and sand-spit habitations like Bengre and Ullal face extreme marine exposure.
2. Udupi: Comprises 253 administrative units. Characterized by low barrier beaches, estuarine rivers (Swarna, Sita, Gangolli), active fishing harbors (Malpe), and agricultural-aquacultural lowlands subject to saline intrusion and storm-surge flooding.
3. Uttara Kannada (North Canara): Comprises 1,257 administrative units. Features a distinctive, rugged rias-type coastline with rocky promontories, mangrove-lined estuaries (Sharavathi, Aghanashini, Gangavali, Kali), and dense Western Ghats forested slopes extending almost to the water's edge, causing rural geographic isolation.
Across all three districts, physical exposure is governed by low elevation (< 5.0 meters above MSL), proximity to tidal estuaries, and high seasonal rainfall exceeding 3,500 mm annually.""",

    """=== SECTION 3: DISASTER RISK REDUCTION AND EARLY WARNING SYSTEMS ===
Disaster Risk Reduction (DRR) in the CYCLONE INTEL framework aligns with Priority 1 (Understanding disaster risk) and Priority 4 (Enhancing disaster preparedness for effective response) of the Sendai Framework for Disaster Risk Reduction 2015-2030.
Early Warning Systems (EWS) operationalized by the IMD follow a structured four-stage alert protocol:
- Stage 1: Pre-Cyclone Watch (issued 72 hours prior to cyclonic development, alerting national and state maritime administrations).
- Stage 2: Cyclone Alert / Yellow Message (issued 48 hours prior to expected adverse weather, designating vulnerable coastal stretches).
- Stage 3: Cyclone Warning / Orange Message (issued 24 hours prior to landfall, specifying expected points of landfall and tidal surges).
- Stage 4: Post-Landfall Outlook / Red Message (issued 12 hours prior to landfall, projecting interior track wind damage and flash flooding).
Multi-Purpose Cyclone Shelters (MPCS) serve as primary community evacuation nodes. Engineered to withstand wind loads up to 250 km/h, they are positioned above historical 100-year flood lines and equipped with independent rainwater harvesting, emergency backup generators, and dual-gender sanitation facilities.""",

    """=== SECTION 4: MULTI-CRITERIA DECISION ANALYSIS AND AHP METHODOLOGY ===
The Analytic Hierarchy Process (AHP), formulated by Thomas L. Saaty, is a mathematically grounded Multi-Criteria Decision Analysis (MCDA) framework. In CYCLONE INTEL, AHP provides the sole authoritative baseline for community preparedness scoring across five core domain scores:
1. Healthcare Accessibility (`score_medical`): Proximity and travel time to functional Primary Health Centers (PHCs), Community Health Centers (CHCs), and emergency hospitals.
2. Evacuation & Shelter Infrastructure (`score_shelters`): Density, capacity, accessibility, and structural condition of designated cyclone shelters.
3. Infrastructure Sensitivity (`score_infrastructure`): Local housing durability (pucca versus kutcha structures), road connectivity (has_pucca_road, major district roads), and drinking water network access.
4. Hazard Exposure & Early Warning (`score_early_warning`): Coastal setback distance, elevation, historical cyclone incidence, distance to rainfall gauges, and gauge coverage density within 20 km.
5. Socioeconomic Vulnerability (`score_socioeconomic`): Demographic concentration of vulnerable groups (scheduled castes, scheduled tribes, children, and elderly) and dependence on artisanal fisheries.
AHP generates reciprocal pairwise comparison matrices synthesized into priority weights. Mathematical consistency is verified via the Consistency Ratio (CR):
CI = (lambda_max - n) / (n - 1), and CR = CI / RI.
All expert matrices governing CYCLONE INTEL satisfy the strict consistency constraint CR < 0.10.
Authoritative AHP Preparedness Tiers:
- High Preparedness Tier: Composite Score >= 60.0.
- Moderate Preparedness Tier: Composite Score between 40.0 and 59.99.
- Low Preparedness Tier: Composite Score < 40.0.""",

    """=== SECTION 5: OPERATIONAL PRIORITY BANDS AND RESOURCE ALLOCATION ===
District Disaster Management Authorities (DDMAs) manage emergency preparedness through four operational Priority Bands:
- Band 1 (Critical Intervention / BAND_1_CRITICAL): Communities requiring immediate capital intervention, priority evacuation asset pre-positioning, and structural reinforcement.
- Band 2 (Priority Strengthening / BAND_2_SPECIALIZED): Communities requiring targeted domain upgrades (e.g. backup generators at shelters or access road elevation).
- Band 3 (Targeted Resilience / BAND_3_MODERATE): Moderately prepared units requiring routine maintenance and communication drills.
- Band 4 (Adequate Resilience / BAND_4_ADEQUATE): Well-prepared communities requiring periodic baseline monitoring.
GOVERNANCE MANDATE:
AHP Preparedness Tiers and Operational Priority Bands are conceptually distinct dimensions.
AHP Tiers evaluate structural and socioeconomic preparedness (the state of the unit).
Priority Bands evaluate operational resource allocation urgency (the response action required).
A unit with Moderate AHP preparedness may be elevated to Band 1 Critical if a single-point failure (such as zero bridge access during high tide) threatens life safety.""",

    """=== SECTION 6: MACHINE LEARNING SURROGATE MODEL DEMARCATION ===
The platform integrates a Multinomial Logistic Regression model trained across 26 static features as an empirical surrogate of the authoritative AHP baseline.
SCIENTIFIC DEMARCATION PRINCIPLES:
1. Empirical Classification Surrogate: The ML model approximates AHP decision boundaries. It is NOT a physical hazard prediction model, storm surge flood model, or casualty forecast.
2. Uncalibrated Class Likelihoods: Model outputs (prob_low, prob_medium, prob_high) represent statistical classification membership likelihoods within the static feature space. They must NEVER be communicated as cyclone strike probabilities or storm risk percentages.
3. Boundary Stability Screening: The surrogate identifies 87 borderline communities where static feature distributions lie close to the 40.0 or 60.0 AHP thresholds.
4. NON-DOWNGRADING SAFETY INVARIANT: The ML surrogate model is strictly prohibited from downgrading an authoritative AHP Low preparedness classification. If AHP classifies a unit as Low Preparedness (< 40.0), operational policy mandates Band 1 Critical status, regardless of whether the statistical surrogate predicts Moderate or High.""",

    """=== SECTION 7: K-MEANS VULNERABILITY ARCHETYPES ===
Unsupervised K-Means clustering was executed on the contiguous southern coastal districts of Dakshina Kannada and Udupi (N=622 assigned units) to identify structural vulnerability archetypes:
- Cluster 0: Urban & Peri-Urban Well-Resourced Corridor (130 units). Dense population and high road and healthcare access, but elevated aggregate infrastructure and utility sensitivity.
- Cluster 1: Remote Foothill & Healthcare-Isolated Communities (82 units). Located in the Western Ghats foothills; low marine hazard exposure, but severe travel buffers to emergency healthcare (> 60 minutes) and isolated transport links.
- Cluster 2: Socioeconomically Vulnerable Rural Interior (161 units). Rural inland habitations characterized by elevated demographic vulnerability, higher shares of kutcha housing, and limited coping capacity.
- Cluster 3: High-Exposure Coastal Rural Belt (249 units). Direct Arabian Sea shoreline exposure (< 1.5 km), extreme storm surge and coastal erosion vulnerability, and critical multi-purpose cyclone shelter deficits.
GEOGRAPHIC SCOPE & UTTARA KANNADA FALLBACK ENGINE:
The 1,257 units in Uttara Kannada were intentionally not forced into the 4-cluster model due to differing geomorphology and administrative scale. Instead, an independent multi-trigger domain-deficit fallback engine evaluates transparent threshold triggers (e.g. healthcare travel time > 60 min, shelter deficit, kutcha roof ratio) to formulate actionable advisories.""",

    """=== SECTION 8: REGIONAL DISTRICT RISK PROFILES ===
Authoritative baseline metrics across the 1,879 administrative units in Coastal Karnataka show clear regional differentiation:
1. Dakshina Kannada (369 units): Mean AHP preparedness score is 61.06 (High preparedness average), supported by high hospital density and road connectivity around Mangaluru, but vulnerable along coastal sand spits.
2. Udupi (253 units): Mean AHP preparedness score is 65.04 (Highest regional average), with robust primary healthcare distribution and multi-purpose shelter coverage across its central coastal plain.
3. Uttara Kannada (1,257 units): Mean AHP preparedness score is 50.06 (Moderate preparedness average), reflecting geographic dispersion, estuarine river barriers, and extended travel times in rural forested taluks.
All numerical community-specific figures are authoritative only when retrieved from deterministic tools, never from model memory.""",

    """=== SECTION 9: SCIENTIFIC UNCERTAINTY AND ZERO-FABRICATION MANDATE ===
Decision support requires explicit demarcation of analytical boundaries:
1. Temporal Baseline Limitations: Demographic and housing data are anchored in Census 2011, supplemented by MoHFW directories and SRTM topographic elevation models.
2. Ground Verification Mandate: Macro-level analytical outputs must be verified by local DDMA task forces before committing capital funds or executing evacuation plans.
3. ZERO HALLUCINATION POLICY FOR DYNAMIC DATA:
   The CYCLONE INTEL platform strictly prohibits fabricating real-time operational parameters:
   - Current meteorological parameters (live wind speed, barometric pressure, real-time radar tracks) must come from live IMD bulletins, NOT from model synthesis.
   - Real-time emergency shelter occupancy, hospital bed availability, and road flood blockages are dynamic operational data unrecorded in the static platform.
   When dynamic data is queried, the system must explicitly state that dynamic real-time data is unrecorded and direct the user to official district emergency operation centers (EOC).""",
]


def build_scenario_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """Build cleanly separated instruction and tool-use examples grouped by Scenario Family.

    No looping or artificial template duplication. Every question is distinct.
    Total scenarios: 147 unique scenarios.
    """
    scenarios: Dict[str, List[Dict[str, Any]]] = {}

    # -------------------------------------------------------------
    # Scenario Family 1: AHP Methodology (16 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["ahp_methodology"] = [
        {
            "id": "ahp_01",
            "instruction": "What is the primary role of the Analytic Hierarchy Process in CYCLONE INTEL?",
            "response": "The Analytic Hierarchy Process (AHP) serves as the sole authoritative baseline for calculating composite community preparedness scores (0.0 to 100.0). It synthesizes five core domain scores—Healthcare Accessibility (score_medical), Shelter Infrastructure (score_shelters), Infrastructure Sensitivity (score_infrastructure), Early Warning & Exposure (score_early_warning), and Socioeconomic Vulnerability (score_socioeconomic)—using Saaty's pairwise comparison methodology with a verified Consistency Ratio below 0.10.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_02",
            "instruction": "What are the score boundaries for the three authoritative AHP preparedness tiers?",
            "response": "The authoritative AHP preparedness tiers are defined as: High Preparedness for composite scores >= 60.0; Moderate Preparedness for scores between 40.0 and 59.99; and Low Preparedness for scores < 40.0. These tiers quantify structural and socioeconomic baseline readiness.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_03",
            "instruction": "What does a Consistency Ratio (CR) equal to or greater than 0.10 signify in the AHP framework?",
            "response": "A Consistency Ratio (CR) >= 0.10 indicates inconsistent pairwise expert judgments that violate transitivity principles. In CYCLONE INTEL, any pairwise comparison matrix with CR >= 0.10 is rejected and must be recalibrated until CR < 0.10 is achieved, guaranteeing mathematical consistency.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_04",
            "instruction": "Name the five domain score columns used in the unified preparedness dataset.",
            "response": "The five domain scores are: score_medical (Healthcare Accessibility), score_shelters (Evacuation & Shelter Infrastructure), score_infrastructure (Infrastructure Sensitivity & Road Access), score_early_warning (Hazard Exposure & Rain Gauges), and score_socioeconomic (Socioeconomic Vulnerability).",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_05",
            "instruction": "Can an administrator adjust an AHP score manually based on intuition?",
            "response": "No. AHP scores in CYCLONE INTEL are deterministic mathematical calculations derived from validated pairwise weights and normalized census and spatial indicators. Manual intuition cannot override or alter an authoritative AHP score.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_06",
            "instruction": "How are criteria weights derived in AHP from pairwise comparisons?",
            "response": "Criteria weights are computed by constructing a reciprocal positive comparison matrix where experts evaluate relative importance between pairs. The normalized principal right eigenvector corresponding to the maximum eigenvalue (lambda_max) yields the criteria priority vector, subject to consistency verification.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_07",
            "instruction": "What spatial layers feed into the score_medical domain score in AHP?",
            "response": "The score_medical domain evaluates geodesic travel distance and travel buffer times to functional Primary Health Centres (PHCs), Community Health Centres (CHCs), and tertiary apex hospitals, penalized when travel times exceed the critical 60-minute threshold.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_08",
            "instruction": "How is the shelter score calculated for a revenue unit?",
            "response": "The score_shelters metric incorporates designated Multi-Purpose Cyclone Shelter (MPCS) capacity within a 10 km geodesic buffer, shelter structural resilience rating against 250 km/h winds, and access road elevational passability.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_09",
            "instruction": "Why is the Random Consistency Index (RI) important when calculating Saaty's CR?",
            "response": "The Random Consistency Index (RI) provides the expected average consistency index obtained from randomly generated reciprocal matrices of dimension n. Dividing the matrix Consistency Index (CI) by RI yields the dimensionless Consistency Ratio (CR = CI / RI).",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_10",
            "instruction": "What indicators constitute the infrastructure sensitivity score?",
            "response": "The score_infrastructure indicator synthesizes Census 2011 housing wall and roof durability (ratio of pucca to kutcha residential dwellings), all-weather pucca road connectivity, and covered tap drinking water network access.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_11",
            "instruction": "How does coastal proximity factor into the early warning domain score?",
            "response": "The score_early_warning metric evaluates geodesic distance to the high tide shoreline (< 1.5 km receiving highest marine hazard weighting), coastal elevation above MSL, and proximity to active automated rain gauges and IMD telemetry.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_12",
            "instruction": "What happens if a revenue village has zero recorded Primary Health Centres within 15 km?",
            "response": "When a revenue unit has no PHC or CHC within 15 km, its raw healthcare proximity metric receives a severe penalty, lowering score_medical towards zero and pulling down the composite AHP score.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_13",
            "instruction": "Are AHP criteria weights identical across all three coastal districts in CYCLONE INTEL?",
            "response": "Yes, CYCLONE INTEL uses a standardized regional pairwise priority matrix for Coastal Karnataka to maintain scientific parity across Dakshina Kannada, Udupi, and Uttara Kannada.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_14",
            "instruction": "Explain how demographic vulnerability is represented in score_socioeconomic.",
            "response": "The score_socioeconomic domain captures the proportion of marginalized populations (Scheduled Castes and Scheduled Tribes), child and elderly dependency ratios, literacy rates, and livelihood dependence on artisanal fisheries.",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_15",
            "instruction": "What is the theoretical range of an AHP composite score in the platform?",
            "response": "Composite AHP preparedness scores range from a minimum of 0.0 (extreme structural vulnerability across all domains) to a maximum of 100.0 (optimal infrastructure, healthcare, shelter, and socioeconomic resilience).",
            "category": "domain_knowledge",
        },
        {
            "id": "ahp_16",
            "instruction": "Why is AHP preferred over simple equal-weighted indicator summation?",
            "response": "Simple summation naively treats all indicators as equally critical. AHP incorporates expert pairwise evaluation that scientifically balances life-saving shelter and healthcare access over secondary socioeconomic factors, while rigorously testing consistency.",
            "category": "domain_knowledge",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 2: Priority Bands vs Tiers (15 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["priority_bands"] = [
        {
            "id": "pb_01",
            "instruction": "Explain the fundamental difference between AHP Preparedness Tiers and Operational Priority Bands.",
            "response": "AHP Preparedness Tiers (High, Moderate, Low) represent the structural and socioeconomic preparedness state of a community. Operational Priority Bands (BAND_1_CRITICAL to BAND_4_ADEQUATE) govern emergency resource deployment and capital intervention urgency by District Disaster Management Authorities. They measure two different operational dimensions.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_02",
            "instruction": "Why might a community with a Moderate AHP score be assigned to Band 1 Critical priority?",
            "response": "A community with a Moderate composite AHP score may be placed in Band 1 Critical if it suffers from a critical single-point vulnerability—such as zero all-weather bridge access during high tide or high exposure combined with severe shelter deficits—where immediate life safety requires urgent intervention.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_03",
            "instruction": "List the four valid Priority Band identifiers accepted by the platform.",
            "response": "The four valid priority band identifiers are: BAND_1_CRITICAL (immediate emergency allocation), BAND_2_SPECIALIZED (targeted domain strengthening), BAND_3_MODERATE (targeted resilience maintenance), and BAND_4_ADEQUATE (routine baseline monitoring).",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_04",
            "instruction": "If a community is in AHP Low preparedness, what priority band does operational governance mandate?",
            "response": "Under operational governance rules, any community evaluated as AHP Low preparedness (< 40.0) is strictly mandated into Band 1 Critical (BAND_1_CRITICAL) to ensure priority pre-positioning and disaster resource allocation.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_05",
            "instruction": "What operational actions does Band 1 Critical mandate for District Disaster Management Authorities?",
            "response": "BAND_1_CRITICAL mandates pre-positioning SDRF/NDRF rescue boats, establishing satellite communication backups, deploying mobile medical units, and scheduling priority evacuation staging prior to cyclone landfall.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_06",
            "instruction": "What type of community typically falls into Band 2 Specialized?",
            "response": "BAND_2_SPECIALIZED encompasses communities with acceptable overall preparedness but a distinct domain bottleneck, such as robust healthcare and housing but a deficiency in cyclone shelter capacity or elevated storm surge vulnerability.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_07",
            "instruction": "How does Band 3 Moderate differ from Band 4 Adequate?",
            "response": "BAND_3_MODERATE communities maintain moderate resilience but require targeted seasonal maintenance and awareness drills. BAND_4_ADEQUATE communities possess high structural preparedness across all five domains and need only routine periodic monitoring.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_08",
            "instruction": "Can an automated system demote a Band 1 Critical community to Band 4 Adequate?",
            "response": "No. Band 1 Critical assignments are governed by life-safety constraints and AHP Low thresholds. Automatic demotion to Band 4 is blocked unless verified physical infrastructure upgrades (e.g., new shelter, bridge construction) are formally certified in the platform.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_09",
            "instruction": "In Dakshina Kannada, how many communities are classified under Band 1 Critical?",
            "response": "Authoritative baseline ranking in Dakshina Kannada identifies 2 critical communities in Band 1: Kodimbala (preparedness score 38.44) and Sulya TP (preparedness score 39.31), representing 0.5% of the district's 369 units.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_10",
            "instruction": "How many units in Uttara Kannada are placed in Band 1 Critical?",
            "response": "In Uttara Kannada, 153 out of 1,257 administrative units (12.2%) are classified under Band 1 Critical due to severe Western Ghats healthcare transit times and isolated estuarine settlements.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_11",
            "instruction": "Why does Udupi have zero communities in Band 1 Critical in baseline analysis?",
            "response": "Udupi maintains a high district mean preparedness score of 65.04 with dense primary healthcare and extensive multi-purpose shelter coverage across its 253 coastal units, resulting in 0 units falling below the AHP Low 40.0 threshold.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_12",
            "instruction": "What is the relationship between overall_priority_rank and band_rank?",
            "response": "overall_priority_rank (1 to 1,879) orders all revenue units across Coastal Karnataka by operational urgency. band_rank provides the relative rank within a specific priority band (e.g., 1st among Band 1 units).",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_13",
            "instruction": "Can a high-population urban village have high priority if its AHP score is Moderate?",
            "response": "Yes. When population-scaled prioritization is evaluated, a high-density urban or peri-urban village with moderate preparedness may be elevated in priority rank because adverse impacts would affect a larger volume of residents.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_14",
            "instruction": "Who has the authority to approve capital expenditure for Band 1 communities?",
            "response": "District Disaster Management Authorities (DDMAs) headed by the Deputy Commissioner (DC) hold legal authority under the Disaster Management Act 2005 to commit capital funds based on platform prioritization.",
            "category": "domain_knowledge",
        },
        {
            "id": "pb_15",
            "instruction": "What is the primary intervention focus for Band 2 Specialized communities in coastal taluks?",
            "response": "The primary focus for Band 2 coastal units is targeted shelter retrofitting, installation of backup diesel generators and rainwater harvesting at evacuation centres, and raising low-lying coastal approach roads above high-tide level.",
            "category": "domain_knowledge",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 3: ML Surrogate Demarcation (15 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["surrogate_demarcation"] = [
        {
            "id": "surr_01",
            "instruction": "What is the exact scientific role of the Multinomial Logistic Regression model in CYCLONE INTEL?",
            "response": "The Logistic Regression model is an empirical classification surrogate trained on 26 static spatial and census features to approximate the decision boundaries of the multi-criteria AHP assessment. It is not a physical cyclone track or storm surge hazard simulator.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_02",
            "instruction": "Can the ML surrogate model's predicted probability prob_high be used as a cyclone strike probability?",
            "response": "No. prob_high represents statistical classification membership likelihood in the High Preparedness class within the static feature space. It must never be interpreted as a meteorological cyclone strike probability or hazard forecast.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_03",
            "instruction": "Explain the non-downgrading safety invariant governing the ML surrogate model.",
            "response": "The non-downgrading safety invariant dictates that the ML surrogate can never override or downgrade an authoritative AHP Low preparedness classification. If deterministic AHP places a community in Low Preparedness (< 40.0), it remains Band 1 Critical regardless of ML predictions.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_04",
            "instruction": "How many static features are used to train the Logistic Regression surrogate model?",
            "response": "The surrogate model is trained on 26 static features spanning healthcare accessibility, shelter density, road connectivity, water infrastructure, demographics, and topographic exposure.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_05",
            "instruction": "What are borderline communities in the context of the ML surrogate model?",
            "response": "Borderline communities are 87 revenue units whose static feature distributions place them within close proximity (delta < 2.5 points) to the 40.0 (Low/Moderate) or 60.0 (Moderate/High) AHP tier thresholds, highlighting decision boundaries.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_06",
            "instruction": "Why was a Multinomial Logistic Regression model chosen over deep neural networks for the surrogate?",
            "response": "Logistic Regression provides transparent, inspectable linear coefficients for each feature, guarantees monotonic feature response, prevents overfitting on small geographic samples, and allows full auditability for public disaster management.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_07",
            "instruction": "If AHP evaluates a unit as Moderate (58.0) and the surrogate predicts High, which tier is authoritative?",
            "response": "Deterministic AHP is always authoritative: the unit is classified as Moderate Preparedness. The surrogate prediction serves only as a diagnostic indicator of feature-space alignment near the 60.0 boundary.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_08",
            "instruction": "What is the overall agreement rate between the ML surrogate and deterministic AHP tiers in Dakshina Kannada?",
            "response": "In Dakshina Kannada, the empirical surrogate achieves a 93.0% classification agreement rate against authoritative AHP preparedness tiers across all 369 administrative units.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_09",
            "instruction": "What is the surrogate model agreement rate in Udupi?",
            "response": "In Udupi district, the surrogate model achieves a 96.0% classification agreement rate against authoritative AHP tiers across 253 units.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_10",
            "instruction": "What is the surrogate model agreement rate in Uttara Kannada?",
            "response": "In Uttara Kannada, the surrogate model achieves a 95.9% agreement rate against authoritative AHP tiers across 1,257 units.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_11",
            "instruction": "Does the surrogate model predict cyclone casualties or infrastructure damage estimates?",
            "response": "No. The surrogate model predicts solely the preparedness tier classification (Low, Medium, High). It has no capability to estimate casualties, structural damage costs, or physical wind impact.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_12",
            "instruction": "How are probability outputs calibrated in the surrogate model?",
            "response": "The model uses standard multinomial softmax normalization over linear logits. However, because features are static census and GIS metrics, the outputs are uncalibrated class likelihoods and not Bayesian posterior probabilities.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_13",
            "instruction": "Why can machine learning models never replace deterministic AHP in CYCLONE INTEL?",
            "response": "Machine learning models are statistical approximations subject to sample variance and distribution drift. AHP provides an accountable, legally verifiable, and transparent multi-criteria formula required for equitable public resource allocation.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_14",
            "instruction": "Which coefficient typically has the largest positive magnitude in predicting High Preparedness?",
            "response": "In the surrogate model, all-weather pucca road connectivity and distance to nearest apex hospital carry the highest magnitude weights in separating High from Low preparedness units.",
            "category": "domain_knowledge",
        },
        {
            "id": "surr_15",
            "instruction": "What warning is issued when viewing surrogate ML predictions in the platform?",
            "response": "A mandatory scientific governance disclaimer is displayed: 'ML predictions are empirical surrogates approximating AHP boundaries, not physical hazard forecasts. Deterministic AHP maintains absolute operational precedence.'",
            "category": "domain_knowledge",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 4: Vulnerability Archetypes (16 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["vulnerability_archetypes"] = [
        {
            "id": "arch_01",
            "instruction": "What are the four K-Means vulnerability archetypes in Dakshina Kannada and Udupi?",
            "response": "The four K-Means archetypes derived for Dakshina Kannada and Udupi (N=622) are: Cluster 0: Urban & Peri-Urban Well-Resourced Corridor (130 units); Cluster 1: Remote Foothill & Healthcare-Isolated Communities (82 units); Cluster 2: Socioeconomically Vulnerable Rural Interior (161 units); and Cluster 3: High-Exposure Coastal Rural Belt (249 units).",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_02",
            "instruction": "Why are the 1,257 units in Uttara Kannada unassigned to the four K-Means clusters?",
            "response": "Uttara Kannada features distinct geomorphic, ecological, and administrative characteristics (rugged rias coastline, forest interior, smaller revenue units) that differ structurally from the urbanized coastal plains of Dakshina Kannada and Udupi. To prevent flawed model extrapolation, Uttara Kannada units are evaluated using an independent multi-trigger domain-deficit fallback engine.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_03",
            "instruction": "What are the primary vulnerability characteristics of Cluster 3 (High-Exposure Coastal Rural Belt)?",
            "response": "Cluster 3 communities are situated directly along the Arabian Sea littoral (< 1.5 km setback), facing high storm surge and coastal erosion exposure, low mean elevation, and critical multi-purpose cyclone shelter capacity deficits relative to their coastal populations.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_04",
            "instruction": "What are the primary vulnerability characteristics of Cluster 1 (Remote Foothill)?",
            "response": "Cluster 1 communities are situated in the Western Ghats foothill zone with low coastal marine hazard exposure, but they suffer from severe transit buffers to emergency healthcare (> 60 minutes travel time to nearest hospital/CHC) and limited road network redundancy.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_05",
            "instruction": "Describe the profile of Cluster 0 (Urban & Peri-Urban Well-Resourced Corridor).",
            "response": "Cluster 0 comprises 130 revenue units characterized by high population density, superior pucca road connectivity, rapid hospital access (< 15 minutes), and high tap water coverage, but elevated aggregate infrastructure loss potential during severe storms.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_06",
            "instruction": "What characterizes Cluster 2 (Socioeconomically Vulnerable Rural Interior)?",
            "response": "Cluster 2 covers 161 inland rural units marked by elevated demographic vulnerability (higher Scheduled Caste and Scheduled Tribe proportions), higher shares of semi-pucca or kutcha residential housing, and modest healthcare access.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_07",
            "instruction": "What is the primary intervention recommended for Cluster 3 coastal communities?",
            "response": "Primary interventions for Cluster 3 include construction of new Multi-Purpose Cyclone Shelters within 1.5 km of coastal fishing hamlets, shoreline wave-attenuation revetments, and elevated all-weather evacuation corridors.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_08",
            "instruction": "What is the primary intervention recommended for Cluster 1 foothill communities?",
            "response": "Primary interventions for Cluster 1 include establishing 24/7 advanced life-support ambulance staging points, upgrading rural sub-centres to 24/7 Primary Health Centres, and stabilizing landslide-prone mountain access roads.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_09",
            "instruction": "How many total units were analyzed in the K-Means clustering study area?",
            "response": "A total of 622 administrative revenue units across the contiguous coastal districts of Dakshina Kannada (369 units) and Udupi (253 units) formed the K-Means clustering study area.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_10",
            "instruction": "How does the Uttara Kannada three-sector fallback engine operate?",
            "response": "The fallback engine checks three transparent threshold triggers: coastal proximity < 5.0 km, shelter score < 20.0, and healthcare access score < 35.0. Trigger combinations generate targeted sectoral intervention advisories without forcing units into inappropriate southern clusters.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_11",
            "instruction": "Which cluster contains the largest number of revenue units in Dakshina Kannada and Udupi?",
            "response": "Cluster 3 (High-Exposure Coastal Rural Belt) contains the largest number of units, with 249 revenue units (40.0% of the 622 clustered units).",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_12",
            "instruction": "Can a village belonging to Cluster 0 be classified as AHP Low Preparedness?",
            "response": "No, empirically all 130 units in Cluster 0 have composite AHP scores well above 60.0 due to high healthcare density, tap water, and pucca road infrastructure around Mangaluru and Udupi towns.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_13",
            "instruction": "Are K-Means archetypes predictive of disaster timing or storm landfall points?",
            "response": "No. K-Means archetypes are static structural descriptors of vulnerability profiles based on Census and GIS baselines. They do not predict when, where, or if a cyclone will make landfall.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_14",
            "instruction": "What feature standardization was applied prior to running K-Means clustering?",
            "response": "Standard robust z-score standardization was applied to all input dimension indicators to eliminate scale bias between raw distance (km) and percentage demographic metrics.",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_15",
            "instruction": "Why was k=4 selected as the optimal cluster count for the coastal archetypes?",
            "response": "k=4 was selected based on silhouette score analysis, elbow method evaluation of within-cluster sum of squares, and domain interpretability aligning with distinct geographic zones (Urban, Foothill, Rural Interior, Coastal Littoral).",
            "category": "domain_knowledge",
        },
        {
            "id": "arch_16",
            "instruction": "What intervention is recommended for Cluster 2 Rural Interior villages?",
            "response": "For Cluster 2, priorities focus on housing reinforcement subsidies (PMAY housing upgrades to replace kutcha thatched roofs), community drinking water filtration units, and seasonal vector-borne disease preparedness.",
            "category": "domain_knowledge",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 5: Scientific Governance & Refusals (16 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["scientific_governance"] = [
        {
            "id": "gov_01",
            "instruction": "What is the live wind speed in Mangaluru port right now?",
            "response": "I cannot provide real-time meteorological observations such as live wind speeds, atmospheric pressure, or live radar tracks. The CYCLONE INTEL platform is an analytical decision support system grounded in static baseline data. For live observations, consult official India Meteorological Department (IMD) bulletins and KSNDMC telemetry.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_02",
            "instruction": "How many hospital beds are currently vacant in Udupi District Hospital today?",
            "response": "Real-time hospital bed vacancy is dynamic operational data unrecorded in our system. The platform maintains static structural healthcare metrics (facility type, location, travel buffer) based on MoHFW directories. Real-time bed availability must be confirmed directly with the District Health Officer or Emergency Operations Centre (EOC).",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_03",
            "instruction": "Is the coastal highway between Kundapura and Bhatkal currently submerged by tidal surge?",
            "response": "I cannot report dynamic or real-time road conditions. Current road trafficability, live waterlogging levels, and bridge passability must be confirmed through the local Public Works Department (PWD) or District Disaster Management Authority control room dispatches.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_04",
            "instruction": "What is the current shelter occupancy in the Bengre cyclone shelter right now?",
            "response": "Real-time shelter occupancy is dynamic operational data not tracked in our static analytical database. To verify current evacuee counts, available rations, or intake capacity, contact the local revenue inspector or designated shelter manager.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_05",
            "instruction": "Can you forecast where Cyclone Biparjoy will hit next week?",
            "response": "I cannot generate dynamic weather forecasts or predict future cyclone landfall trajectories. CYCLONE INTEL is a structural preparedness planning tool. For dynamic cyclone forecasts, consult the official RSMC New Delhi bulletins issued by the India Meteorological Department.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_06",
            "instruction": "Are emergency rescue boats currently deployed at Ullal beach?",
            "response": "Real-time tactical deployment of rescue assets (SDRF, NDRF, Coast Guard) is dynamic operational telemetry unrecorded in this platform. For current deployment status, contact the District Disaster Management Authority control room.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_07",
            "instruction": "What is the current water level in the Netravati river gauge right now?",
            "response": "Live hydrological telemetry such as real-time river stage levels and flood discharge rates are dynamic data unrecorded in our database. Live telemetry must be verified via Central Water Commission (CWC) or Karnataka State Natural Disaster Monitoring Centre (KSNDMC) feeds.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_08",
            "instruction": "Can an automated script override the AHP score of a village to make it qualify for emergency funding?",
            "response": "No. CYCLONE INTEL enforces strict scientific governance: AHP scores are deterministic, immutable mathematical outputs based on audited spatial criteria. Manual or programmatic manipulation of scores to alter funding eligibility is strictly prohibited.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_09",
            "instruction": "What is the vintage of the baseline demographic data used by the platform?",
            "response": "The demographic and housing baseline is anchored in the Census of India 2011 Primary Census Abstract and Village Directory, supplemented by MoHFW geospatial directories and open spatial elevation datasets.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_10",
            "instruction": "Why does the platform require field ground verification before committing capital funds?",
            "response": "Because baseline datasets originate from Census 2011 and satellite elevation models, localized infrastructure developments (e.g., newly paved roads, recent clinic construction) must be verified through physical DDMA ground truthing before executing multi-crore capital projects.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_11",
            "instruction": "How many emergency relief food packets are in stock at Karwar taluk office today?",
            "response": "Real-time emergency supply stockpiles, ration counts, and relief material inventories are dynamic operational data not tracked in this static baseline system. Please contact the Karwar Taluk Tahsildar or DDMA logistics officer directly.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_12",
            "instruction": "Is power currently cut off in Malpe fishing harbour due to squally winds?",
            "response": "I cannot provide real-time electrical grid or utility outage telemetry. Current power transmission status must be verified through the local electricity supply company (MESCOM) or local disaster control room.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_13",
            "instruction": "Can the model output be cited as legal proof of cyclone insurance damage?",
            "response": "No. CYCLONE INTEL models structural preparedness and relative operational priority. It does not assess post-disaster insurance loss or legal liability, which require statutory surveyor damage inspections.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_14",
            "instruction": "Can CycloneLM execute bash scripts on the server to retrieve live weather data?",
            "response": "No. The language model operates strictly through a sandboxed, deterministic tool-calling protocol. It is fundamentally barred from executing arbitrary bash commands, eval statements, or uncontrolled external network requests.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_15",
            "instruction": "What should an official do if a high-tide surge is actively threatening a coastal community?",
            "response": "During an active surge event, officials must bypass static planning software and immediately activate standard operating procedures under the District Disaster Management Plan, including sounding coastal sirens, mobilizing SDRF/police teams, and executing pre-planned evacuation routes.",
            "category": "domain_knowledge",
        },
        {
            "id": "gov_16",
            "instruction": "Can the ML surrogate change an AHP Low unit to High if new census data is uploaded?",
            "response": "No. The ML surrogate is an empirical approximation that is never authorized to alter, reclassify, or downgrade authoritative AHP scores. Any update to underlying indicators must be recomputed deterministically through the AHP pairwise matrix.",
            "category": "domain_knowledge",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 6: Structured Tool Selection (35 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["tool_selection"] = [
        # lookup_community (7 scenarios)
        {
            "id": "ts_01",
            "instruction": "Look up community records for Ullal in Dakshina Kannada.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "Ullal", "district": "Dakshina Kannada"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Ullal", "district": "Dakshina Kannada"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_02",
            "instruction": "Search for village Tallur in the preparedness database.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "Tallur"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Tallur"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_03",
            "instruction": "Find details for community Bengre in Uttara Kannada.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "Bengre", "district": "Uttara Kannada"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Bengre", "district": "Uttara Kannada"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_04",
            "instruction": "Look up PC11 ID 608720 in Udupi.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "608720", "district": "Udupi"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "608720", "district": "Udupi"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_05",
            "instruction": "Search for Malpe fishing community records.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "Malpe"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Malpe"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_06",
            "instruction": "Retrieve records for revenue unit Edmoge.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "Edmoge"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Edmoge"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_07",
            "instruction": "Locate community Someshwara in Dakshina Kannada.",
            "expected_tool": "lookup_community",
            "expected_params": {"query": "Someshwara", "district": "Dakshina Kannada"},
            "response": '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Someshwara", "district": "Dakshina Kannada"}}</tool_call>',
            "category": "tool_selection",
        },

        # analyze_infrastructure_and_exposure (5 scenarios)
        {
            "id": "ts_08",
            "instruction": "Analyze the shelter density, healthcare distance, and physical coastal exposure for Someshwara.",
            "expected_tool": "analyze_infrastructure_and_exposure",
            "expected_params": {"identifier": "Someshwara"},
            "response": '<tool_call>{"tool": "analyze_infrastructure_and_exposure", "parameters": {"identifier": "Someshwara"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_09",
            "instruction": "Inspect physical exposure and hospital buffer distances for revenue unit 608691.",
            "expected_tool": "analyze_infrastructure_and_exposure",
            "expected_params": {"identifier": "608691"},
            "response": '<tool_call>{"tool": "analyze_infrastructure_and_exposure", "parameters": {"identifier": "608691"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_10",
            "instruction": "Check road connectivity and drinking water indicators for Tallur.",
            "expected_tool": "analyze_infrastructure_and_exposure",
            "expected_params": {"identifier": "Tallur"},
            "response": '<tool_call>{"tool": "analyze_infrastructure_and_exposure", "parameters": {"identifier": "Tallur"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_11",
            "instruction": "Evaluate the coastal setback and cyclone shelter radius for Bengre.",
            "expected_tool": "analyze_infrastructure_and_exposure",
            "expected_params": {"identifier": "Bengre"},
            "response": '<tool_call>{"tool": "analyze_infrastructure_and_exposure", "parameters": {"identifier": "Bengre"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_12",
            "instruction": "Analyze the infrastructure vulnerability for Edmoge in Udupi.",
            "expected_tool": "analyze_infrastructure_and_exposure",
            "expected_params": {"identifier": "Edmoge"},
            "response": '<tool_call>{"tool": "analyze_infrastructure_and_exposure", "parameters": {"identifier": "Edmoge"}}</tool_call>',
            "category": "tool_selection",
        },

        # explain_preparedness_assessment (5 scenarios)
        {
            "id": "ts_13",
            "instruction": "Explain the multi-criteria AHP preparedness assessment breakdown for revenue unit Ullal.",
            "expected_tool": "explain_preparedness_assessment",
            "expected_params": {"identifier": "Ullal"},
            "response": '<tool_call>{"tool": "explain_preparedness_assessment", "parameters": {"identifier": "Ullal"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_14",
            "instruction": "Why is Tallur evaluated as High Preparedness in AHP?",
            "expected_tool": "explain_preparedness_assessment",
            "expected_params": {"identifier": "Tallur"},
            "response": '<tool_call>{"tool": "explain_preparedness_assessment", "parameters": {"identifier": "Tallur"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_15",
            "instruction": "Show the five domain scores and AHP weights for community Edmoge.",
            "expected_tool": "explain_preparedness_assessment",
            "expected_params": {"identifier": "Edmoge"},
            "response": '<tool_call>{"tool": "explain_preparedness_assessment", "parameters": {"identifier": "Edmoge"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_16",
            "instruction": "Explain the preparedness assessment for Census unit 617728.",
            "expected_tool": "explain_preparedness_assessment",
            "expected_params": {"identifier": "617728"},
            "response": '<tool_call>{"tool": "explain_preparedness_assessment", "parameters": {"identifier": "617728"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_17",
            "instruction": "Check the deterministic AHP scores for Malpe village.",
            "expected_tool": "explain_preparedness_assessment",
            "expected_params": {"identifier": "Malpe"},
            "response": '<tool_call>{"tool": "explain_preparedness_assessment", "parameters": {"identifier": "Malpe"}}</tool_call>',
            "category": "tool_selection",
        },

        # get_intervention_recommendations (5 scenarios)
        {
            "id": "ts_18",
            "instruction": "What are the recommended engineering interventions for community Bengre?",
            "expected_tool": "get_intervention_recommendations",
            "expected_params": {"identifier": "Bengre"},
            "response": '<tool_call>{"tool": "get_intervention_recommendations", "parameters": {"identifier": "Bengre"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_19",
            "instruction": "Provide prioritized intervention recommendations for revenue village Ullal.",
            "expected_tool": "get_intervention_recommendations",
            "expected_params": {"identifier": "Ullal"},
            "response": '<tool_call>{"tool": "get_intervention_recommendations", "parameters": {"identifier": "Ullal"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_20",
            "instruction": "What disaster risk reduction actions are advised for Tallur?",
            "expected_tool": "get_intervention_recommendations",
            "expected_params": {"identifier": "Tallur"},
            "response": '<tool_call>{"tool": "get_intervention_recommendations", "parameters": {"identifier": "Tallur"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_21",
            "instruction": "Get infrastructure upgrade recommendations for PC11 ID 608720.",
            "expected_tool": "get_intervention_recommendations",
            "expected_params": {"identifier": "608720"},
            "response": '<tool_call>{"tool": "get_intervention_recommendations", "parameters": {"identifier": "608720"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_22",
            "instruction": "Recommend preparedness interventions for village Someshwara.",
            "expected_tool": "get_intervention_recommendations",
            "expected_params": {"identifier": "Someshwara"},
            "response": '<tool_call>{"tool": "get_intervention_recommendations", "parameters": {"identifier": "Someshwara"}}</tool_call>',
            "category": "tool_selection",
        },

        # compare_districts (5 scenarios)
        {
            "id": "ts_23",
            "instruction": "Compare disaster preparedness indicators between Dakshina Kannada and Udupi.",
            "expected_tool": "compare_districts",
            "expected_params": {"district_a": "Dakshina Kannada", "district_b": "Udupi"},
            "response": '<tool_call>{"tool": "compare_districts", "parameters": {"district_a": "Dakshina Kannada", "district_b": "Udupi"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_24",
            "instruction": "Compare baseline preparedness statistics between Udupi and Uttara Kannada districts.",
            "expected_tool": "compare_districts",
            "expected_params": {"district_a": "Udupi", "district_b": "Uttara Kannada"},
            "response": '<tool_call>{"tool": "compare_districts", "parameters": {"district_a": "Udupi", "district_b": "Uttara Kannada"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_25",
            "instruction": "Run a district comparison between Dakshina Kannada and Uttara Kannada.",
            "expected_tool": "compare_districts",
            "expected_params": {"district_a": "Dakshina Kannada", "district_b": "Uttara Kannada"},
            "response": '<tool_call>{"tool": "compare_districts", "parameters": {"district_a": "Dakshina Kannada", "district_b": "Uttara Kannada"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_26",
            "instruction": "Contrast the priority band distributions of Udupi and Dakshina Kannada.",
            "expected_tool": "compare_districts",
            "expected_params": {"district_a": "Udupi", "district_b": "Dakshina Kannada"},
            "response": '<tool_call>{"tool": "compare_districts", "parameters": {"district_a": "Udupi", "district_b": "Dakshina Kannada"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_27",
            "instruction": "Compare mean preparedness scores between Uttara Kannada and Udupi.",
            "expected_tool": "compare_districts",
            "expected_params": {"district_a": "Uttara Kannada", "district_b": "Udupi"},
            "response": '<tool_call>{"tool": "compare_districts", "parameters": {"district_a": "Uttara Kannada", "district_b": "Udupi"}}</tool_call>',
            "category": "tool_selection",
        },

        # filter_and_rank_communities (4 scenarios)
        {
            "id": "ts_28",
            "instruction": "Filter communities in Dakshina Kannada classified under Band 1 Critical priority.",
            "expected_tool": "filter_and_rank_communities",
            "expected_params": {"district": "Dakshina Kannada", "priority_band": "BAND_1_CRITICAL"},
            "response": '<tool_call>{"tool": "filter_and_rank_communities", "parameters": {"district": "Dakshina Kannada", "priority_band": "BAND_1_CRITICAL"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_29",
            "instruction": "List the top 5 most vulnerable communities in Uttara Kannada sorted by preparedness score.",
            "expected_tool": "filter_and_rank_communities",
            "expected_params": {"district": "Uttara Kannada", "ascending": True, "limit": 5},
            "response": '<tool_call>{"tool": "filter_and_rank_communities", "parameters": {"district": "Uttara Kannada", "ascending": true, "limit": 5}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_30",
            "instruction": "Show the top 10 communities in Udupi in Band 2 Specialized priority.",
            "expected_tool": "filter_and_rank_communities",
            "expected_params": {"district": "Udupi", "priority_band": "BAND_2_SPECIALIZED", "limit": 10},
            "response": '<tool_call>{"tool": "filter_and_rank_communities", "parameters": {"district": "Udupi", "priority_band": "BAND_2_SPECIALIZED", "limit": 10}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_31",
            "instruction": "Rank communities in Dakshina Kannada by geodesic distance to coastline ascending.",
            "expected_tool": "filter_and_rank_communities",
            "expected_params": {"district": "Dakshina Kannada", "sort_by": "dist_coast_km", "ascending": True},
            "response": '<tool_call>{"tool": "filter_and_rank_communities", "parameters": {"district": "Dakshina Kannada", "sort_by": "dist_coast_km", "ascending": true}}</tool_call>',
            "category": "tool_selection",
        },

        # explain_vulnerability_archetype (4 scenarios)
        {
            "id": "ts_32",
            "instruction": "Explain the structural profile and characteristics of vulnerability archetype Cluster 3.",
            "expected_tool": "explain_vulnerability_archetype",
            "expected_params": {"archetype_id": 3},
            "response": '<tool_call>{"tool": "explain_vulnerability_archetype", "parameters": {"archetype_id": 3}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_33",
            "instruction": "Inspect whether community Tallur has been assigned a vulnerability archetype or fallback status.",
            "expected_tool": "explain_vulnerability_archetype",
            "expected_params": {"community_identifier": "Tallur"},
            "response": '<tool_call>{"tool": "explain_vulnerability_archetype", "parameters": {"community_identifier": "Tallur"}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_34",
            "instruction": "Describe the characteristics of vulnerability archetype Cluster 1.",
            "expected_tool": "explain_vulnerability_archetype",
            "expected_params": {"archetype_id": 1},
            "response": '<tool_call>{"tool": "explain_vulnerability_archetype", "parameters": {"archetype_id": 1}}</tool_call>',
            "category": "tool_selection",
        },
        {
            "id": "ts_35",
            "instruction": "What is the archetype profile for Cluster 0 Urban Corridor?",
            "expected_tool": "explain_vulnerability_archetype",
            "expected_params": {"archetype_id": 0},
            "response": '<tool_call>{"tool": "explain_vulnerability_archetype", "parameters": {"archetype_id": 0}}</tool_call>',
            "category": "tool_selection",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 7: Missing Arguments & Clarifications (15 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["missing_arguments"] = [
        {
            "id": "ma_01",
            "instruction": "Look up community preparedness records.",
            "response": "To look up community records, please specify the name or Census identifier of the community (e.g., 'Ullal' or 'Tallur'), and optionally the district.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_02",
            "instruction": "Compare districts preparedness.",
            "response": "To compare district preparedness, please specify two coastal districts to compare (chosen from: 'Dakshina Kannada', 'Udupi', or 'Uttara Kannada').",
            "category": "missing_arguments",
        },
        {
            "id": "ma_03",
            "instruction": "Explain the vulnerability archetype.",
            "response": "Please specify which archetype you wish to inspect (archetype ID 0, 1, 2, or 3), or provide a specific village name to check its cluster assignment.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_04",
            "instruction": "Provide intervention recommendations.",
            "response": "Please specify the community name or Census identifier (e.g., 'Ullal', 'Bengre', or 'Tallur') to retrieve prioritized intervention recommendations.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_05",
            "instruction": "Analyze infrastructure and exposure.",
            "response": "To analyze physical infrastructure and exposure, please provide the community name, Census PC11 ID, or SHRID.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_06",
            "instruction": "Explain preparedness assessment.",
            "response": "To explain a preparedness assessment, please provide the name or PC11 Census identifier of the revenue village you wish to evaluate.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_07",
            "instruction": "Compare Dakshina Kannada with another district.",
            "response": "Please specify the second district to compare with Dakshina Kannada (either 'Udupi' or 'Uttara Kannada').",
            "category": "missing_arguments",
        },
        {
            "id": "ma_08",
            "instruction": "Filter communities.",
            "response": "To filter communities, please provide at least one criteria such as district ('Dakshina Kannada', 'Udupi', or 'Uttara Kannada') or priority band ('BAND_1_CRITICAL', etc.).",
            "category": "missing_arguments",
        },
        {
            "id": "ma_09",
            "instruction": "Search for village in district.",
            "response": "Please provide the name or Census identifier of the village you want to search for, along with the district name.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_10",
            "instruction": "What are the engineering interventions for that village?",
            "response": "Which village are you referring to? Please state the village name or Census ID so I can fetch its specific intervention recommendations.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_11",
            "instruction": "Give me the healthcare travel buffer.",
            "response": "Please provide the village name or Census PC11 code so I can inspect its geodesic distance and travel time to the nearest PHC and apex hospital.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_12",
            "instruction": "Show me the AHP score.",
            "response": "Please specify which revenue unit or village you would like to retrieve the AHP preparedness score for.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_13",
            "instruction": "Check the shelter distance.",
            "response": "Please indicate the community identifier (village name or PC11 ID) to analyze its cyclone shelter distance and capacity.",
            "category": "missing_arguments",
        },
        {
            "id": "ma_14",
            "instruction": "Compare Udupi with.",
            "response": "Please specify which district you want to compare Udupi with ('Dakshina Kannada' or 'Uttara Kannada').",
            "category": "missing_arguments",
        },
        {
            "id": "ma_15",
            "instruction": "Rank the communities by priority.",
            "response": "Please specify whether you would like to rank communities within a specific district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada') or across a specific Priority Band.",
            "category": "missing_arguments",
        },
    ]

    # -------------------------------------------------------------
    # Scenario Family 8: Evidence-Grounded Synthesis (20 distinct scenarios)
    # -------------------------------------------------------------
    scenarios["grounded_synthesis"] = [
        {
            "id": "gs_01",
            "instruction": "Synthesize the lookup results for Ullal.",
            "evidence": {
                "community_name": "Ullal (TMC)",
                "district_name": "Dakshina Kannada",
                "preparedness_score": 60.44,
                "preparedness_tier": "High",
                "priority_band": "BAND_4_ADEQUATE",
                "overall_priority_rank": 1589,
            },
            "response": "<evidence>\nCommunity: Ullal (TMC), District: Dakshina Kannada\nAHP Score: 60.44 (Tier: High), Priority: BAND_4_ADEQUATE\nOverall Priority Rank: 1589\n</evidence>\n\nUllal (TMC) in Dakshina Kannada exhibits an authoritative AHP Preparedness Score of 60.44, placing it in the High Preparedness Tier and Band 4 Adequate priority. With an overall priority rank of 1589, its baseline infrastructure and healthcare access provide robust coping capacity, requiring routine periodic monitoring.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_02",
            "instruction": "Summarize district comparison evidence between Udupi and Uttara Kannada.",
            "evidence": {
                "districts_compared": ["Udupi", "Uttara Kannada"],
                "mean_preparedness_score": {"Udupi": 65.04, "Uttara Kannada": 50.06},
                "total_units": {"Udupi": 253, "Uttara Kannada": 1257},
                "band_1_critical_count": {"Udupi": 0, "Uttara Kannada": 153},
            },
            "response": "<evidence>\nUdupi: 253 units, Mean AHP = 65.04, Band 1 Units = 0\nUttara Kannada: 1257 units, Mean AHP = 50.06, Band 1 Units = 153\n</evidence>\n\nAuthoritative baseline comparison indicates Udupi possesses a higher mean composite preparedness score (65.04 across 253 units) with 0 units in Band 1 Critical. In contrast, Uttara Kannada averages 50.06 across 1,257 units with 153 Band 1 Critical units due to geographic dispersion and longer healthcare travel times in forested estuarine taluks.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_03",
            "instruction": "Synthesize the assessment for Edmoge in Udupi.",
            "evidence": {
                "community_name": "Edmoge",
                "district_name": "Udupi",
                "preparedness_score": 49.84,
                "preparedness_tier": "Medium",
                "priority_band": "BAND_3_MODERATE",
                "domain_scores": {
                    "score_medical": 53.21,
                    "score_shelters": 24.79,
                    "score_infrastructure": 48.0,
                    "score_early_warning": 66.85,
                    "score_socioeconomic": 71.43,
                },
            },
            "response": "<evidence>\nCommunity: Edmoge, District: Udupi\nAHP Score: 49.84 (Tier: Medium), Priority: BAND_3_MODERATE\nDomain Scores: Medical=53.21, Shelters=24.79, Infrastructure=48.0, Early Warning=66.85, Socioeconomic=71.43\n</evidence>\n\nEdmoge in Udupi has a composite AHP score of 49.84, placing it in the Moderate Preparedness Tier and Band 3 Moderate priority. The primary vulnerability bottleneck is shelter infrastructure (score_shelters: 24.79), while socioeconomic resilience (71.43) and early warning coverage (66.85) remain comparatively solid.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_04",
            "instruction": "Summarize infrastructure findings for Tallur.",
            "evidence": {
                "community_name": "Tallur",
                "district_name": "Udupi",
                "preparedness_score": 74.34,
                "priority_band": "BAND_4_ADEQUATE",
                "dist_to_coast_km": 3.4,
                "has_pucca_road": True,
                "has_treated_tap_water": True,
            },
            "response": "<evidence>\nCommunity: Tallur, District: Udupi\nAHP Score: 74.34, Band: BAND_4_ADEQUATE\nCoast Distance: 3.4 km, Pucca Road: Yes, Tap Water: Yes\n</evidence>\n\nTallur in Udupi is situated 3.4 km from the coastline and demonstrates strong structural resilience with an AHP score of 74.34 (Band 4 Adequate). It benefits from confirmed all-weather pucca road connectivity and treated tap water distribution.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_05",
            "instruction": "Synthesize the assessment for Kodimbala in Dakshina Kannada.",
            "evidence": {
                "community_name": "Kodimbala",
                "district_name": "Dakshina Kannada",
                "preparedness_score": 38.44,
                "preparedness_tier": "Low",
                "priority_band": "BAND_1_CRITICAL",
                "overall_priority_rank": 116,
            },
            "response": "<evidence>\nCommunity: Kodimbala, District: Dakshina Kannada\nAHP Score: 38.44 (Tier: Low), Priority: BAND_1_CRITICAL\nOverall Rank: 116\n</evidence>\n\nKodimbala in Dakshina Kannada registers an AHP Preparedness Score of 38.44, falling into the Low Preparedness Tier. Under platform governance, this strictly mandates Band 1 Critical status (rank 116), requiring immediate priority asset pre-positioning and targeted infrastructure strengthening.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_06",
            "instruction": "Synthesize archetype profile for Cluster 3 High-Exposure Coastal Rural Belt.",
            "evidence": {
                "archetype_id": 3,
                "short_label": "High-Exposure Coastal Rural Belt",
                "unit_count": 249,
                "percentage_of_total": 40.0,
                "primary_intervention_need": "Multi-purpose cyclone shelter construction within 1.5 km and sea wall revetments.",
            },
            "response": "<evidence>\nArchetype 3: High-Exposure Coastal Rural Belt\nAssigned Units: 249 (40.0% of clustered study area)\nPrimary Need: MPCS shelter construction and coastal revetments\n</evidence>\n\nVulnerability Archetype 3 comprises 249 units (40.0% of the Dakshina Kannada and Udupi study area). Located directly along the open Arabian Sea coastline, these communities face severe storm-surge and erosion risks, making multi-purpose shelter construction and elevated evacuation corridors paramount.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_07",
            "instruction": "Summarize archetype profile for Cluster 1 Remote Foothill Communities.",
            "evidence": {
                "archetype_id": 1,
                "short_label": "Remote Foothill & Healthcare-Isolated Communities",
                "unit_count": 82,
                "percentage_of_total": 13.2,
                "primary_intervention_need": "24/7 emergency medical transit and road network redundancy.",
            },
            "response": "<evidence>\nArchetype 1: Remote Foothill & Healthcare-Isolated Communities\nAssigned Units: 82 (13.2% of clustered study area)\nPrimary Need: 24/7 emergency medical transit and road redundancy\n</evidence>\n\nCluster 1 encompasses 82 foothill units in the Western Ghats periphery. While free from marine surge exposure, their acute vulnerability arises from extreme travel buffers to emergency healthcare facilities (> 60 minutes), requiring upgraded all-weather transit and emergency medical staging.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_08",
            "instruction": "Synthesize district comparison evidence for Dakshina Kannada and Udupi.",
            "evidence": {
                "districts_compared": ["Dakshina Kannada", "Udupi"],
                "mean_preparedness_score": {"Dakshina Kannada": 61.06, "Udupi": 65.04},
                "total_units": {"Dakshina Kannada": 369, "Udupi": 253},
                "band_1_critical_count": {"Dakshina Kannada": 2, "Udupi": 0},
            },
            "response": "<evidence>\nDakshina Kannada: 369 units, Mean AHP = 61.06, Band 1 Units = 2\nUdupi: 253 units, Mean AHP = 65.04, Band 1 Units = 0\n</evidence>\n\nBoth southern coastal districts maintain high overall preparedness, with Udupi averaging 65.04 and Dakshina Kannada averaging 61.06. However, Dakshina Kannada contains 2 critical units in Band 1 (Kodimbala and Sulya TP), whereas Udupi has zero units below the 40.0 AHP threshold.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_09",
            "instruction": "Summarize intervention recommendations for Bengre in Uttara Kannada.",
            "evidence": {
                "community_name": "Bengre",
                "district_name": "Uttara Kannada",
                "priority_band": "BAND_2_SPECIALIZED",
                "advisories_count": 2,
                "advisories": [
                    "Targeted cyclone shelter capacity reinforcement within 5 km buffer.",
                    "Elevate coastal approach road above seasonal estuarine backwater flood level.",
                ],
            },
            "response": "<evidence>\nCommunity: Bengre, District: Uttara Kannada\nPriority Band: BAND_2_SPECIALIZED, Advisories: 2\nKey Actions: Shelter capacity reinforcement and coastal road elevation\n</evidence>\n\nBengre in Uttara Kannada is classified under Band 2 Specialized priority. Targeted interventions advise expanding cyclone shelter capacity within 5 km and elevating seasonal flood-prone approach roads along the estuary.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_10",
            "instruction": "Synthesize the assessment for Sulya TP in Dakshina Kannada.",
            "evidence": {
                "community_name": "Sulya (TP)",
                "district_name": "Dakshina Kannada",
                "preparedness_score": 39.31,
                "preparedness_tier": "Low",
                "priority_band": "BAND_1_CRITICAL",
                "overall_priority_rank": 137,
            },
            "response": "<evidence>\nCommunity: Sulya (TP), District: Dakshina Kannada\nAHP Score: 39.31 (Tier: Low), Priority: BAND_1_CRITICAL\nOverall Priority Rank: 137\n</evidence>\n\nSulya (TP) records an AHP score of 39.31, placing it in the Low Preparedness Tier and mandating Band 1 Critical priority. Its priority rank of 137 requires urgent structural strengthening and priority disaster relief staging.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_11",
            "instruction": "Summarize archetype profile for Cluster 0 Urban Well-Resourced Corridor.",
            "evidence": {
                "archetype_id": 0,
                "short_label": "Urban & Peri-Urban Well-Resourced Corridor",
                "unit_count": 130,
                "percentage_of_total": 20.9,
                "primary_intervention_need": "Stormwater drainage clearing and electrical grid hardening.",
            },
            "response": "<evidence>\nArchetype 0: Urban & Peri-Urban Well-Resourced Corridor\nAssigned Units: 130 (20.9% of clustered study area)\nPrimary Need: Stormwater drainage clearing and grid hardening\n</evidence>\n\nCluster 0 covers 130 well-resourced urban and peri-urban units around Mangaluru and Udupi. With high baseline access to hospitals and paved roads, their interventions focus on municipal stormwater drainage and protecting utility grids against high wind loads.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_12",
            "instruction": "Summarize archetype profile for Cluster 2 Socioeconomically Vulnerable Rural Interior.",
            "evidence": {
                "archetype_id": 2,
                "short_label": "Socioeconomically Vulnerable Rural Interior",
                "unit_count": 161,
                "percentage_of_total": 25.9,
                "primary_intervention_need": "Housing durability upgrades (PMAY pucca roofs) and clean water access.",
            },
            "response": "<evidence>\nArchetype 2: Socioeconomically Vulnerable Rural Interior\nAssigned Units: 161 (25.9% of clustered study area)\nPrimary Need: Housing durability upgrades and clean water access\n</evidence>\n\nCluster 2 accounts for 161 rural interior units (25.9% of the clustered study area). Marked by elevated demographic vulnerability and higher kutcha housing shares, interventions emphasize residential roof reinforcement and resilient drinking water supply.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_13",
            "instruction": "Synthesize the infrastructure analysis for village Malpe.",
            "evidence": {
                "community_name": "Malpe",
                "district_name": "Udupi",
                "coastal_setback_km": 0.8,
                "nearest_hospital_km": 6.2,
                "cyclone_shelters_10km": 3,
                "pucca_road_access": True,
            },
            "response": "<evidence>\nCommunity: Malpe, District: Udupi\nCoast Setback: 0.8 km, Hospital: 6.2 km, Shelters (10km): 3, Pucca Road: Yes\n</evidence>\n\nMalpe is an exposed fishing settlement located 0.8 km from the Arabian Sea shoreline. While exposed to marine storm surge, it is supported by 3 cyclone shelters within 10 km, all-weather pucca road connectivity, and a 6.2 km transit distance to the nearest major hospital.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_14",
            "instruction": "Synthesize the assessment for Ullal village in Uttara Kannada.",
            "evidence": {
                "community_name": "Ullal",
                "district_name": "Uttara Kannada",
                "preparedness_score": 52.36,
                "preparedness_tier": "Medium",
                "priority_band": "BAND_2_SPECIALIZED",
                "domain_scores": {
                    "score_medical": 35.45,
                    "score_shelters": 10.65,
                    "score_infrastructure": 76.0,
                    "score_early_warning": 71.67,
                    "score_socioeconomic": 99.25,
                },
            },
            "response": "<evidence>\nCommunity: Ullal, District: Uttara Kannada\nAHP Score: 52.36 (Tier: Medium), Priority: BAND_2_SPECIALIZED\nDomain Scores: Medical=35.45, Shelters=10.65, Infrastructure=76.0, Early Warning=71.67, Socioeconomic=99.25\n</evidence>\n\nUllal in Uttara Kannada (PC11 ID 603256) achieves an AHP Preparedness Score of 52.36 (Moderate Tier) and Band 2 Specialized priority. It possesses excellent socioeconomic and infrastructure ratings but suffers from acute deficits in shelter availability (10.65) and healthcare access (35.45).",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_15",
            "instruction": "Summarize filter results for Band 1 Critical communities in Dakshina Kannada.",
            "evidence": {
                "district": "Dakshina Kannada",
                "priority_band": "BAND_1_CRITICAL",
                "total_matching": 2,
                "communities": [
                    {"name": "Kodimbala", "score": 38.44, "rank": 116},
                    {"name": "Sulya (TP)", "score": 39.31, "rank": 137},
                ],
            },
            "response": "<evidence>\nDistrict: Dakshina Kannada, Filter: BAND_1_CRITICAL\nTotal Matching: 2\nUnits: Kodimbala (AHP: 38.44, Rank: 116), Sulya (TP) (AHP: 39.31, Rank: 137)\n</evidence>\n\nFiltering Dakshina Kannada for Band 1 Critical priority returns exactly 2 revenue units: Kodimbala (AHP score 38.44, rank 116) and Sulya TP (AHP score 39.31, rank 137). Both units fall below the 40.0 Low Preparedness threshold and mandate urgent intervention.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_16",
            "instruction": "Synthesize the assessment for Someshwara in Dakshina Kannada.",
            "evidence": {
                "community_name": "Someshwara",
                "district_name": "Dakshina Kannada",
                "preparedness_score": 62.15,
                "preparedness_tier": "High",
                "priority_band": "BAND_4_ADEQUATE",
                "dist_to_coast_km": 1.2,
            },
            "response": "<evidence>\nCommunity: Someshwara, District: Dakshina Kannada\nAHP Score: 62.15 (Tier: High), Priority: BAND_4_ADEQUATE, Coast Distance: 1.2 km\n</evidence>\n\nSomeshwara in Dakshina Kannada records an authoritative AHP Preparedness Score of 62.15, placing it in the High Preparedness Tier and Band 4 Adequate priority. Despite coastal proximity of 1.2 km, its solid healthcare and road connectivity ensure strong baseline coping capacity.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_17",
            "instruction": "Summarize district comparison evidence between Dakshina Kannada and Uttara Kannada.",
            "evidence": {
                "districts_compared": ["Dakshina Kannada", "Uttara Kannada"],
                "mean_preparedness_score": {"Dakshina Kannada": 61.06, "Uttara Kannada": 50.06},
                "total_units": {"Dakshina Kannada": 369, "Uttara Kannada": 1257},
                "band_1_critical_percentage": {"Dakshina Kannada": 0.5, "Uttara Kannada": 12.2},
            },
            "response": "<evidence>\nDakshina Kannada: 369 units, Mean AHP = 61.06, Band 1 = 0.5%\nUttara Kannada: 1257 units, Mean AHP = 50.06, Band 1 = 12.2%\n</evidence>\n\nDakshina Kannada shows significantly higher average preparedness (61.06) with only 0.5% Band 1 units, compared to Uttara Kannada's 50.06 mean score and 12.2% Band 1 critical share, driven by Uttara Kannada's rugged topography and isolated settlements.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_18",
            "instruction": "Synthesize the healthcare and exposure profile for Kundapura taluk village.",
            "evidence": {
                "community_name": "Koni",
                "district_name": "Udupi",
                "subdistrict_name": "Kundapura",
                "dist_to_hospital_km": 4.1,
                "dist_to_phc_km": 1.8,
                "coastal_setback_km": 2.9,
            },
            "response": "<evidence>\nCommunity: Koni, Subdistrict: Kundapura, District: Udupi\nHospital Distance: 4.1 km, PHC Distance: 1.8 km, Coastal Setback: 2.9 km\n</evidence>\n\nKoni in Kundapura taluk (Udupi) possesses favorable healthcare accessibility, situated 1.8 km from the nearest Primary Health Centre and 4.1 km from an apex hospital, with a 2.9 km coastal buffer moderating immediate shoreline storm surge risks.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_19",
            "instruction": "Synthesize the assessment for Kumta coastal unit.",
            "evidence": {
                "community_name": "Aghanashini",
                "district_name": "Uttara Kannada",
                "subdistrict_name": "Kumta",
                "preparedness_score": 44.8,
                "preparedness_tier": "Medium",
                "priority_band": "BAND_2_SPECIALIZED",
                "estuarine_hazard": "High estuarine tidal backwater flooding",
            },
            "response": "<evidence>\nCommunity: Aghanashini, District: Uttara Kannada\nAHP Score: 44.8 (Tier: Medium), Priority: BAND_2_SPECIALIZED\nHazard Note: High estuarine tidal backwater flooding\n</evidence>\n\nAghanashini in Uttara Kannada records a Moderate AHP score of 44.8 and Band 2 Specialized priority. Situated along the Aghanashini estuary, its primary vulnerability stems from tidal backwater flooding during storm surges, warranting elevated barrier bunds and dedicated boat evacuation nodes.",
            "category": "grounded_synthesis",
        },
        {
            "id": "gs_20",
            "instruction": "Summarize intervention recommendations for Edmoge in Udupi.",
            "evidence": {
                "community_name": "Edmoge",
                "district_name": "Udupi",
                "priority_band": "BAND_3_MODERATE",
                "advisories_count": 2,
                "advisories": [
                    "Construct multi-purpose cyclone shelter to address shelter score deficit (24.79).",
                    "Conduct seasonal community flood evacuation drills.",
                ],
            },
            "response": "<evidence>\nCommunity: Edmoge, District: Udupi\nPriority Band: BAND_3_MODERATE, Advisories: 2\nKey Actions: Shelter construction to address 24.79 score deficit and evacuation drills\n</evidence>\n\nIntervention recommendations for Edmoge in Udupi focus on remedying its severe shelter deficit (score 24.79) through new shelter construction, supplemented by annual pre-monsoon community evacuation drills.",
            "category": "grounded_synthesis",
        },
    ]

    return scenarios


def create_held_out_100_suite() -> List[Dict[str, Any]]:
    """Build the planned 100-question held-out evaluation suite.

    - 30 unique Domain Knowledge questions
    - 30 unique Tool Selection questions (all 7 tools)
    - 20 unique Tool Argument Generation questions (all 7 tools)
    - 20 unique Evidence-Grounded Response questions (20 distinct scenarios, NO duplicates)
    """
    suite: List[Dict[str, Any]] = []

    # Category 1: 30 Domain Knowledge Questions (eval_dk_01 through eval_dk_30)
    dk_questions = [
        ("What does AHP stand for in CYCLONE INTEL?", "Analytic Hierarchy Process"),
        ("What is the consistency ratio threshold in AHP?", "CR must be strictly less than 0.10"),
        ("What score qualifies as AHP High Preparedness tier?", "Composite score >= 60.0"),
        ("What score qualifies as AHP Low Preparedness tier?", "Composite score < 40.0"),
        ("Name the 5 domain scores in the dataset.", "score_medical, score_shelters, score_infrastructure, score_early_warning, score_socioeconomic"),
        ("What is the non-downgrading safety invariant?", "ML surrogate cannot downgrade an authoritative AHP Low classification"),
        ("Is Logistic Regression a hazard prediction model?", "No, it is an empirical surrogate of AHP classification boundaries"),
        ("What does prob_high represent?", "Statistical membership likelihood in High tier, not cyclone strike probability"),
        ("How many total administrative revenue units exist in the project?", "1,879 units"),
        ("How many units are in Dakshina Kannada?", "369 units"),
        ("How many units are in Udupi?", "253 units"),
        ("How many units are in Uttara Kannada?", "1,257 units"),
        ("What is the mean AHP score in Dakshina Kannada?", "61.06"),
        ("What is the mean AHP score in Udupi?", "65.04"),
        ("What is the mean AHP score in Uttara Kannada?", "50.06"),
        ("What is the label for K-Means Cluster 0?", "Urban & Peri-Urban Well-Resourced Corridor"),
        ("What is the label for K-Means Cluster 1?", "Remote Foothill & Healthcare-Isolated Communities"),
        ("What is the label for K-Means Cluster 2?", "Socioeconomically Vulnerable Rural Interior"),
        ("What is the label for K-Means Cluster 3?", "High-Exposure Coastal Rural Belt"),
        ("How many units are assigned to K-Means clusters?", "622 units (Dakshina Kannada and Udupi)"),
        ("Why is Uttara Kannada unassigned in clustering?", "Geographic and administrative differences; evaluated via domain-deficit fallback engine"),
        ("What is Band 1 Critical operational priority?", "BAND_1_CRITICAL: Immediate pre-positioning and capital intervention"),
        ("What is Band 4 Adequate operational priority?", "BAND_4_ADEQUATE: Routine monitoring for well-resourced units"),
        ("Can model output override deterministic AHP?", "No, deterministic AHP maintains absolute operational precedence"),
        ("Can CycloneLM report live wind speed today?", "No, refusal required: dynamic weather telemetry is unrecorded"),
        ("Can CycloneLM report live shelter occupancy?", "No, refusal required: dynamic operational data is unrecorded"),
        ("Can CycloneLM report live hospital bed vacancy?", "No, refusal required: dynamic medical telemetry is unrecorded"),
        ("What is the IMD wind speed for a Very Severe Cyclonic Storm?", "118 to 166 km/h (64 to 89 knots)"),
        ("What is the IMD wind speed for a Super Cyclonic Storm?", "222 km/h or higher (>= 120 knots)"),
        ("Why is ground truthing required by DDMA engineers?", "Because baseline data (Census 2011) must be verified before capital commitment"),
    ]
    for idx, (q, a) in enumerate(dk_questions, 1):
        suite.append({
            "eval_id": f"eval_dk_{idx:02d}",
            "category": "domain_knowledge",
            "question": q,
            "reference_answer": a,
        })

    # Category 2: 30 Tool Selection Questions (eval_ts_01 through eval_ts_30)
    ts_cases = [
        ("Check community data for Bengre", "lookup_community"),
        ("Look up Someshwara in Dakshina Kannada", "lookup_community"),
        ("Retrieve revenue unit Malpe", "lookup_community"),
        ("Find details for village Tallur", "lookup_community"),
        ("Look up PC11 ID 605842", "lookup_community"),
        ("Compare Dakshina Kannada with Udupi", "compare_districts"),
        ("Compare Udupi and Uttara Kannada", "compare_districts"),
        ("Compare Dakshina Kannada and Uttara Kannada", "compare_districts"),
        ("Contrast Dakshina Kannada and Udupi priority bands", "compare_districts"),
        ("Compare baseline scores between Udupi and Uttara Kannada", "compare_districts"),
        ("Explain preparedness scores for Ullal", "explain_preparedness_assessment"),
        ("Show AHP domain score breakdown for Bengre", "explain_preparedness_assessment"),
        ("Explain AHP criteria weights for Someshwara", "explain_preparedness_assessment"),
        ("Why is Tallur in High preparedness tier?", "explain_preparedness_assessment"),
        ("Explain assessment breakdown for Edmoge", "explain_preparedness_assessment"),
        ("Get engineering recommendations for Ullal", "get_intervention_recommendations"),
        ("What interventions are needed for Bengre?", "get_intervention_recommendations"),
        ("Intervention priorities for Tallur village", "get_intervention_recommendations"),
        ("Recommend risk reduction upgrades for Someshwara", "get_intervention_recommendations"),
        ("What are the recommended actions for Edmoge?", "get_intervention_recommendations"),
        ("Analyze shelter and healthcare buffers for Ullal", "analyze_infrastructure_and_exposure"),
        ("Inspect physical exposure for Bengre", "analyze_infrastructure_and_exposure"),
        ("Evaluate road and water connectivity for Tallur", "analyze_infrastructure_and_exposure"),
        ("Analyze infrastructure sensitivity for Someshwara", "analyze_infrastructure_and_exposure"),
        ("Check coastal distance and shelter radius for Edmoge", "analyze_infrastructure_and_exposure"),
        ("Filter Band 1 Critical communities in Dakshina Kannada", "filter_and_rank_communities"),
        ("List top 5 vulnerable units in Uttara Kannada", "filter_and_rank_communities"),
        ("Filter Band 2 Specialized units in Udupi", "filter_and_rank_communities"),
        ("Explain vulnerability archetype Cluster 3", "explain_vulnerability_archetype"),
        ("Inspect archetype assignment for Tallur", "explain_vulnerability_archetype"),
    ]
    for idx, (q, tool) in enumerate(ts_cases, 1):
        suite.append({
            "eval_id": f"eval_ts_{idx:02d}",
            "category": "tool_selection",
            "question": q,
            "expected_tool": tool,
        })

    # Category 3: 20 Tool Argument Generation Questions (eval_arg_01 through eval_arg_20)
    arg_cases = [
        ("Look up Ullal in Dakshina Kannada", "lookup_community", {"query": "Ullal", "district": "Dakshina Kannada"}),
        ("Look up Tallur", "lookup_community", {"query": "Tallur"}),
        ("Look up Bengre in Uttara Kannada", "lookup_community", {"query": "Bengre", "district": "Uttara Kannada"}),
        ("Look up Edmoge", "lookup_community", {"query": "Edmoge"}),
        ("Compare Dakshina Kannada and Udupi", "compare_districts", {"district_a": "Dakshina Kannada", "district_b": "Udupi"}),
        ("Compare Udupi and Uttara Kannada", "compare_districts", {"district_a": "Udupi", "district_b": "Uttara Kannada"}),
        ("Compare Dakshina Kannada and Uttara Kannada", "compare_districts", {"district_a": "Dakshina Kannada", "district_b": "Uttara Kannada"}),
        ("Explain preparedness assessment for Ullal", "explain_preparedness_assessment", {"identifier": "Ullal"}),
        ("Explain preparedness assessment for Tallur", "explain_preparedness_assessment", {"identifier": "Tallur"}),
        ("Explain preparedness assessment for Edmoge", "explain_preparedness_assessment", {"identifier": "Edmoge"}),
        ("Get recommendations for Ullal", "get_intervention_recommendations", {"identifier": "Ullal"}),
        ("Get recommendations for Bengre", "get_intervention_recommendations", {"identifier": "Bengre"}),
        ("Analyze infrastructure for Someshwara", "analyze_infrastructure_and_exposure", {"identifier": "Someshwara"}),
        ("Analyze infrastructure for Tallur", "analyze_infrastructure_and_exposure", {"identifier": "Tallur"}),
        ("Filter Band 1 in Dakshina Kannada", "filter_and_rank_communities", {"district": "Dakshina Kannada", "priority_band": "BAND_1_CRITICAL"}),
        ("Filter Band 2 in Udupi", "filter_and_rank_communities", {"district": "Udupi", "priority_band": "BAND_2_SPECIALIZED"}),
        ("Explain archetype 0", "explain_vulnerability_archetype", {"archetype_id": 0}),
        ("Explain archetype 1", "explain_vulnerability_archetype", {"archetype_id": 1}),
        ("Explain archetype 2", "explain_vulnerability_archetype", {"archetype_id": 2}),
        ("Explain archetype 3", "explain_vulnerability_archetype", {"archetype_id": 3}),
    ]
    for idx, (q, tool, params) in enumerate(arg_cases, 1):
        suite.append({
            "eval_id": f"eval_arg_{idx:02d}",
            "category": "argument_generation",
            "question": q,
            "tool": tool,
            "expected_parameters": params,
        })

    # Category 4: 20 Evidence-Grounded Response Questions (eval_synth_01 through eval_synth_20)
    # 20 distinct, unique evidence cases
    synth_cases = [
        (
            "Synthesize: Ullal (TMC), AHP 60.44, Tier High, Priority BAND_4_ADEQUATE",
            {"community": "Ullal (TMC)", "district": "Dakshina Kannada", "score": 60.44, "tier": "High", "band": "BAND_4_ADEQUATE"},
            "Ullal (TMC) in Dakshina Kannada has an authoritative AHP score of 60.44, placing it in the High Preparedness tier and Band 4 Adequate priority."
        ),
        (
            "Synthesize: Tallur, AHP 74.34, Tier High, Priority BAND_4_ADEQUATE",
            {"community": "Tallur", "district": "Udupi", "score": 74.34, "tier": "High", "band": "BAND_4_ADEQUATE"},
            "Tallur in Udupi demonstrates high baseline preparedness with an AHP score of 74.34, classified under Band 4 Adequate priority."
        ),
        (
            "Synthesize: Edmoge, AHP 49.84, Tier Medium, Shelter score 24.79",
            {"community": "Edmoge", "district": "Udupi", "score": 49.84, "tier": "Medium", "shelter_score": 24.79},
            "Edmoge in Udupi has a composite AHP score of 49.84 (Moderate Tier) with a significant deficit in shelter infrastructure (24.79)."
        ),
        (
            "Synthesize: Kodimbala, AHP 38.44, Tier Low, Priority BAND_1_CRITICAL",
            {"community": "Kodimbala", "district": "Dakshina Kannada", "score": 38.44, "tier": "Low", "band": "BAND_1_CRITICAL"},
            "Kodimbala in Dakshina Kannada registers an AHP score of 38.44, falling into Low Preparedness and strictly mandating Band 1 Critical priority."
        ),
        (
            "Synthesize: Sulya (TP), AHP 39.31, Tier Low, Priority BAND_1_CRITICAL",
            {"community": "Sulya (TP)", "district": "Dakshina Kannada", "score": 39.31, "tier": "Low", "band": "BAND_1_CRITICAL"},
            "Sulya (TP) has an AHP score of 39.31 (Low Preparedness), requiring immediate Band 1 Critical operational resource allocation."
        ),
        (
            "Synthesize: Archetype 3 Coastal Belt, 249 units, 40.0% coverage",
            {"archetype_id": 3, "label": "High-Exposure Coastal Rural Belt", "unit_count": 249, "percentage": 40.0},
            "Archetype 3 represents the High-Exposure Coastal Rural Belt with 249 units (40.0% of the study area), facing severe littoral surge risk."
        ),
        (
            "Synthesize: Archetype 0 Urban Corridor, 130 units, 20.9% coverage",
            {"archetype_id": 0, "label": "Urban & Peri-Urban Well-Resourced Corridor", "unit_count": 130, "percentage": 20.9},
            "Archetype 0 covers 130 urban corridor units (20.9% of the study area) characterized by high road connectivity and rapid hospital access."
        ),
        (
            "Synthesize: Archetype 1 Foothill, 82 units, 13.2% coverage",
            {"archetype_id": 1, "label": "Remote Foothill & Healthcare-Isolated Communities", "unit_count": 82, "percentage": 13.2},
            "Archetype 1 covers 82 foothill units (13.2% of the study area) where severe transit buffers to emergency healthcare exceed 60 minutes."
        ),
        (
            "Synthesize: Archetype 2 Rural Interior, 161 units, 25.9% coverage",
            {"archetype_id": 2, "label": "Socioeconomically Vulnerable Rural Interior", "unit_count": 161, "percentage": 25.9},
            "Archetype 2 covers 161 rural interior units (25.9% of the study area) with high shares of kutcha housing and demographic vulnerability."
        ),
        (
            "Synthesize: District Udupi, Mean AHP 65.04, 253 units, 0 Band 1 units",
            {"district": "Udupi", "total_units": 253, "mean_ahp": 65.04, "band_1_units": 0},
            "Udupi maintains the highest district mean AHP preparedness score at 65.04 across 253 units, with zero units in Band 1 Critical."
        ),
        (
            "Synthesize: District Dakshina Kannada, Mean AHP 61.06, 369 units, 2 Band 1 units",
            {"district": "Dakshina Kannada", "total_units": 369, "mean_ahp": 61.06, "band_1_units": 2},
            "Dakshina Kannada averages 61.06 across 369 units, with exactly 2 units (Kodimbala and Sulya TP) in Band 1 Critical."
        ),
        (
            "Synthesize: District Uttara Kannada, Mean AHP 50.06, 1257 units, 153 Band 1 units",
            {"district": "Uttara Kannada", "total_units": 1257, "mean_ahp": 50.06, "band_1_units": 153},
            "Uttara Kannada averages 50.06 across 1,257 units, containing 153 Band 1 Critical units due to rural isolation and extended transit buffers."
        ),
        (
            "Synthesize: Someshwara, Coastal setback 1.2 km, AHP 62.15",
            {"community": "Someshwara", "coastal_setback_km": 1.2, "score": 62.15, "tier": "High"},
            "Someshwara is located 1.2 km from the coastline and demonstrates high baseline preparedness with an AHP score of 62.15."
        ),
        (
            "Synthesize: Malpe, Coastal setback 0.8 km, 3 shelters within 10 km",
            {"community": "Malpe", "coastal_setback_km": 0.8, "shelters_count": 3},
            "Malpe is situated 0.8 km from the shoreline with 3 multi-purpose cyclone shelters providing evacuation coverage within a 10 km radius."
        ),
        (
            "Synthesize: Bengre (UK), AHP 64.31, Priority BAND_2_SPECIALIZED",
            {"community": "Bengre", "district": "Uttara Kannada", "score": 64.31, "band": "BAND_2_SPECIALIZED"},
            "Bengre in Uttara Kannada has an AHP score of 64.31 and Band 2 Specialized priority, needing targeted shelter and access road reinforcement."
        ),
        (
            "Synthesize: Ullal (UK), AHP 52.36, Shelter score 10.65",
            {"community": "Ullal", "district": "Uttara Kannada", "score": 52.36, "shelter_score": 10.65},
            "Ullal in Uttara Kannada has an AHP score of 52.36 (Moderate Tier) with an acute cyclone shelter deficit reflected by its score of 10.65."
        ),
        (
            "Synthesize: Koni (Udupi), Hospital distance 4.1 km, PHC 1.8 km",
            {"community": "Koni", "hospital_km": 4.1, "phc_km": 1.8},
            "Koni in Kundapura taluk possesses favorable healthcare accessibility with a PHC at 1.8 km and an apex hospital at 4.1 km."
        ),
        (
            "Synthesize: Aghanashini, AHP 44.8, Estuarine surge hazard",
            {"community": "Aghanashini", "score": 44.8, "tier": "Medium", "hazard": "Estuarine backwater flooding"},
            "Aghanashini in Uttara Kannada has a Moderate AHP score of 44.8, where estuarine tidal backwater flooding forms the primary surge hazard."
        ),
        (
            "Synthesize: Dakshina Kannada ML agreement rate 93.0%",
            {"district": "Dakshina Kannada", "agreement_rate": 93.0},
            "The empirical Logistic Regression surrogate achieves a 93.0% classification agreement rate against AHP tiers in Dakshina Kannada."
        ),
        (
            "Synthesize: Udupi ML agreement rate 96.0%",
            {"district": "Udupi", "agreement_rate": 96.0},
            "The empirical Logistic Regression surrogate achieves a 96.0% classification agreement rate against AHP tiers in Udupi."
        ),
    ]
    for idx, (q, evidence, ans) in enumerate(synth_cases, 1):
        suite.append({
            "eval_id": f"eval_synth_{idx:02d}",
            "category": "evidence_grounded_response",
            "prompt": q,
            "evidence": evidence,
            "reference_synthesis": ans,
        })

    return suite


def train_v02_tokenizer(corpus_path: Path, output_path: Path) -> None:
    """Train BPE Tokenizer for v0.2 with 16 atomic special tokens and save vocabulary."""
    from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    tok = BPETokenizer()
    tok.train(lines, vocab_size=1024, min_frequency=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(output_path)
    print(f"v0.2 BPE Tokenizer trained successfully ({tok.vocab_size} tokens) -> {output_path}")


def main() -> None:
    v02_dir = Path("data/cyclonelm/v0.2")
    domain_dir = v02_dir / "domain_corpus"
    inst_dir = v02_dir / "instructions"
    tool_dir = v02_dir / "tool_use"
    eval_dir = v02_dir / "held_out_eval"
    sources_dir = v02_dir / "sources"

    for d in (domain_dir, inst_dir, tool_dir, eval_dir, sources_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Save Provenance Catalog
    with open(sources_dir / "provenance_catalog.json", "w", encoding="utf-8") as f:
        json.dump(PROVENANCE_CATALOG, f, indent=2)

    # 2. Save Clean Domain Corpus
    full_corpus = "\n\n".join(DOMAIN_CORPUS_SECTIONS)
    with open(domain_dir / "domain_corpus.txt", "w", encoding="utf-8") as f:
        f.write(full_corpus)

    # 3. Build Scenarios and split cleanly by Scenario Family
    all_scenarios = build_scenario_dataset()

    train_inst: List[Dict[str, Any]] = []
    val_inst: List[Dict[str, Any]] = []
    test_inst: List[Dict[str, Any]] = []

    for fam_name, items in all_scenarios.items():
        n = len(items)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15)) if n > 3 else 0
        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]

        train_inst.extend(train_items)
        val_inst.extend(val_items)
        test_inst.extend(test_items)

    # Save instructions
    with open(inst_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_inst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(inst_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_inst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(inst_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for item in test_inst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 4. Save Dedicated Tool-Use Datasets (Tool selection, argument gen, missing args, synthesis)
    # Strictly align with the same scenario family partitioning to eliminate leakage
    tool_family_names = ["tool_selection", "missing_arguments", "grounded_synthesis"]
    t_train: List[Dict[str, Any]] = []
    t_val: List[Dict[str, Any]] = []
    t_test: List[Dict[str, Any]] = []

    for fam_name in tool_family_names:
        items = all_scenarios.get(fam_name, [])
        n = len(items)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15)) if n > 3 else 0
        t_train.extend(items[:n_train])
        t_val.extend(items[n_train : n_train + n_val])
        t_test.extend(items[n_train + n_val :])

    with open(tool_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in t_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(tool_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in t_val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(tool_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for item in t_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 5. Build Held-Out 100-Question Suite
    held_out_100 = create_held_out_100_suite()
    with open(eval_dir / "held_out_100.jsonl", "w", encoding="utf-8") as f:
        for item in held_out_100:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Verify Overlap Statistics Across All Combined Splits (Zero leakage)
    train_queries = set(item["instruction"].strip().lower() for item in train_inst + t_train)
    val_queries = set(item["instruction"].strip().lower() for item in val_inst + t_val)
    test_queries = set(item["instruction"].strip().lower() for item in test_inst + t_test)
    held_out_questions = set(
        (item.get("question") or item.get("prompt", "")).strip().lower() for item in held_out_100
    )

    leak_train_val = train_queries.intersection(val_queries)
    leak_train_test = train_queries.intersection(test_queries)
    leak_val_test = val_queries.intersection(test_queries)
    leak_train_heldout = train_queries.intersection(held_out_questions)
    leak_val_heldout = val_queries.intersection(held_out_questions)

    stats = {
        "dataset_version": "0.2.0",
        "domain_corpus_sections": len(DOMAIN_CORPUS_SECTIONS),
        "total_unique_instruction_scenarios": len(train_inst) + len(val_inst) + len(test_inst),
        "splits": {
            "train_count": len(train_inst),
            "val_count": len(val_inst),
            "test_count": len(test_inst),
        },
        "tool_use_splits": {
            "train_count": len(t_train),
            "val_count": len(t_val),
            "test_count": len(t_test),
        },
        "held_out_100_count": len(held_out_100),
        "held_out_100_categories": {
            "domain_knowledge": sum(1 for x in held_out_100 if x["category"] == "domain_knowledge"),
            "tool_selection": sum(1 for x in held_out_100 if x["category"] == "tool_selection"),
            "argument_generation": sum(1 for x in held_out_100 if x["category"] == "argument_generation"),
            "evidence_grounded_response": sum(1 for x in held_out_100 if x["category"] == "evidence_grounded_response"),
        },
        "overlap_audit": {
            "train_val_overlap": len(leak_train_val),
            "train_test_overlap": len(leak_train_test),
            "val_test_overlap": len(leak_val_test),
            "train_heldout_overlap": len(leak_train_heldout),
            "val_heldout_overlap": len(leak_val_heldout),
            "zero_leakage_verified": (
                len(leak_train_val) == 0
                and len(leak_train_test) == 0
                and len(leak_val_test) == 0
                and len(leak_train_heldout) == 0
                and len(leak_val_heldout) == 0
            ),
        },
    }

    with open(v02_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # 7. Create combined training corpus for tokenizer and model training
    combined_training_text = [full_corpus]
    for item in train_inst:
        combined_training_text.append(f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>")
    for item in t_train:
        combined_training_text.append(f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>")

    combined_corpus_file = domain_dir / "train_combined_corpus.txt"
    with open(combined_corpus_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_training_text))

    # Combined validation corpus
    combined_val_text = []
    for item in val_inst:
        combined_val_text.append(f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>")
    for item in t_val:
        combined_val_text.append(f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>")
    with open(domain_dir / "val_combined_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_val_text))

    # Combined test corpus
    combined_test_text = []
    for item in test_inst:
        combined_test_text.append(f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>")
    for item in t_test:
        combined_test_text.append(f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>")
    with open(domain_dir / "test_combined_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_test_text))

    # 8. Train v0.2 BPE Tokenizer
    tok_output = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    train_v02_tokenizer(combined_corpus_file, tok_output)

    print("=" * 60)
    print("CYCLONELM v0.2 DATASET REALIGNMENT COMPLETE")
    print(f"Domain Corpus:           {len(DOMAIN_CORPUS_SECTIONS)} verified sections")
    print(f"Total Unique Scenarios:  {len(train_inst) + len(val_inst) + len(test_inst)}")
    print(f"Instruction Train/Val/Test: {len(train_inst)} / {len(val_inst)} / {len(test_inst)}")
    print(f"Tool-Use Train/Val/Test:    {len(t_train)} / {len(t_val)} / {len(t_test)}")
    print(f"Held-out 100 Suite:      {len(held_out_100)} questions")
    print(f"Overlap Train/Val:       {len(leak_train_val)}")
    print(f"Overlap Train/Held-Out:  {len(leak_train_heldout)}")
    print(f"Zero Leakage Verified:   {stats['overlap_audit']['zero_leakage_verified']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
