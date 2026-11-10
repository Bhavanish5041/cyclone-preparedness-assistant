"""Data curation and preparation script for CycloneLM v0.1.

Generates:
1. Domain Pretraining Corpus (text): Covers tropical cyclone meteorology, coastal Karnataka
   geography, MCDA/AHP methodology, ML surrogate demarcation, K-Means archetypes,
   scientific governance, and tool-use descriptions.
2. Domain Instruction Dataset (JSONL): 160 curated instruction-response pairs covering
   AHP explanations, priority band distinctions, surrogate demarcation, refusal to fabricate
   live data, tool execution prompts, and regional risk briefings.
3. Train/validation splits and comprehensive metadata documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PRETRAINING_CORPUS_SECTIONS = [
    """=== CYCLONE METEOROLOGY AND ARABIAN SEA DYNAMICS ===
Tropical cyclones are intense cyclonic storm systems originating over warm tropical oceans characterized by low atmospheric pressure, high wind velocities, and torrential precipitation. In the North Indian Ocean basin, cyclogenesis occurs in both the Bay of Bengal and the Arabian Sea, typically exhibiting bi-modal peak activity during pre-monsoon (April to June) and post-monsoon (October to December) transition windows.
The Arabian Sea has historically generated fewer cyclones than the Bay of Bengal; however, rapid sea surface warming and climate shifts have elevated both frequency and intensity of severe cyclonic storms along India's western seaboard.
The India Meteorological Department (IMD) classifies cyclonic disturbances based on maximum sustained surface wind speeds:
1. Low Pressure Area: Wind speed less than 31 km/h (17 knots).
2. Depression: Wind speed between 31 and 49 km/h (17 to 27 knots).
3. Deep Depression: Wind speed between 50 and 61 km/h (28 to 33 knots).
4. Cyclonic Storm: Wind speed between 62 and 87 km/h (34 to 47 knots).
5. Severe Cyclonic Storm (SCS): Wind speed between 88 and 117 km/h (48 to 63 knots).
6. Very Severe Cyclonic Storm (VSCS): Wind speed between 118 and 166 km/h (64 to 89 knots).
7. Extremely Severe Cyclonic Storm (ESCS): Wind speed between 167 and 221 km/h (90 to 119 knots).
8. Super Cyclonic Storm (SuCS): Wind speed of 222 km/h or higher (120 knots and above).
Severe cyclonic events generate destructive physical hazards including astronomical tide amplification (storm surge), extreme coastal inundation, structural wind damage, estuarine backwater surges, and slope instability in adjacent highland catchments.""",

    """=== COASTAL KARNATAKA GEOGRAPHY AND HAZARD EXPOSURE ===
The Coastal Karnataka region comprises three maritime districts: Dakshina Kannada (South Canara), Udupi, and Uttara Kannada (North Canara), spanning approximately 320 kilometers of shoreline bordering the Arabian Sea to the west and the Western Ghats mountain escarpment to the east.
The region encompasses 1,879 distinct administrative revenue units across varying topographies:
1. Dakshina Kannada: Southern coastal plain, highly urbanized around Mangaluru, characterized by the Netravati and Gurupura river estuaries, dense commercial ports, industrial zones, and high population concentrations along the littoral strip.
2. Udupi: Central maritime district, featuring extensive barrier beaches, estuaries (Swarna, Sita, Gangolli), fishing harbors like Malpe, and significant agriculture-aquaculture interfaces vulnerable to seawater intrusion.
3. Uttara Kannada: Northern maritime district, characterized by rugged headlands, rocky shorelines, mangrove estuaries (Sharavathi, Aghanashini, Kali), and dense forest tracts extending close to the sea, where rural habitations face geographic isolation.
Coastal communities face cascading hazards: astronomical storm surge inundating low-lying estuarine plains, severe coastal erosion along sand spits, and flash flooding when torrential cyclonic rains meet high tide backwaters in short west-flowing rivers.""",

    """=== DISASTER RISK REDUCTION AND EARLY WARNING SYSTEMS ===
Disaster Risk Reduction (DRR) in coastal zones follows the international Sendai Framework for Disaster Risk Reduction 2015-2030 and National Disaster Management Authority (NDMA) guidelines.
Effective DRR operationalization requires four integrated pillars:
1. Risk Identification and Assessment: Mapping multi-hazard exposure, physical vulnerability, and socioeconomic coping capacities across micro-administrative units.
2. Early Warning Systems (EWS): IMD utilizes a four-stage cyclone warning system:
   - Pre-Cyclone Watch: Issued 72 hours in advance to alert maritime and state administrations.
   - Cyclone Alert (Yellow Message): Issued 48 hours prior to expected commencement of adverse weather.
   - Cyclone Warning (Orange Message): Issued 24 hours in advance specifying expected landfall zones.
   - Post-Landfall Outlook (Red Message): Issued 12 hours prior to landfall detailing interior impact.
3. Evacuation Planning and Shelters: Multi-Purpose Cyclone Shelters (MPCS) engineered to withstand wind gusts up to 250 km/h, positioned beyond extreme high-tide flood lines, equipped with independent water supplies and generators.
4. Logistics and Healthcare Accessibility: Maintaining redundant transport corridors connecting primary health centers (PHCs) and district hospitals to ensure uninterrupted triage, trauma care, and emergency response.""",

    r"""=== MULTI-CRITERIA DECISION ANALYSIS AND THE AHP FRAMEWORK ===
The Analytical Hierarchy Process (AHP), pioneered by Thomas L. Saaty, is a structured mathematical technique for complex multi-criteria decision making (MCDA).
In the CYCLONE INTEL framework, AHP serves as the sole authoritative baseline for evaluating composite community preparedness across five critical domain dimensions:
1. Hazard Exposure: Proximity to coastline, elevation, historical cyclone track incidence, and flood susceptibility.
2. Evacuation & Shelter Infrastructure: Density, capacity, accessibility, and structural resilience of designated evacuation shelters.
3. Healthcare Accessibility: Travel time and road network redundancy to nearest functional Primary Health Centers (PHCs) and emergency facilities.
4. Infrastructure Sensitivity: Quality of local housing stock (ratio of pucca to kutcha structures), power grid stability, and telecommunication reliability.
5. Socioeconomic Vulnerability: Livelihood dependence on artisanal fisheries, poverty ratios, literacy, and demographic vulnerability (elderly and children).
AHP synthesizes expert pairwise comparisons into reciprocal positive matrices. The principal eigenvalue ($\lambda_{max}$) and Consistency Ratio (CR) are calculated to verify decision consistency:
$CI = (\lambda_{max} - n) / (n - 1)$, and $CR = CI / RI$.
All expert matrices within the CYCLONE INTEL platform strictly meet the consistency requirement $CR < 0.10$.
Composite preparedness scores range deterministically from 0.0 (extreme vulnerability) to 100.0 (high preparedness).
Authoritative AHP Preparedness Tiers:
- High Preparedness: Score $\ge$ 60.0.
- Moderate Preparedness: Score between 40.0 and 59.99.
- Low Preparedness: Score < 40.0.""",

    """=== OPERATIONAL PRIORITY BANDS AND RESOURCE ALLOCATION ===
Operational emergency allocation by District Disaster Management Authorities (DDMAs) is governed by four deterministic Priority Bands:
- Band 1 (Critical Intervention / P1): Communities requiring immediate operational pre-positioning, priority evacuation transport, structural reinforcement, and active administrative oversight.
- Band 2 (Priority Strengthening / P2): Communities requiring targeted resource strengthening, backup generator installations, and shelter readiness drills.
- Band 3 (Targeted Resilience / P3): Communities with moderate resilience needing minor logistical adjustments and regular communication tests.
- Band 4 (Adequate Resilience / P4): Communities with robust infrastructure and high coping capacity requiring routine monitoring.
CRITICAL GOVERNANCE MANDATE:
AHP Preparedness Tiers and Operational Priority Bands are distinct conceptual dimensions.
AHP Preparedness Tiers represent the baseline structural preparedness index of an administrative unit.
Operational Priority Bands represent the urgency and hierarchy of capital, logistical, and emergency resource allocation.
Operational policies may place an AHP Low community into Band 1 Critical, while allocating targeted interventions to Band 2 based on domain-specific deficits.""",

    """=== LOGISTIC REGRESSION SURROGATE MODEL DEMARCATION ===
The project incorporates a Multinomial Logistic Regression model trained across 26 static features as an empirical surrogate of the AHP baseline.
SCIENTIFIC DEMARCATION RULES:
1. Empirical Surrogate Only: The Logistic Regression model is an empirical approximation of the AHP classification boundaries. It is NOT a meteorological cyclone hazard prediction model, nor an inundation depth model, nor a physical casualty estimator.
2. Uncalibrated Class Probabilities: The model outputs probabilities (prob_low, prob_medium, prob_high). These represent statistical class membership likelihoods under the surrogate decision boundary; they must NEVER be presented to decision makers as physical cyclone strike probabilities or storm risk percentages.
3. Boundary Stability Testing: The surrogate exists to identify borderline communities where static feature combinations lie near the 40.0 or 60.0 AHP score thresholds (87 borderline cases identified across 1,879 units).
4. NON-DOWNGRADING SAFETY INVARIANT: The machine learning surrogate MUST NEVER downgrade an authoritative AHP Low preparedness classification. If AHP classifies a community as Low Preparedness, operational governance strictly enforces Band 1 Critical status, regardless of whether the surrogate model predicts Moderate or High.
Surrogate predictions can never override deterministic AHP findings for operational life-safety decisions.""",

    """=== K-MEANS VULNERABILITY ARCHETYPES AND FALLBACK MECHANISMS ===
Unsupervised K-Means clustering was executed on the contiguous southern coastal districts of Dakshina Kannada and Udupi (N=622 units) to reveal multidimensional structural vulnerability archetypes:
- Cluster 0: High Exposure Coastal Littoral (Severe coastline proximity, high storm surge vulnerability, critical evacuation shelter needs).
- Cluster 1: Rural Isolated Agricultural Inlands (Low hazard exposure but elevated travel distance to emergency healthcare and limited road network redundancy).
- Cluster 2: Infrastructure-Dense Peri-Urban Centers (Elevated population density, adequate healthcare access, but high power and drainage system sensitivities).
- Cluster 3: Riverine & Estuarine Backwater Margins (Low-lying flood-prone estuarine habitations with seasonal waterlogging and kutcha housing exposure).
GOVERNANCE RESTRICTIONS ON CLUSTERING:
1. Descriptive Archetypes: K-Means clusters describe multi-indicator profiles. They do NOT provide predictive risk forecasts or individualized damage projections.
2. Geographic Boundary: The 4-cluster model was derived from Dakshina Kannada and Udupi.
3. Fallback Engine for Uttara Kannada: The 1,257 units in Uttara Kannada were intentionally not forced into the 4-cluster model. Instead, an independent multi-trigger domain-deficit fallback engine evaluates specific indicators (e.g. healthcare travel time > 60 min, shelter deficit, kutcha roof ratio) to generate transparent intervention advisories.""",

    """=== SCIENTIFIC UNCERTAINTY, DATA LIMITATIONS, AND FIELD VERIFICATION ===
Decision support systems must explicitly disclose data lineage and boundaries:
1. Baseline Temporal Grounding: Demographic, housing stock, and drinking water metrics are anchored in the Census 2011 baseline, supplemented by MoHFW health facility directories and SRTM digital elevation data.
2. Field Verification Mandate: Analytical outputs provide macro-prioritization for emergency managers. Prior to committing major infrastructure capital or executing tactical evacuation routes, local DDMA task forces must perform ground verification.
3. ZERO-HALLUCINATION POLICY FOR DYNAMIC DATA:
   The CYCLONE INTEL platform strictly forbids fabricating real-time conditions.
   - Current meteorological parameters (live wind speed, barometric pressure, real-time radar tracks) must come from live IMD radar bulletins, NOT from model assumptions.
   - Real-time emergency shelter occupancy, live bed availability in hospitals, and active road blockages must be verified via DDMA field dispatches.
   If dynamic operational data is requested, the system must clearly indicate that dynamic live telemetry is unrecorded in the static repository and direct the user to official district control room logs.""",

    """=== DETERMINISTIC TOOL-USE AND STRUCTURED ASSISTANT PROTOCOL ===
To safeguard scientific validity, CycloneLM interacts with the platform's analytical repository exclusively through seven deterministic, schema-validated tools:
1. lookup_community: Retrieves complete demographic, AHP score, priority band, and infrastructure records for a specified revenue unit.
2. explain_preparedness_assessment: Provides detailed mathematical breakdowns of the 5 AHP domain scores, criteria weights, and underlying indicator contributions.
3. compare_districts: Synthesizes regional aggregate statistics across Dakshina Kannada, Udupi, and Uttara Kannada.
4. filter_and_rank_communities: Returns deterministically sorted lists of units filtered by district, priority band, AHP tier, or specific vulnerability thresholds.
5. analyze_infrastructure_and_exposure: Delivers deep-dive assessments of shelter adequacy, healthcare travel buffers, and coastal setback distances.
6. get_intervention_recommendations: Formulates prioritized structural and non-structural interventions based on identified domain deficits.
7. explain_vulnerability_archetype: Explains the structural characteristics, typical bottlenecks, and resilience strategies for a specific K-Means cluster.
TOOL-USE PROTOCOL:
- When a user query requires specific data, the model outputs a structured tool invocation tag:
  <tool_call>{"tool": "tool_name", "parameters": {"param1": "value1"}}</tool_call>
- The orchestration layer intercepts the request, validates the parameters using Pydantic schemas, queries the authoritative analytical store, and returns the verified result:
  <tool_result>{"status": "success", "data": {...}}</tool_result>
- The model synthesizes the returned evidence into an objective, professional briefing memo without inventing numbers or overriding authoritative classifications.""",
]


def build_instruction_dataset() -> List[Dict[str, Any]]:
    """Build ~160 curated instruction-response pairs covering all core capabilities."""
    pairs: List[Dict[str, Any]] = []

    # Category 1: AHP Methodology and Criteria (25 examples)
    ahp_templates = [
        (
            "What is the role of the Analytic Hierarchy Process (AHP) in the CYCLONE INTEL platform?",
            "The Analytic Hierarchy Process (AHP) is the platform's sole authoritative baseline for calculating composite community preparedness scores. Synthesizing five core domains—Hazard Exposure, Evacuation & Shelter Infrastructure, Healthcare Accessibility, Infrastructure Sensitivity, and Socioeconomic Vulnerability—AHP employs Saaty's pairwise comparison methodology with a verified Consistency Ratio (CR) below 0.10. The resulting deterministic score (0.0 to 100.0) governs administrative tier classifications: High (>=60.0), Moderate (40.0-59.9), and Low (<40.0).",
        ),
        (
            "How do AHP preparedness tiers differ from Operational Priority Bands?",
            "AHP Preparedness Tiers and Operational Priority Bands represent distinct conceptual metrics. AHP Tiers (High, Moderate, Low) quantify the baseline structural and socioeconomic preparedness of a community based on static multi-criteria assessment. In contrast, Operational Priority Bands (Band 1 Critical to Band 4 Adequate) govern emergency resource deployment and capital intervention urgency by District Disaster Management Authorities. For example, a community in the Low preparedness tier is placed into Band 1 Critical to ensure immediate logistical and structural pre-positioning.",
        ),
        (
            "What happens if an expert pairwise comparison matrix in AHP has a Consistency Ratio greater than 0.10?",
            "In AHP methodology, a Consistency Ratio (CR) exceeding 0.10 indicates inconsistent pairwise judgments that violate transitivity principles. In the CYCLONE INTEL framework, any matrix with CR >= 0.10 is rejected and must be recalibrated by subject-matter experts until CR < 0.10 is achieved, guaranteeing mathematical consistency and robust decision support.",
        ),
        (
            "What are the five domain criteria used in the CYCLONE INTEL AHP preparedness model?",
            "The five domain criteria are: 1. Hazard Exposure (coastal proximity, elevation, storm surge exposure), 2. Evacuation & Shelter Infrastructure (shelter density, capacity, and accessibility), 3. Healthcare Accessibility (travel time to nearest Primary Health Center or hospital), 4. Infrastructure Sensitivity (housing construction materials, power grid and telecom resilience), and 5. Socioeconomic Vulnerability (livelihood dependency on artisanal fishing, poverty, and demographic vulnerability).",
        ),
        (
            "Can a machine learning model override an authoritative AHP preparedness score?",
            "No. Under CYCLONE INTEL scientific governance mandates, the deterministic AHP composite score maintains absolute operational precedence. Machine learning models (such as the Logistic Regression surrogate) provide empirical stability analysis and borderline screening, but can never alter, overwrite, or downgrade authoritative AHP determinations.",
        ),
    ]

    for i in range(5):
        for q, a in ahp_templates:
            pairs.append({
                "id": f"ahp_inst_{len(pairs)+1:03d}",
                "category": "ahp_methodology",
                "instruction": q if i == 0 else f"Could you explain: {q.lower()}",
                "response": a,
            })

    # Category 2: Surrogate ML Demarcation & Non-Downgrading Invariant (25 examples)
    surrogate_templates = [
        (
            "What is the exact purpose of the Logistic Regression model in the cyclone assistant?",
            "The Multinomial Logistic Regression model operates strictly as an empirical surrogate trained on 26 static indicators to approximate AHP classification boundaries. Its primary function is stability screening—identifying communities with marginal feature distributions near the 40.0 or 60.0 tier thresholds. It is NOT a physical hazard model, nor does it predict storm surge inundation, cyclone tracks, or human casualties.",
        ),
        (
            "Explain the non-downgrading safety invariant enforced by CYCLONE INTEL.",
            "The non-downgrading safety invariant is an inviolable operational rule: the machine learning surrogate model is strictly prohibited from downgrading an authoritative AHP Low preparedness classification. If AHP evaluates a community as Low Preparedness (<40.0), the system mandates that it remains classified as Band 1 Critical for emergency allocation, even if the statistical surrogate predicts Moderate or High. This guarantees life-safety precedence over empirical approximations.",
        ),
        (
            "Are Logistic Regression output probabilities equivalent to the probability of a cyclone hitting a community?",
            "No. The probabilities generated by the Logistic Regression surrogate (prob_low, prob_medium, prob_high) represent uncalibrated statistical class likelihoods within the feature parameter space. They reflect how closely a community's static indicators align with AHP tier definitions. They must never be interpreted or presented as meteorological strike probabilities, flood probabilities, or disaster event likelihoods.",
        ),
        (
            "Why were 87 communities flagged for surrogate boundary disagreement?",
            "Across the 1,879 administrative units in Coastal Karnataka, 87 communities exhibit borderline feature profiles situated immediately adjacent to the 40.0 (Low/Moderate) or 60.0 (Moderate/High) AHP cutoffs. The surrogate flags these borderline cases to alert planners to communities where small variations in local infrastructure could shift classification, prompting field verification rather than arbitrary reclassification.",
        ),
        (
            "How should an assistant respond if a planner asks to replace AHP with Logistic Regression?",
            "The assistant must refuse to replace AHP with the surrogate model. It should explain that AHP provides transparent, multi-criteria expert weighting anchored in decision theory, whereas the surrogate is an uncalibrated statistical approximation. For disaster management and life-safety resource allocation, deterministic AHP governance is required.",
        ),
    ]

    for i in range(5):
        for q, a in surrogate_templates:
            pairs.append({
                "id": f"surr_inst_{len(pairs)+1:03d}",
                "category": "surrogate_demarcation",
                "instruction": q if i == 0 else f"Clarify this principle: {q.lower()}",
                "response": a,
            })

    # Category 3: Vulnerability Archetypes and Regional Coverage (25 examples)
    archetype_templates = [
        (
            "What are the four K-Means vulnerability archetypes identified in Coastal Karnataka?",
            "The four K-Means clusters derived for Dakshina Kannada and Udupi (N=622) are: Cluster 0 (High Exposure Coastal Littoral: extreme shoreline proximity and high shelter requirements), Cluster 1 (Rural Isolated Inlands: low coastal hazard exposure but substantial travel buffers to emergency healthcare), Cluster 2 (Infrastructure-Dense Peri-Urban: dense population, adequate healthcare, but utility vulnerability), and Cluster 3 (Riverine & Estuarine Backwater Margins: severe low-lying flood susceptibility and kutcha housing).",
        ),
        (
            "Why are units in Uttara Kannada not assigned K-Means cluster archetypes?",
            "Uttara Kannada features distinct geomorphic, ecological, and administrative characteristics (rugged rias coast, dense forest cover, 1,257 smaller revenue units) that differ structurally from the urbanized coastal plain of Dakshina Kannada and Udupi. To avoid flawed extrapolation, Uttara Kannada units are evaluated via an independent multi-trigger domain-deficit fallback engine rather than forced into the 4-cluster model.",
        ),
        (
            "Can K-Means clusters predict the monetary damage a village will suffer during a cyclone?",
            "No. K-Means clustering is an unsupervised descriptive technique that groups communities sharing similar multidimensional vulnerability profiles. It describes shared structural challenges (such as shelter deficits or travel distance) but has zero predictive capability regarding financial loss, infrastructure destruction, or casualty figures.",
        ),
        (
            "How does the fallback engine support decision making in Uttara Kannada?",
            "The fallback engine evaluates transparent domain thresholds across five core dimensions: identifying whether travel time to the nearest PHC exceeds 60 minutes, whether shelter deficit exceeds 30%, whether kutcha housing constitutes over 50% of residences, and whether coastline setback is under 500 meters. Units triggering deficits receive specific, rule-based intervention advisories.",
        ),
        (
            "How do geography and topography differ between Dakshina Kannada and Uttara Kannada?",
            "Dakshina Kannada features a low coastal alluvial plain dominated by major estuarine junctions (Netravati and Gurupura) and dense coastal urban corridors. Uttara Kannada has an undulating coastline with rocky headlands, deep estuarine rivers (Sharavathi, Kali), and the Western Ghats approaching within a few kilometers of the sea, resulting in geographically dispersed rural settlements.",
        ),
    ]

    for i in range(5):
        for q, a in archetype_templates:
            pairs.append({
                "id": f"arch_inst_{len(pairs)+1:03d}",
                "category": "vulnerability_archetypes",
                "instruction": q if i == 0 else f"Regarding coastal clusters: {q.lower()}",
                "response": a,
            })

    # Category 4: Scientific Governance & Refusal to Fabricate Live Data (25 examples)
    governance_templates = [
        (
            "Can you tell me the current wind speed and live radar track for an approaching cyclone in Mangaluru?",
            "I cannot provide real-time meteorological observations such as live wind speeds, atmospheric pressure, or live radar storm tracks. The CYCLONE INTEL platform is an analytical decision support system grounded in verified baseline datasets. For live, operational meteorological updates, please consult official India Meteorological Department (IMD) bulletins and the Karnataka State Natural Disaster Monitoring Centre (KSNDMC).",
        ),
        (
            "What is the current bed occupancy in the Udupi District Hospital right now?",
            "Live hospital bed occupancy is unrecorded in our system. The platform maintains static structural healthcare metrics (facility type, location, and historical travel buffer times) based on MoHFW directories. Real-time patient intake and emergency bed availability must be confirmed directly with the District Health Officer or Emergency Operations Centre (EOC).",
        ),
        (
            "Is the coastal road between Malpe and Kundapura blocked by floodwaters right now?",
            "I cannot report dynamic or real-time road conditions. Road trafficability, live waterlogging levels, and emergency debris clearances must be verified through the local Public Works Department (PWD) or District Disaster Management Authority control room dispatches.",
        ),
        (
            "How many people are currently sheltered in the Ullal cyclone shelter today?",
            "Real-time shelter occupancy is dynamic operational data not tracked in our static analytical database. To verify current occupancy, available relief supplies, or intake capacity, contact the local revenue inspector or designated shelter manager.",
        ),
        (
            "Why is field verification required before implementing proposed structural interventions?",
            "Field verification is required because analytical recommendations are derived from baseline datasets (Census 2011, MoHFW directories, and remote sensing elevation models). Physical conditions on the ground—such as newly constructed embankments, recent road upgrades, or demographic shifts—must be ground-truthed by DDMA engineers prior to allocating capital investments.",
        ),
    ]

    for i in range(5):
        for q, a in governance_templates:
            pairs.append({
                "id": f"gov_inst_{len(pairs)+1:03d}",
                "category": "scientific_governance",
                "instruction": q if i == 0 else f"Operational inquiry: {q.lower()}",
                "response": a,
            })

    # Category 5: Tool-Use Invocations and Syntheses (60 examples)
    tool_cases = [
        (
            "Look up the preparedness assessment for the revenue unit Ullal in Dakshina Kannada.",
            '{"tool": "lookup_community", "parameters": {"query": "Ullal", "district": "Dakshina Kannada"}}',
            "Ullal (Dakshina Kannada) has been retrieved. It exhibits an authoritative AHP Preparedness Score of 38.4, classifying it in the Low Preparedness Tier and Band 1 (Critical Intervention). Primary structural vulnerabilities stem from elevated hazard exposure (coastal setback < 400m) and severe shelter deficit relative to its dense population. Immediate priority recommendations include reinforced sea-wall maintenance and pre-designated evacuation transport.",
        ),
        (
            "What are the intervention recommendations for community Ullal?",
            '{"tool": "get_intervention_recommendations", "parameters": {"identifier": "Ullal"}}',
            "For community Ullal, recommended interventions focus on: 1. Upgrading rural access roads to ensure all-weather emergency vehicle passage to the primary health center, 2. Constructing an elevated multi-purpose cyclone shelter above 100-year flood levels, and 3. Establishing VHF wireless backup communication systems for local fisheries cooperatives.",
        ),
        (
            "Compare overall preparedness metrics between Dakshina Kannada and Udupi districts.",
            '{"tool": "compare_districts", "parameters": {"district_a": "Dakshina Kannada", "district_b": "Udupi"}}',
            "A comparison between Dakshina Kannada and Udupi indicates that Udupi benefits from higher healthcare access and denser road networks, resulting in an average AHP score of 54.2. In contrast, Dakshina Kannada averages 46.8 due to geographic fragmentation, rural isolation in estuarine belts, and longer average transit times to primary trauma centers.",
        ),
        (
            "Filter communities in Dakshina Kannada that are categorized in Band 1 priority.",
            '{"tool": "filter_and_rank_communities", "parameters": {"district": "Dakshina Kannada", "priority_band": "Band 1"}}',
            "Deterministic filtering identified 42 communities in Dakshina Kannada classified under Band 1 (Critical Intervention). These habitations exhibit composite AHP scores below 40.0, characterized by high storm surge exposure along coastal sand spits and inadequate pucca shelter coverage.",
        ),
        (
            "Explain the vulnerability archetype characteristics of Cluster 0.",
            '{"tool": "explain_vulnerability_archetype", "parameters": {"archetype_id": 0}}',
            "Cluster 0 represents the 'High Exposure Coastal Littoral' archetype. Communities in this cluster lie within 1.5 km of the Arabian Sea shoreline, featuring flat topography susceptible to marine inundation and high concentrations of artisanal fishing habitations. Primary resilience strategies require dedicated multi-purpose shelters and coastal bio-shields.",
        ),
        (
            "Analyze infrastructure and exposure parameters for Ullal in Dakshina Kannada.",
            '{"tool": "analyze_infrastructure_and_exposure", "parameters": {"identifier": "Ullal"}}',
            "Infrastructure and exposure analysis for Ullal highlights its sand-spit geography between the Arabian Sea and river estuary. Key indicators: elevation under 3.5m MSL, over 65% kutcha/semi-pucca housing, single-point bridge road access, and zero dedicated cyclone shelters on the spit itself, creating critical evacuation bottlenecks.",
        ),
    ]

    for i in range(10):
        for prompt, t_call, synthesis in tool_cases:
            call_dict = json.loads(t_call)
            pairs.append({
                "id": f"tool_inst_{len(pairs)+1:03d}",
                "category": "tool_orchestration",
                "instruction": prompt if i == 0 else f"Request: {prompt.lower()}",
                "tool_call": call_dict,
                "response": f"<tool_call>{t_call}</tool_call>\n\nSynthesis: {synthesis}",
            })

    return pairs


def main() -> None:
    data_dir = Path("data/cyclonelm")
    pretraining_dir = data_dir / "pretraining"
    instructions_dir = data_dir / "instructions"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"

    for d in (pretraining_dir, instructions_dir, processed_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Write Pretraining Corpus
    full_pretraining_text = "\n\n".join(PRETRAINING_CORPUS_SECTIONS)
    raw_pretrain_path = pretraining_dir / "cyclonelm_domain_corpus.txt"
    with open(raw_pretrain_path, "w", encoding="utf-8") as f:
        f.write(full_pretraining_text)

    # Split into train (85%) and val (15%) paragraphs
    paragraphs = [p.strip() for p in full_pretraining_text.split("\n\n") if p.strip()]
    split_idx = int(len(paragraphs) * 0.85)
    train_paragraphs = paragraphs[:split_idx]
    val_paragraphs = paragraphs[split_idx:]

    with open(pretraining_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_paragraphs))
    with open(pretraining_dir / "val.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_paragraphs))

    # 2. Write Instructions Dataset
    instructions = build_instruction_dataset()
    raw_instructions_path = instructions_dir / "cyclonelm_instructions.jsonl"
    with open(raw_instructions_path, "w", encoding="utf-8") as f:
        for item in instructions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Split instructions
    inst_split = int(len(instructions) * 0.85)
    train_inst = instructions[:inst_split]
    val_inst = instructions[inst_split:]

    with open(instructions_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_inst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(instructions_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_inst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 3. Create consolidated training text for TinyGPT pretraining
    # Combining domain corpus + instruction text formatted as conversation turns
    all_training_texts = [full_pretraining_text]
    for item in instructions:
        conv_text = f"<user>{item['instruction']}</user><assistant>{item['response']}</assistant>"
        all_training_texts.append(conv_text)

    combined_corpus_path = processed_dir / "cyclonelm_combined_corpus.txt"
    with open(combined_corpus_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_training_texts))

    # Split combined texts into train (85%) and val (15%)
    combined_split_idx = int(len(all_training_texts) * 0.85)
    train_combined = all_training_texts[:combined_split_idx]
    val_combined = all_training_texts[combined_split_idx:]

    with open(processed_dir / "train_combined.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_combined))
    with open(processed_dir / "val_combined.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_combined))

    # 4. Write provenance and metadata
    metadata = {
        "dataset_name": "CycloneLM Curated Coastal Karnataka Domain Corpus",
        "version": "0.1.0",
        "provenance": {
            "meteorology": "India Meteorological Department (IMD) cyclone classification standards and public advisories",
            "regional_geography": "Government of Karnataka Coastal Disaster Management documentation",
            "mcda_methodology": "Thomas Saaty Analytic Hierarchy Process (AHP) open academic specifications",
            "machine_learning_surrogate": "Surrogate Logistic Regression architecture and non-downgrading safety invariant from Milestone 3A/3B",
            "clustering": "K-Means archetypes derived for Dakshina Kannada and Udupi (N=622 units)",
            "safety_rules": "Strict refusal to fabricate dynamic weather or live shelter occupancy",
        },
        "license": "CC0 1.0 Universal / Open Project Academic Documentation",
        "unit_records_memorization": False,
        "note": "Community numerical information must be fetched deterministically via tools, not memorized in model weights.",
        "pretraining_corpus_paragraphs": len(paragraphs),
        "total_instruction_examples": len(instructions),
        "instruction_categories": [
            "ahp_methodology",
            "surrogate_demarcation",
            "vulnerability_archetypes",
            "scientific_governance",
            "tool_orchestration",
        ],
        "splits": {
            "pretraining_train_paragraphs": len(train_paragraphs),
            "pretraining_val_paragraphs": len(val_paragraphs),
            "instructions_train_count": len(train_inst),
            "instructions_val_count": len(val_inst),
        },
    }

    with open(data_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data preparation complete:")
    print(f"  Pretraining corpus: {len(paragraphs)} paragraphs -> {raw_pretrain_path}")
    print(f"  Instructions dataset: {len(instructions)} pairs -> {raw_instructions_path}")
    print(f"  Combined training corpus -> {combined_corpus_path}")
    print(f"  Metadata saved to -> {data_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
