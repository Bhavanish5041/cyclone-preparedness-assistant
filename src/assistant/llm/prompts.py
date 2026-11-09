"""System prompts, scientific guardrails, and deterministic fallback templates.

Enforces:
- Strict closed-domain reasoning grounded exclusively in deterministic tool outputs.
- Authoritative operational precedence of deterministic AHP preparedness scores and tiers.
- Demarcation of Logistic Regression as an empirical classification surrogate (not physical hazard risk).
- Disclosure of historical Census 2011 and MoHFW directory limitations.
- Zero hallucination of real-time operational or meteorological parameters.
"""

from typing import Any, Dict, List, Optional

SCIENTIFIC_PRECEDENCE_DISCLAIMER = (
    "AHP PRECEDENCE MANDATE: Deterministic AHP Preparedness Scores and 4-band priority classifications "
    "maintain absolute operational precedence for disaster preparedness and emergency resource allocation. "
    "Logistic Regression class probabilities represent an empirical tier surrogate approximating AHP classification boundaries; "
    "they are uncalibrated class likelihoods and do NOT represent physical cyclone-risk, inundation depth, "
    "or casualty forecasts. Infrastructure and demographic indicators reflect the 2011 Census historical baseline."
)

SYSTEM_INSTRUCTION = """You are the AI Cyclone Preparedness Technical Assistant for coastal Karnataka (covering Dakshina Kannada, Udupi, and Uttara Kannada across 1,879 administrative units).

Your duty is to assist District Disaster Management Authorities (DDMAs) and emergency planners by retrieving verified, deterministic preparedness assessments and formulating objective, grounded briefing memos.

STRICT SCIENTIFIC AND OPERATIONAL RULES:

1. CLOSED-WORLD EVIDENCE GROUNDING:
   - Base your statements, scores, ranks, and recommendations EXCLUSIVELY on the data returned by the provided tools.
   - Do NOT invent external metrics, hospital bed counts, emergency supplies, or casualty figures.
   - If information is unrecorded or unavailable in the tool output, explicitly state that it is unrecorded.

2. AHP ABSOLUTE OPERATIONAL PRECEDENCE:
   - The deterministic Analytical Hierarchy Process (AHP) composite preparedness index (0.0 - 100.0) and its 4-band priority hierarchy (Band 1 Critical, Band 2 Specialized, Band 3 Moderate, Band 4 Adequate) govern all operational decision support.
   - You can NEVER alter, recalculate, or override an AHP score or priority band.
   - If an AHP tier is Low and the Logistic Regression surrogate predicts Medium or High, operational governance mandates that the community remains Band 1 Critical.

3. SURROGATE ML DEMARCATION:
   - Frame the Multinomial Logistic Regression model strictly as an "empirical tier surrogate" trained on 26 static features to approximate AHP tiers.
   - NEVER refer to model probabilities (prob_low, prob_medium, prob_high) as physical cyclone strike, storm surge, flood, or casualty probabilities.
   - Explain borderline disagreements (87 cases across 1,879 units) as classification sensitivity near the 40.0 or 60.0 AHP cutoff thresholds.

4. DESCRIPTIVE VULNERABILITY ARCHETYPES:
   - The 4 K-Means clusters are descriptive structural groupings for Dakshina Kannada and Udupi (N=622).
   - They provide contextual background on shared structural profiles; they do not imply individual unverified deficiencies.

5. UTTARA KANNADA DETERMINISTIC FALLBACK:
   - The 1,257 revenue units of Uttara Kannada are unassigned in K-Means clustering.
   - They are evaluated strictly via deterministic AHP domain scores and the audited 3-sector fallback rules (coastal proximity < 5 km, shelter score < 20, medical score < 35). Explain this transparently when discussing Uttara Kannada.

6. CENSUS 2011 AND DIRECTORY BASELINE CAVEATS:
   - Explicitly disclose that demographic counts, household numbers, water access, and road connectivity reflect the 2011 Census baseline (over 15 years old).
   - Note that healthcare facility registration does NOT guarantee active doctor staffing, emergency bed availability, or functioning generator diesel during power outages.
   - HEALTHCARE METRIC DISAMBIGUATION: Clearly distinguish between Primary Health Centres (PHC, registered 24/7 basic medical centres, e.g. 3.09 km for Tallur), local primary health sub-centres (in-village clinics, 0.00 km if located inside the village boundary), and secondary/tertiary referral apex hospitals (measured in a separate radius). Never conflate a local sub-centre with a full Primary Health Centre or apex hospital.

7. ZERO REAL-TIME OPERATIONAL INVENTION:
   - NEVER invent live weather, wind speeds, or cyclone track/landfall forecasts (the India Meteorological Department holds exclusive legal jurisdiction).
   - NEVER invent real-time shelter occupancy, boat availability, or hospital staffing.
   - Never claim to have dispatched teams, issued evacuation decrees, or contacted field officers.

8. TOOL INVOCATION WORKFLOW:
   - When asked about a specific community, use `lookup_community` if the exact ID is unknown, then `explain_preparedness_assessment`, `analyze_infrastructure_and_exposure`, or `get_intervention_recommendations`.
   - When asked to compare districts, use `compare_districts`.
   - When asked to rank or filter, use `filter_and_rank_communities`.
   - When asked about archetypes or cluster traits, use `explain_vulnerability_archetype`.
   - Synthesize all returned metrics clearly with appropriate units (km, %, score points).
"""


def format_deterministic_fallback_response(
    query: str,
    reason: str,
    community_eval: Optional[Dict[str, Any]] = None,
    advisories: Optional[List[Dict[str, Any]]] = None,
    infra_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a transparent briefing when external LLM is unavailable."""
    return format_community_assessment_response(community_eval, advisories, query, reason, infra_data)


def format_community_assessment_response(
    community_eval: Optional[Dict[str, Any]],
    advisories: Optional[List[Dict[str, Any]]],
    query: str,
    reason: str = "Deterministic evaluation mode",
    infra_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a community preparedness evaluation briefing from deterministic tool outputs."""
    sections = [
        "### Automated Decision Support Briefing (Deterministic Mode)",
        "",
        f"> **System Notice:** Direct local decision-support evaluation active ({reason}). "
        "All metrics retrieved from authoritative AHP calculations and historical Census baselines.",
        "",
    ]

    if community_eval:
        comm_name = community_eval.get("community_name") or community_eval.get("name", "Unknown Unit")
        pc11_id = community_eval.get("pc11_id", "N/A")
        shrid2 = community_eval.get("shrid2", "N/A")
        district = community_eval.get("district_name") or community_eval.get("district", "N/A")
        taluk = community_eval.get("subdistrict_name") or community_eval.get("taluk", "N/A")

        ahp = community_eval.get("deterministic_ahp", {})
        score = ahp.get("preparedness_score", community_eval.get("ahp_score", 0.0))
        tier = ahp.get("preparedness_tier", community_eval.get("preparedness_tier", "Unknown"))

        prioritization = community_eval.get("operational_prioritization", {})
        band = prioritization.get("priority_band", community_eval.get("priority_band", "BAND_3_MODERATE"))
        overall_rank = prioritization.get("overall_priority_rank", "N/A")
        band_rank = prioritization.get("band_rank", "N/A")
        explanation = prioritization.get("priority_explanation", "")

        sections.extend([
            f"#### Community Profile: {comm_name} ({district})",
            f"- **Census PC11 ID:** `{pc11_id}` | **SHRID:** `{shrid2}` | **Taluk:** {taluk.strip() if isinstance(taluk, str) else taluk}",
            f"- **Deterministic AHP Preparedness Score:** **{score:.2f} / 100.0** ({tier} Tier)",
            f"- **Assigned Priority Band:** `{band}` (Overall Priority Rank: #{overall_rank}, Band Rank: #{band_rank})",
            "",
            "**Priority Justification and Supporting Indicators:**",
            explanation if explanation else "Evaluation determined by multi-criteria infrastructure, health, and demographic indicators.",
            "",
        ])

        # Domain breakdown if present
        domain_scores = ahp.get("domain_scores", {})
        if domain_scores:
            def _fmt_score(*keys: str) -> str:
                for k in keys:
                    val = domain_scores.get(k)
                    if val is not None and isinstance(val, (int, float)):
                        return f"{val:.1f} / 100.0"
                return "N/A"

            sections.extend([
                "**AHP Domain Sub-Scores:**",
                f"- Primary Emergency Shelters: {_fmt_score('score_shelters', 'shelters')}",
                f"- Primary Healthcare Access: {_fmt_score('score_medical', 'medical')}",
                f"- Infrastructure & Physical Connectivity: {_fmt_score('score_infrastructure', 'infrastructure')}",
                f"- Early Warning & Communication: {_fmt_score('score_early_warning', 'early_warning')}",
                f"- Socioeconomic Vulnerability Baseline: {_fmt_score('score_socioeconomic', 'socioeconomic', 'demographic')}",
                "",
            ])

        # Granular Infrastructure and Physical Exposure if present
        if infra_data and isinstance(infra_data, dict):
            phys = infra_data.get("physical_exposure", {})
            hc = infra_data.get("healthcare_access", {})
            sh = infra_data.get("shelter_and_transit", {})
            amen = infra_data.get("amenities_and_demographics", {})
            ops = infra_data.get("operational_limitations", {})

            sections.extend([
                "#### Granular Infrastructure & Physical Exposure (Census 2011 Baseline):",
                f"- **Coastal Proximity:** {phys.get('dist_coast_km', 'N/A')} km | **Mean Elevation:** {phys.get('elevation_mean_m', 'N/A')} m",
                f"- **Healthcare Proximity:** Nearest Primary Health Centre (PHC): {hc.get('dist_nearest_phc_km', 'N/A')} km | Local Health Sub-centre: {hc.get('dist_nearest_local_facility_km', '0.00')} km | Nearest Apex Hospital: {hc.get('dist_nearest_hospital_km', 'N/A')} km ({hc.get('hospitals_within_10km', 0)} hospitals within 10 km)",
                f"- *Healthcare Facility Disambiguation:* PHC distance measures access to registered 24/7 Primary Health Centres ({hc.get('dist_nearest_phc_km', 'N/A')} km). Local primary sub-centres within settlement boundaries register as 0.00 km, while secondary taluk and tertiary referral apex hospitals are measured independently.",
                f"- **Emergency Shelters:** Nearest shelter: {sh.get('dist_nearest_shelter_km', 'N/A')} km ({sh.get('shelters_within_10km', 0)} shelters within 10 km, total capacity: {sh.get('shelter_capacity_nearby', 'N/A')})",
                f"- **Transit & Connectivity:** All-weather pucca road: {'Yes' if sh.get('has_pucca_road') else 'No'} | Bus connectivity: {'Yes' if sh.get('bus_connectivity') else 'No'} | Road connectivity score: {sh.get('road_connectivity_score', 'N/A')}",
                f"- **Demographics & Amenities:** Population: {amen.get('population', 'N/A')} | Households: {amen.get('households', 'N/A')} | Drinking water score: {amen.get('drinking_water_score', 'N/A')}",
                f"- *Operational Baseline Caveat:* {ops.get('operational_caveat', 'Physical facilities represent registration, not active staffing during cyclonic disruption.')}",
                "",
            ])

        if advisories:
            sections.append("#### Evidence-Grounded Field Advisories:")
            for idx, adv in enumerate(advisories, 1):
                focus = adv.get("advisory_focus", "Operational Focus")
                rec = adv.get("operational_recommendation", "")
                agency = adv.get("lead_coordinating_agency", "DDMA")
                code = adv.get("directive_id", f"ADV_{idx}")
                sections.append(
                    f"{idx}. **[{code}] {focus}:** {rec}  \n"
                    f"   *Coordinating Authority:* {agency}"
                )
            sections.append("")

        # Surrogate ML Demarcation
        ml_data = community_eval.get("surrogate_ml") or community_eval.get("ml_surrogate_assessment")
        if ml_data and isinstance(ml_data, dict):
            pred_tier = ml_data.get("predicted_preparedness_tier") or ml_data.get("tier", "N/A")
            sections.extend([
                "#### Surrogate ML Demarcation:",
                f"- **Logistic Regression Predicted Tier:** `{pred_tier}`",
                "- *Scientific Precaution:* Logistic Regression outputs serve strictly as an empirical tier surrogate approximating macro-feature boundaries. They are uncalibrated class likelihoods and do NOT represent physical cyclone strike, storm-surge inundation, or casualty forecasts.",
                "",
            ])

    else:
        sections.extend([
            f"**Inquiry:** {query}",
            "",
            "Direct assessment data for a specific community was not retrieved. "
            "Please provide a specific revenue village name, Census PC11 ID (e.g. `608691`), or SHRID to inspect granular indicators.",
            "",
        ])

    sections.extend([
        "---",
        f"*{SCIENTIFIC_PRECEDENCE_DISCLAIMER}*",
    ])
    return "\n".join(sections)


def format_district_comparison_response(
    comp_data: Dict[str, Any],
    query: str,
) -> str:
    """Format comparative district cross-tabulation response from compare_districts tool output."""
    districts = comp_data.get("districts_compared", ["District A", "District B"])
    dist_a = districts[0] if len(districts) > 0 else "District A"
    dist_b = districts[1] if len(districts) > 1 else "District B"

    comparison = comp_data.get("comparison", {})
    metrics_a = comparison.get(dist_a, {})
    metrics_b = comparison.get(dist_b, {})
    key_obs = comp_data.get("key_observations", [])
    method_note = comp_data.get("methodological_note", "")

    total_a = metrics_a.get("total_units", 0)
    total_b = metrics_b.get("total_units", 0)

    # Extract priority counts
    bands_a = metrics_a.get("priority_bands", {})
    bands_b = metrics_b.get("priority_bands", {})

    b1_a = bands_a.get("BAND_1_CRITICAL", 0)
    b1_a_pct = (b1_a / total_a * 100) if total_a else 0.0
    b1_b = bands_b.get("BAND_1_CRITICAL", 0)
    b1_b_pct = (b1_b / total_b * 100) if total_b else 0.0

    b2_a = bands_a.get("BAND_2_SPECIALIZED", 0)
    b2_a_pct = (b2_a / total_a * 100) if total_a else 0.0
    b2_b = bands_b.get("BAND_2_SPECIALIZED", 0)
    b2_b_pct = (b2_b / total_b * 100) if total_b else 0.0

    b3_a = bands_a.get("BAND_3_MODERATE", 0)
    b3_a_pct = (b3_a / total_a * 100) if total_a else 0.0
    b3_b = bands_b.get("BAND_3_MODERATE", 0)
    b3_b_pct = (b3_b / total_b * 100) if total_b else 0.0

    b4_a = bands_a.get("BAND_4_ADEQUATE", 0)
    b4_a_pct = (b4_a / total_a * 100) if total_a else 0.0
    b4_b = bands_b.get("BAND_4_ADEQUATE", 0)
    b4_b_pct = (b4_b / total_b * 100) if total_b else 0.0

    avg_score_a_val = metrics_a.get("mean_preparedness_score", "N/A")
    avg_score_b_val = metrics_b.get("mean_preparedness_score", "N/A")
    avg_coast_a_val = metrics_a.get("mean_dist_coast_km", "N/A")
    avg_coast_b_val = metrics_b.get("mean_dist_coast_km", "N/A")

    score_a_str = f"{avg_score_a_val:.2f}" if isinstance(avg_score_a_val, (int, float)) else str(avg_score_a_val)
    score_b_str = f"{avg_score_b_val:.2f}" if isinstance(avg_score_b_val, (int, float)) else str(avg_score_b_val)
    coast_a_str = f"{avg_coast_a_val:.2f}" if isinstance(avg_coast_a_val, (int, float)) else str(avg_coast_a_val)
    coast_b_str = f"{avg_coast_b_val:.2f}" if isinstance(avg_coast_b_val, (int, float)) else str(avg_coast_b_val)

    sections = [
        f"### District Comparative Analysis: {dist_a} vs. {dist_b} (Deterministic Mode)",
        "",
        "> **System Notice:** Direct local decision-support evaluation active. Metrics retrieved from verified cross-tabulations across surveyed coastal units.",
        "",
    ]

    # Specific follow-up query focus handling
    q_lower = query.lower()
    if any(k in q_lower for k in ["more band 1", "which district has more band", "band 1 communities", "band 1 critical"]):
        if b1_a > b1_b:
            more_d, less_d, diff_units = dist_a, dist_b, b1_a - b1_b
            sections.extend([
                "#### Direct Inquiry Focus: Band 1 Critical Allocation",
                f"**{more_d}** has more Band 1 Critical communities ({b1_a} units, {b1_a_pct:.1f}%) compared to **{less_d}** ({b1_b} units, {b1_b_pct:.1f}%), a disparity of {diff_units} units.",
                "",
            ])
        elif b1_b > b1_a:
            more_d, less_d, diff_units = dist_b, dist_a, b1_b - b1_a
            sections.extend([
                "#### Direct Inquiry Focus: Band 1 Critical Allocation",
                f"**{more_d}** has more Band 1 Critical communities ({b1_b} units, {b1_b_pct:.1f}%) compared to **{less_d}** ({b1_a} units, {b1_a_pct:.1f}%), a disparity of {diff_units} units.",
                "",
            ])
        else:
            sections.extend([
                "#### Direct Inquiry Focus: Band 1 Critical Allocation",
                f"Both **{dist_a}** and **{dist_b}** have an identical number of Band 1 Critical units ({b1_a} units).",
                "",
            ])
    elif any(k in q_lower for k in ["difference in their average ahp", "average ahp scores", "difference in", "average score", "score difference"]):
        if isinstance(avg_score_a_val, (int, float)) and isinstance(avg_score_b_val, (int, float)):
            diff_score = abs(avg_score_a_val - avg_score_b_val)
            higher_d = dist_a if avg_score_a_val > avg_score_b_val else dist_b
            lower_d = dist_b if avg_score_a_val > avg_score_b_val else dist_a
            sections.extend([
                "#### Direct Inquiry Focus: Average AHP Score Disparity",
                f"**{higher_d}** exhibits a higher average AHP preparedness score ({max(avg_score_a_val, avg_score_b_val):.2f}) than **{lower_d}** ({min(avg_score_a_val, avg_score_b_val):.2f}), representing a difference of **{diff_score:.2f} points**.",
                "",
            ])

    sections.extend([
        "#### Summary Metrics",
        f"| Preparedness Indicator | {dist_a} | {dist_b} |",
        "|---|---|---|",
        f"| Total Surveyed Units | **{total_a}** | **{total_b}** |",
        f"| Average AHP Preparedness Score | **{score_a_str} / 100.0** | **{score_b_str} / 100.0** |",
        f"| Average Distance to Coastline | {coast_a_str} km | {coast_b_str} km |",
        f"| Band 1: Critical Priority | {b1_a} ({b1_a_pct:.1f}%) | {b1_b} ({b1_b_pct:.1f}%) |",
        f"| Band 2: Specialized Readiness | {b2_a} ({b2_a_pct:.1f}%) | {b2_b} ({b2_b_pct:.1f}%) |",
        f"| Band 3: Moderate Priority | {b3_a} ({b3_a_pct:.1f}%) | {b3_b} ({b3_b_pct:.1f}%) |",
        f"| Band 4: Adequate Baseline | {b4_a} ({b4_a_pct:.1f}%) | {b4_b} ({b4_b_pct:.1f}%) |",
        "",
    ])

    if key_obs:
        sections.append("#### Key Disparities and Observations:")
        for obs in key_obs:
            sections.append(f"- {obs}")
        sections.append("")

    if method_note:
        sections.extend([
            "#### Methodological Context:",
            method_note,
            "",
        ])

    sections.extend([
        "---",
        f"*{SCIENTIFIC_PRECEDENCE_DISCLAIMER}*",
    ])
    return "\n".join(sections)


def format_band_filter_response(
    filter_data: Dict[str, Any],
    query: str,
) -> str:
    """Format priority band filter roster response from filter_and_rank_communities tool output."""
    filters = filter_data.get("applied_filters", {})
    band = filters.get("priority_band") or filter_data.get("priority_band") or "Filtered Roster"
    total_matching = filter_data.get("total_matching", filter_data.get("total_matching_units", 0))
    returned_count = filter_data.get("returned_count", 0)
    units = filter_data.get("units") or filter_data.get("communities", [])

    sections = [
        f"### Priority Band Roster: {band} (Deterministic Mode)",
        "",
        "> **System Notice:** Direct local decision-support evaluation active. Units retrieved from authoritative multi-criteria ranking.",
        "",
        f"- **Filter Applied:** Priority Band = `{band}`",
        f"- **Total Matching Units Across Study Area:** **{total_matching}** (showing top {returned_count} ranked by priority)",
        "",
        "#### Top Ranked Units for Pre-Landfall Focus",
        "| Overall Rank | Village Name | District | Taluk | Census PC11 ID | AHP Score | Band Rank |",
        "|---|---|---|---|---|---|---|",
    ]

    for comm in units:
        rank = comm.get("overall_priority_rank", "N/A")
        name = comm.get("village_name", "Unknown")
        dist = comm.get("district_name", "N/A")
        taluk = comm.get("subdistrict_name", "N/A")
        if isinstance(taluk, str):
            taluk = taluk.strip()
        pc11 = comm.get("pc11_id", "N/A")
        score = comm.get("preparedness_score", 0.0)
        b_rank = comm.get("band_rank", "N/A")
        sections.append(
            f"| #{rank} | **{name}** | {dist} | {taluk} | `{pc11}` | **{score:.2f}** | #{b_rank} |"
        )

    sections.extend([
        "",
        "---",
        f"*{SCIENTIFIC_PRECEDENCE_DISCLAIMER}*",
    ])
    return "\n".join(sections)


def format_uttara_kannada_fallback_response(
    arch_data: Dict[str, Any],
    query: str,
    regional_summary: Optional[Any] = None,
) -> str:
    """Format briefing on Uttara Kannada unassigned status and 3-sector fallback engine."""
    sections = [
        "### Uttara Kannada Deterministic Fallback Methodology (Audit Briefing)",
        "",
        "> **System Notice:** Direct architectural briefing regarding the audited 3-sector fallback engine for unassigned northern coastal units.",
        "",
        "#### Study Area Boundary and Data Availability",
        "- **Clustered Region (K-Means, N=622):** Dakshina Kannada (369 units) and Udupi (253 units) form the primary spatial clustering study area with complete historical sub-district census covariates.",
        "- **Unassigned Region (N=1,257):** The 1,257 administrative units of Uttara Kannada were excluded from unsupervised K-Means clustering because historical census covariates were unassigned at the sub-district tier.",
        "",
        "#### Audited Three-Sector Fallback Engine",
        "To prevent operational planning gaps and ensure rigorous decision support, unassigned Uttara Kannada units are evaluated using authoritative AHP domain scores and the audited three-sector fallback engine:",
        "1. **Coastal Hazard Proximity Trigger:** Distance to coastline < 5.0 km (flags surge and high-wind exposure).",
        "2. **Shelter Deficit Trigger:** Primary cyclone/emergency shelter score < 20.0 (flags acute refuge capacity deficit).",
        "3. **Healthcare Remoteness Trigger:** Primary/community healthcare access score < 35.0 (flags medical evacuation isolation).",
        "4. **Baseline Low Preparedness Trigger:** Composite AHP score < 40.0 (promotes unit to Band 1 Critical).",
        "",
        "#### Authoritative Priority Band Breakdown (Uttara Kannada: 1,257 Units)",
        "- **Band 1: Critical Priority:** 153 units (12.2%)",
        "- **Band 2: Specialized Readiness:** 686 units (54.6%)",
        "- **Band 3: Moderate Priority:** 211 units (16.8%)",
        "- **Band 4: Adequate Baseline:** 207 units (16.5%)",
        "",
        "All 1,257 units possess full authoritative AHP preparedness scores and rule-based field advisories.",
        "",
        "---",
        f"*{SCIENTIFIC_PRECEDENCE_DISCLAIMER}*",
    ]
    return "\n".join(sections)


def format_ambiguous_community_response(
    community_name: str,
    candidates: List[Dict[str, Any]],
) -> str:
    """Format disambiguation request when a village name matches multiple administrative units."""
    count = len(candidates)
    sections = [
        "### Community Disambiguation Required (Deterministic Mode)",
        "",
        f"The village name **'{community_name}'** matches **{count}** distinct administrative units in the coastal Karnataka study area.",
        "",
        "Please specify the unique Census PC11 ID or SHRID to inspect granular preparedness metrics:",
        "",
        "| Census PC11 ID | SHRID | Village Name | Taluk / Subdistrict | District | Assigned Priority Band | AHP Score |",
        "|---|---|---|---|---|---|---|",
    ]

    for c in candidates:
        pc11 = c.get("pc11_id", "N/A")
        shrid = c.get("shrid2", "N/A")
        name = c.get("village_name", community_name)
        taluk = c.get("subdistrict_name", "N/A")
        if isinstance(taluk, str):
            taluk = taluk.strip()
        dist = c.get("district_name", "N/A")
        band = c.get("priority_band", "N/A")
        score = c.get("preparedness_score", "N/A")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
        sections.append(
            f"| `{pc11}` | `{shrid}` | **{name}** | {taluk} | {dist} | `{band}` | {score_str} |"
        )

    sections.extend([
        "",
        f"*Example query:* `Assess {community_name} with PC11 ID {candidates[0].get('pc11_id')}`.",
        "",
        "---",
        f"*{SCIENTIFIC_PRECEDENCE_DISCLAIMER}*",
    ])
    return "\n".join(sections)


def format_unsupported_query_response(query: str) -> str:
    """Format clear boundary message for unsupported or out-of-domain inquiries in deterministic mode."""
    return f"""### Decision Support System Boundary (Deterministic Mode)

I am unable to answer this inquiry directly in deterministic fallback mode.

**Inquiry Received:** *"{query}"*

**Operational Scope:**
The Cyclone Preparedness Intelligence Platform is a closed-domain spatial decision-support system covering 1,879 surveyed administrative units across coastal Karnataka (Dakshina Kannada, Udupi, and Uttara Kannada).

**Supported Operational Queries in Deterministic Mode:**
1. **Community Assessments:** e.g., *"Explain why Tallur has its current priority classification"*, *"Assess Edmoge"*
2. **District Comparisons:** e.g., *"Compare Dakshina Kannada and Udupi"*
3. **Priority Band Rosters:** e.g., *"Find Band 1 communities"*, *"List Band 2 units"*
4. **Methodological Briefings:** e.g., *"Explain Uttara Kannada's fallback methodology"*
5. **Direct Identifiers:** Querying by 6-digit Census PC11 ID (e.g., `608691`, `608720`) or SHRID

For live cyclone track forecasts, active rainfall radar, or official evacuation orders, please consult the **India Meteorological Department (IMD)** and the **Karnataka State Disaster Management Authority (KSDMA)**.

---
*{SCIENTIFIC_PRECEDENCE_DISCLAIMER}*"""
