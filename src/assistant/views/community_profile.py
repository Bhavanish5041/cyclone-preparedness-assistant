"""Community Profiles: Granular Administrative Unit Decision Inspection.

Provides:
- Detailed multi-sector profile across all 1,879 revenue villages and census towns in coastal Karnataka.
- Authoritative AHP Preparedness Score and Tier badge.
- Operational Priority Band, Band Rank, and Overall Regional Rank.
- Four Sector Indicators: Medical, Shelter, Civil Infrastructure, Coastal and Hazard Exposure.
- Demographic Profile: Census population, households, SC and ST share, sex ratio, density.
- Unsupervised Vulnerability Archetype profiling (Assigned Clusters 0-3 or Uttara Kannada Sector Fallback).
- Multinomial Logistic Regression surrogate tier predictions and class probabilities with strict caveats.
- Deterministic decision-support triggers, primary rationale, and grounded operational advisories.
- Multidimensional Polar Radar Chart comparing unit against district medians (for units with dimensional scores).
- Scientific data provenance, vintage limitations, and non-operational disclaimers.
"""

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.charts import generate_radar_chart
from src.assistant.theme import (
    ARCHETYPE_COLORS,
    BAND_COLORS,
    TIER_COLORS,
    render_disclaimer,
    render_header,
    render_kpi_card,
)


def render_community_profile_page(client: DecisionSupportApiClient) -> None:
    """Render the granular community inspection profile page."""
    st.markdown(
        render_header(
            title="Community Preparedness Profile and Inspection Dossier",
            subtitle="Deep-dive inspection of authoritative AHP capacity, surrogate ML diagnostic probabilities, infrastructure indicators, and tailored response advisories for any of the 1,879 administrative units.",
            badge="Community Dossier",
        ),
        unsafe_allow_html=True,
    )

    # Load complete unit index (1,879 units)
    try:
        unit_df = client.get_unit_index()
    except Exception as err:
        st.error(f"Error loading unit index: {err}")
        return

    name_col = "village_name" if "village_name" in unit_df.columns else "unit_name"
    all_sorted_units = unit_df.sort_values(name_col)

    unit_options = [
        f"{row[name_col]} ({row['district_name']}, Taluk: {row.get('subdistrict_name', 'N/A')}) | PC11: {row['pc11_id']}"
        for _, row in all_sorted_units.iterrows()
    ]
    pc11_map = {opt: str(row["pc11_id"]) for opt, (_, row) in zip(unit_options, all_sorted_units.iterrows())}

    # 1. Search Controls and Quick Picks
    st.markdown("### Select Administrative Unit")
    quick_col, sel_col = st.columns([1, 2])

    with quick_col:
        st.markdown("**Regional Quick Picks:**")
        q1, q2 = st.columns(2)
        with q1:
            pick_tallur = st.button("Tallur (Coast, Udupi)", key="cp_tallur")
            pick_edmoge = st.button("Edmoge (Foothill, Udupi)", key="cp_edmoge")
            pick_bhatkal = st.button("Mudbhatkal (Coast, UK)", key="cp_bhatkal")
        with q2:
            pick_shim = st.button("Shimanthur (Rural, DK)", key="cp_shim")
            pick_siddapur = st.button("Siddapur (Upland, UK)", key="cp_siddapur")
            pick_sirsi = st.button("Sirsi (Upland, UK)", key="cp_sirsi")

    # Handle cross-page selection or quick pick triggers
    default_idx = 0
    if pick_tallur:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Tallur" in opt), 0)
    elif pick_edmoge:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Edmoge" in opt), 0)
    elif pick_bhatkal:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Mudbhatkal" in opt), 0)
    elif pick_shim:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Shimanthur" in opt), 0)
    elif pick_siddapur:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Siddapur" in opt), 0)
    elif pick_sirsi:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Sirsi" in opt), 0)
    elif "selected_community_id" in st.session_state:
        cur_id = str(st.session_state["selected_community_id"])
        match_idx = next((i for i, opt in enumerate(unit_options) if cur_id in opt), None)
        if match_idx is not None:
            default_idx = match_idx

    with sel_col:
        selected_option = st.selectbox(
            "Search by Unit Name, Census PC11 ID, or Taluk",
            options=unit_options,
            index=default_idx,
            key="cp_select_box",
        )

    selected_pc11 = pc11_map[selected_option]

    # Retrieve full DecisionSupportEvaluation from client
    try:
        evaluation = client.get_unit(selected_pc11)
    except Exception as err:
        st.error(f"Error retrieving decision evaluation for unit '{selected_pc11}': {err}")
        return

    # Also obtain raw indicators from ranked_df
    raw_match = client.service.ranked_df[client.service.ranked_df["pc11_id"] == selected_pc11]
    raw_row = raw_match.iloc[0] if not raw_match.empty else None

    # 2. Executive Unit Header and Key Metrics
    u_name = evaluation.village_name or "Unit"
    d_name = evaluation.district_name or ""
    sub_dist = evaluation.subdistrict_name or "N/A"
    u_type = evaluation.unit_type or "VILLAGE"
    shrid = evaluation.shrid2 or "N/A"

    st.markdown(
        f"""
        <div style="background: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <h2 style="margin: 0; color: #F3E9DC; font-family: 'DM Serif Display', Georgia, serif; font-size: 1.6rem; font-weight: normal;">{u_name} <span style="font-size: 0.9rem; color: #918170; font-family: 'DM Sans', sans-serif;">({u_type})</span></h2>
                    <div style="color: #BDAE9C; font-size: 0.88rem; margin-top: 0.35rem;">
                        District: <b style="color: #F3E9DC;">{d_name}</b> &nbsp;|&nbsp;
                        Taluk / Subdistrict: <b style="color: #F3E9DC;">{sub_dist}</b> &nbsp;|&nbsp;
                        Census PC11: <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">{selected_pc11}</code> &nbsp;|&nbsp;
                        SHRID: <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">{shrid}</code>
                    </div>
                </div>
                <div style="text-align: right; margin-top: 0.2rem;">
                    <div style="font-size: 0.8rem; color: #918170;">Geographic Coordinates</div>
                    <code style="color: #C18B62; font-size: 0.88rem; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">{evaluation.lat:.4f}° N, {evaluation.lon:.4f}° E</code>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Key KPI Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        t_meta = TIER_COLORS.get(evaluation.preparedness_tier, {"hex": "#C18B62"})
        st.markdown(
            render_kpi_card(
                title="AHP Preparedness Score",
                value=f"{evaluation.preparedness_score:.2f}",
                caption=f"Tier: <b>{evaluation.preparedness_tier}</b>",
                color=t_meta["hex"],
                badge_label="Authoritative AHP",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        b_meta = BAND_COLORS.get(evaluation.provisional_priority_band, {"hex": "#C18B62", "short_label": "Band"})
        st.markdown(
            render_kpi_card(
                title="Operational Priority Band",
                value=b_meta["short_label"],
                caption=f"Rank: <b>#{evaluation.overall_priority_rank}</b> of 1,879",
                color=b_meta["hex"],
                badge_label=f"Band Rank #{evaluation.band_rank}",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        arch_lbl = evaluation.vulnerability_archetype_label or "Uttara Kannada Sector Fallback"
        arch_id = evaluation.vulnerability_archetype_id
        if arch_id is not None and arch_id in ARCHETYPE_COLORS:
            arch_color = ARCHETYPE_COLORS[arch_id]["hex"]
            arch_display = f"Cluster {arch_id}"
        else:
            arch_color = "#918170"
            arch_display = "Sector Fallback"
        st.markdown(
            render_kpi_card(
                title="Vulnerability Archetype",
                value=arch_display,
                caption=arch_lbl,
                color=arch_color,
                badge_label="K-Means (k=4)",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        coast_dist = float(raw_row.get("dist_coast_km", 0.0)) if raw_row is not None else 0.0
        st.markdown(
            render_kpi_card(
                title="Distance to Coastline",
                value=f"{coast_dist:.1f} km",
                caption="Geodesic Shoreline Proximity",
                color="#C95D4A" if coast_dist < 5.0 else "#C18B62",
                badge_label="<5km Critical" if coast_dist < 5.0 else "Inland",
            ),
            unsafe_allow_html=True,
        )

    # 3. Sector Indicators Grid (Medical, Shelter, Infrastructure, Exposure)
    st.markdown("### Multi-Sector Preparedness Indicators")
    sec1, sec2, sec3, sec4 = st.columns(4)

    # Domain scores
    dom = evaluation.domain_scores or {}
    s_med = dom.get("score_medical", 0.0)
    s_she = dom.get("score_shelters", 0.0)
    s_inf = dom.get("score_infrastructure", 0.0)
    s_ew = dom.get("score_early_warning", 0.0)

    with sec1:
        hosp_dist = float(raw_row.get("dist_nearest_hospital_km", 0.0)) if raw_row is not None else 0.0
        phc_dist = float(raw_row.get("dist_nearest_phc_km", 0.0)) if raw_row is not None else 0.0
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="color: #D6B28C; font-weight: 600; margin-bottom: 0.5rem;">Medical Preparedness</div>
                <div style="font-size: 1.4rem; font-weight: 600; color: #F3E9DC; font-feature-settings: 'tnum' 1;">{s_med:.1f} <span style="font-size:0.8rem; color:#918170;">/ 100</span></div>
                <hr style="border-color: #514033; margin: 0.5rem 0;">
                <div style="font-size: 0.82rem; color: #BDAE9C; line-height: 1.5;">
                    Nearest Hospital: <b style="color: #F3E9DC;">{hosp_dist:.1f} km</b><br>
                    Nearest PHC/CHC: <b style="color: #F3E9DC;">{phc_dist:.1f} km</b><br>
                    Facilities in 10 km: <b style="color: #F3E9DC;">{int(raw_row.get('n_hc_10km', 0)) if raw_row is not None else 'N/A'}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sec2:
        she_dist = float(raw_row.get("dist_nearest_shelter_km", 0.0)) if raw_row is not None else 0.0
        she_cap = int(raw_row.get("shelter_capacity_nearby", 0)) if raw_row is not None else 0
        she_count = int(raw_row.get("n_shelters_10km", 0)) if raw_row is not None else 0
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="color: #C18B62; font-weight: 600; margin-bottom: 0.5rem;">Shelter Preparedness</div>
                <div style="font-size: 1.4rem; font-weight: 600; color: #F3E9DC; font-feature-settings: 'tnum' 1;">{s_she:.1f} <span style="font-size:0.8rem; color:#918170;">/ 100</span></div>
                <hr style="border-color: #514033; margin: 0.5rem 0;">
                <div style="font-size: 0.82rem; color: #BDAE9C; line-height: 1.5;">
                    Nearest Shelter: <b style="color: #F3E9DC;">{she_dist:.1f} km</b><br>
                    Shelters in 10 km: <b style="color: #F3E9DC;">{she_count}</b><br>
                    Nearby Capacity: <b style="color: #F3E9DC;">{she_cap:,} persons</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sec3:
        has_pucca = bool(raw_row.get("has_pucca_road", 0)) if raw_row is not None else False
        water_score = float(raw_row.get("drinking_water_score", 0.0)) if raw_row is not None else 0.0
        bus_conn = bool(raw_row.get("bus_connectivity", 0)) if raw_row is not None else False
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="color: #8D9A78; font-weight: 600; margin-bottom: 0.5rem;">Civil Infrastructure</div>
                <div style="font-size: 1.4rem; font-weight: 600; color: #F3E9DC; font-feature-settings: 'tnum' 1;">{s_inf:.1f} <span style="font-size:0.8rem; color:#918170;">/ 100</span></div>
                <hr style="border-color: #514033; margin: 0.5rem 0;">
                <div style="font-size: 0.82rem; color: #BDAE9C; line-height: 1.5;">
                    Pucca Road: <b style="color: #F3E9DC;">{'Yes' if has_pucca else 'No'}</b><br>
                    Bus Transit: <b style="color: #F3E9DC;">{'Yes' if bus_conn else 'No'}</b><br>
                    Drinking Water Index: <b style="color: #F3E9DC;">{water_score:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sec4:
        elev = float(raw_row.get("elevation_mean", 0.0)) if raw_row is not None else 0.0
        tauktae = bool(raw_row.get("tauktae_affected", 0)) if raw_row is not None else False
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="color: #BDAE9C; font-weight: 600; margin-bottom: 0.5rem;">Hazard and Early Warning</div>
                <div style="font-size: 1.4rem; font-weight: 600; color: #F3E9DC; font-feature-settings: 'tnum' 1;">{s_ew:.1f} <span style="font-size:0.8rem; color:#918170;">/ 100</span></div>
                <hr style="border-color: #514033; margin: 0.5rem 0;">
                <div style="font-size: 0.82rem; color: #BDAE9C; line-height: 1.5;">
                    Mean Elevation: <b style="color: #F3E9DC;">{elev:.1f} m ASL</b><br>
                    Shoreline Distance: <b style="color: #F3E9DC;">{coast_dist:.1f} km</b><br>
                    Tauktae 2021 Impact: <b style="color: #F3E9DC;">{'Flooded' if tauktae else 'Normal'}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Demographic Profile
    if raw_row is not None:
        pop = int(raw_row.get("population", 0))
        hh = int(raw_row.get("households", 0))
        sc_share = float(raw_row.get("sc_share", 0.0)) * 100
        st_share = float(raw_row.get("st_share", 0.0)) * 100
        density = float(raw_row.get("pop_density_sqkm", 0.0))
        st.markdown(
            f"""
            <div style="background: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 0.75rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.85rem; color: #BDAE9C; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.8rem;">
                <span>Census Population: <b style="color: #F3E9DC;">{pop:,}</b></span>
                <span>Households: <b style="color: #F3E9DC;">{hh:,}</b></span>
                <span>Density: <b style="color: #F3E9DC;">{density:.1f}/km²</b></span>
                <span>SC Share: <b style="color: #F3E9DC;">{sc_share:.1f}%</b></span>
                <span>ST Share: <b style="color: #F3E9DC;">{st_share:.1f}%</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Logistic Regression Surrogate Diagnostic Panel (Scientific Transparency)
    st.markdown("### Logistic Regression Surrogate Model Diagnostics")
    probs = evaluation.class_probabilities or {}
    p_low = probs.get("prob_low", 0.0)
    p_med = probs.get("prob_medium", 0.0)
    p_high = probs.get("prob_high", 0.0)

    col_ml1, col_ml2 = st.columns([1, 1])
    with col_ml1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="font-weight: 600; color: #F3E9DC; margin-bottom: 0.4rem;">Surrogate Classifier Prediction</div>
                <div style="font-size: 1.2rem; color: #C18B62; font-weight: 600;">
                    Predicted Tier: <span style="color:#F3E9DC;">{evaluation.predicted_preparedness_tier}</span>
                </div>
                <div style="font-size: 0.84rem; color: #BDAE9C; margin-top: 0.3rem;">
                    Agreement Status: <code style="color:#8D9A78; background:#211914; padding:2px 5px; border-radius:3px;">{evaluation.model_agreement_status}</code> &nbsp;|&nbsp;
                    Boundary Case: <b>{'Yes' if evaluation.is_boundary_case else 'No'}</b> &nbsp;|&nbsp;
                    Margin: <b>{evaluation.probability_margin:.3f}</b>
                </div>
                <hr style="border-color: #514033; margin: 0.6rem 0;">
                <div style="font-size: 0.82rem; color: #BDAE9C;">
                    <b>Uncalibrated Surrogate Class Probabilities:</b><br>
                    Low Preparedness Prob: <code style="color:#C95D4A; background:#211914; padding:1px 4px; border-radius:3px;">{p_low:.1%}</code><br>
                    Medium Preparedness Prob: <code style="color:#D48B47; background:#211914; padding:1px 4px; border-radius:3px;">{p_med:.1%}</code><br>
                    High Preparedness Prob: <code style="color:#7A8B63; background:#211914; padding:1px 4px; border-radius:3px;">{p_high:.1%}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ml2:
        st.markdown(
            render_disclaimer(
                text="The Logistic Regression model (W&B Artifact v0) serves strictly as an analytical surrogate approximating deterministic AHP-derived tiers from 26 feature variables. Predicted probabilities represent multinomial simplex likelihoods of structural capacity, NOT calibrated probabilities of meteorological cyclone strikes, storm surges, structural destruction, or casualties. AHP remains the sole authoritative baseline for operational prioritization.",
                title="Scientific Caveat on Model Probabilities",
            ),
            unsafe_allow_html=True,
        )

    # 6. Operational Priority Grounding and Rationale
    st.markdown("### Operational Priority Grounding and Directives")
    st.markdown(
        f"""
        <div style="background: #2B211B; border: 1px solid #514033; border-left: 3px solid #C18B62; border-radius: 4px; padding: 0.85rem 1.1rem; margin-bottom: 1rem; color: #F3E9DC; font-size: 0.88rem;">
            <b>Decision Support Rationale:</b> {evaluation.priority_explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Grounded Advisories
    if evaluation.recommended_advisories:
        st.markdown("**Tailored Incident Command Advisories:**")
        for adv in evaluation.recommended_advisories:
            st.markdown(
                f"""
                <div class="checklist-item">
                    <b style="color: #C18B62;">[{adv.directive_id}] {adv.advisory_focus}:</b> {adv.operational_recommendation}<br>
                    <span style="font-size: 0.76rem; color: #918170;">Agency: {adv.lead_coordinating_agency} | Rule: <code>{adv.eligibility_rule}</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 7. Radar Profile (For 622 units with dimension scores)
    unit_shrid = evaluation.shrid2
    if unit_shrid and unit_shrid in client.service.df["shrid2"].values:
        st.markdown("### Multidimensional Deficit Profile vs District Median")
        unit_match = client.service.df[client.service.df["shrid2"] == unit_shrid].iloc[0]
        u_scores = {
            "healthcare_deficit": float(unit_match["dim_healthcare_deficit"]),
            "demographic_vulnerability": float(unit_match["dim_demographic_vulnerability"]),
            "infrastructure_deficit": float(unit_match["dim_infrastructure_deficit"]),
            "coastal_exposure": float(unit_match["dim_coastal_exposure"]),
        }
        dist_df = client.service.df[client.service.df["district_name"] == evaluation.district_name]
        d_medians = {
            "healthcare_deficit": float(dist_df["dim_healthcare_deficit"].median()),
            "demographic_vulnerability": float(dist_df["dim_demographic_vulnerability"].median()),
            "infrastructure_deficit": float(dist_df["dim_infrastructure_deficit"].median()),
            "coastal_exposure": float(dist_df["dim_coastal_exposure"].median()),
        }

        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            radar_fig = generate_radar_chart(
                unit_scores=u_scores,
                district_medians=d_medians,
                unit_name=u_name,
                district_name=d_name,
            )
            st.pyplot(radar_fig, use_container_width=True)
            plt.close(radar_fig)
        with r_col2:
            st.markdown(
                """
                <div style="padding-top: 1.5rem; font-size: 0.88rem; color: #BDAE9C; line-height: 1.6;">
                    <b>Radar Chart Interpretation:</b><br>
                    The warm ochre polygon portrays the selected community's normalized deficit scores (0 = lowest deficit, 1 = maximum deficit).<br><br>
                    The dashed taupe line indicates the median deficit profile of all evaluated units in the same district.<br><br>
                    Communities protruding outward along the <b>Coastal Proximity</b> or <b>Healthcare Deficit</b> axes exhibit severe spatial vulnerability requiring pre-staged logistics.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Scientific Data Provenance and Operational Limitations
    st.markdown(
        render_disclaimer(
            text="Census statistics represent the 2011 decennial baseline (over 15 years old). Healthcare facility coordinates represent administrative directory registration, NOT verified active medical personnel staffing, surgical oxygen reserves, or backup diesel generator status during severe cyclonic power outages. Ground reconnaissance by District Disaster Management Authorities is mandatory.",
            title="Data Provenance and Operational Caveat",
        ),
        unsafe_allow_html=True,
    )
