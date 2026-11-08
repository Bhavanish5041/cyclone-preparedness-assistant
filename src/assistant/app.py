"""Cyclone Preparedness Assistant — Decision Support Dashboard.

Streamlit multi-page application for coastal Karnataka (Dakshina Kannada, Udupi, and Uttara Kannada).
Provides:
- Page 0: Regional Decision Dashboard (Milestone 4B.2) — Macroscopic prioritization overview across all 1,879 units.
- Page 1: Prioritization Map & Policy Studio — Dynamic MCDA weight simulation across 622 units.
- Page 2: Community Deep-Dive & Inspection Tool — Multidimensional radar profiles and tailored response checklists.
- Page 3: Scenario Comparison Studio — Rank sensitivity and divergence analysis between policy scenarios.
- Page 4: Archetypes & Methodology Reference — Comprehensive documentation of K-Means clusters and AHP criteria.

Architecture:
- Utilizes DecisionSupportApiClient (Milestone 4B.1) for standardized data access across both
  Direct In-Memory mode and FastAPI REST mode.
- Preserves full backward compatibility with existing analytical pipelines and visualizations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.schemas import (
    PrioritizationScenarioRequest,
    ScenarioComparisonRequest,
    WeightConfiguration,
)
from src.assistant.api_client import DecisionSupportApiClient, get_api_client
from src.assistant.decision_support import DecisionSupportService
from src.assistant.ui_components import (
    ARCHETYPE_COLORS,
    BAND_COLORS,
    METRIC_LABELS,
    TIER_COLORS,
    build_folium_choropleth_map,
    generate_archetype_distribution_chart,
    generate_district_band_chart,
    generate_radar_chart,
    generate_rank_scatter_chart,
    generate_tier_distribution_chart,
    get_custom_css,
    render_kpi_card,
)
from src.features.prioritization import PRIORITIZATION_DISCLAIMER
from src.models.clustering import ARCHETYPE_DISCLAIMER, ARCHETYPE_LABELS_K4

# Configure Streamlit page layout
st.set_page_config(
    page_title="Cyclone Preparedness Assistant | Decision Support Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Cyclone Preparedness Client...")
def get_client() -> DecisionSupportApiClient:
    """Initialize and cache the DecisionSupportApiClient singleton."""
    adapter_mode = os.getenv("CYCLONE_ADAPTER_MODE", "direct")
    return get_api_client(mode=adapter_mode)


def render_sidebar(client: DecisionSupportApiClient) -> Tuple[str, Optional[str]]:
    """Render the sidebar with navigation, legacy district filter, and architecture notes."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <h2 style="margin: 0; color: #38bdf8; font-size: 1.5rem;">🌀 Cyclone Assistant</h2>
            <p style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Spatial Decision Support System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "📊 Regional Decision Dashboard",
            "🗺️ Prioritization Map & Policy Studio",
            "🔍 Community Deep-Dive",
            "⚖️ Scenario Comparison",
            "📚 Archetypes & Methodology",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Legacy Filter (Pages 1–4)")
    district_choice = st.sidebar.selectbox(
        "District Scope (Legacy Tools)",
        options=["All Coastal Karnataka (622 Units)", "Dakshina Kannada (369 Units)", "Udupi (253 Units)"],
        index=0,
    )
    district_filter = None
    if "Dakshina Kannada" in district_choice:
        district_filter = "Dakshina Kannada"
    elif "Udupi" in district_choice:
        district_filter = "Udupi"

    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Architecture & Mode"):
        st.markdown(
            f"""
            **Integration Adapter (Active)**
            - Mode: `{client.mode.value}`
            - Total Study Area: `1,879 Units`
            - Client: `DecisionSupportApiClient`

            **Backend Services**
            - Direct In-Memory Service: `Cached`
            - FastAPI REST Endpoints: `/api/v1/decision-support/...`
            """
        )

    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 2rem;">
            RV College of Engineering • Dept. of CSE (AI & ML)<br>
            Coastal Karnataka Preparedness Rebuild
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page, district_filter


# ==============================================================================
# PAGE 0: Regional Decision Dashboard (Milestone 4B.2)
# ==============================================================================
def render_page_regional_dashboard(client: DecisionSupportApiClient) -> None:
    """Render the macroscopic regional decision-support dashboard across all 1,879 units."""
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Regional Cyclone Preparedness Dashboard</h1>
            <p class="hero-subtitle">
                Macroscopic decision-support overview, priority-band allocation, and comparative vulnerability
                across all 1,879 revenue villages and census towns in coastal Karnataka.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Governance & Vintage Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Scientific Governance & Baseline Vintage:</b> Decision support evaluations are grounded in the
            decennial 2011 Census baseline, MoHFW healthcare registries, and deterministic AHP/MCDA criteria.
            All priority classifications and operational advisories serve as structured decision-support guides
            for disaster management authorities and require field inspection. Real-time facility staffing,
            ICU capacity, and generator fuel status are not modeled.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Dynamic KPI Retrieval & Aggregation
    try:
        summary = client.get_regional_summary()
        total_units = summary.total_units
        b1 = summary.band_counts.get("BAND_1_CRITICAL", 0)
        b2 = summary.band_counts.get("BAND_2_SPECIALIZED", 0)
        b3 = summary.band_counts.get("BAND_3_MODERATE", 0)
        b4 = summary.band_counts.get("BAND_4_ADEQUATE", 0)
    except Exception as err:
        st.error(f"⚠️ Error retrieving regional decision summary: {err}")
        return

    # Data Integrity Check
    sum_bands = b1 + b2 + b3 + b4
    if sum_bands != total_units:
        st.markdown(
            f"""
            <div class="data-integrity-warning">
                ⚠️ <b>Data Integrity Alert:</b> Sum of priority bands ({sum_bands:,}) does not reconcile with
                total study area units ({total_units:,}). Aggregation discrepancy detected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Regional KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            render_kpi_card(
                title="Study Area Total",
                value=f"{total_units:,}",
                caption="100% Accounted Units",
                color="#f8fafc",
                badge_label="Total",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_kpi_card(
                title="Band 1 — Critical",
                value=f"{b1:,}",
                percentage=(b1 / total_units * 100) if total_units else 0,
                caption="Immediate Evac / Hardening",
                color="#ef4444",
                badge_label="Critical",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_kpi_card(
                title="Band 2 — Specialized",
                value=f"{b2:,}",
                percentage=(b2 / total_units * 100) if total_units else 0,
                caption="Targeted Deficits & Hazards",
                color="#f59e0b",
                badge_label="Specialized",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            render_kpi_card(
                title="Band 3 — Moderate",
                value=f"{b3:,}",
                percentage=(b3 / total_units * 100) if total_units else 0,
                caption="Secondary Readiness",
                color="#38bdf8",
                badge_label="Moderate",
            ),
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            render_kpi_card(
                title="Band 4 — Adequate",
                value=f"{b4:,}",
                percentage=(b4 / total_units * 100) if total_units else 0,
                caption="Baseline Continuity",
                color="#10b981",
                badge_label="Adequate",
            ),
            unsafe_allow_html=True,
        )

    # 2. Interactive Dashboard Filters
    st.markdown("### 🎛️ Regional Dashboard Filters")
    col_f1, col_f2, col_f3, col_reset = st.columns([2, 2, 2, 1])

    with col_f1:
        f_dist = st.selectbox(
            "District Scope",
            options=["All Districts", "Dakshina Kannada", "Udupi", "Uttara Kannada"],
            index=0,
            key="rb_filter_district",
        )
    with col_f2:
        f_band = st.selectbox(
            "Priority Band Scope",
            options=[
                "All Priority Bands",
                "Band 1 — Critical Priority",
                "Band 2 — Specialized Readiness",
                "Band 3 — Moderate Priority",
                "Band 4 — Adequate Baseline",
            ],
            index=0,
            key="rb_filter_band",
        )
    with col_f3:
        f_tier = st.selectbox(
            "AHP Preparedness Tier Scope",
            options=[
                "All Preparedness Tiers",
                "High Preparedness (>60)",
                "Medium Preparedness (40–60)",
                "Low Preparedness (<40)",
            ],
            index=0,
            key="rb_filter_tier",
        )
    def reset_filters_callback():
        st.session_state["rb_filter_district"] = "All Districts"
        st.session_state["rb_filter_band"] = "All Priority Bands"
        st.session_state["rb_filter_tier"] = "All Preparedness Tiers"

    with col_reset:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        st.button(
            "🔄 Reset Filters",
            on_click=reset_filters_callback,
            use_container_width=True,
            key="rb_reset_btn",
        )

    # Load unit index for filtering
    try:
        unit_df = client.get_unit_index()
    except Exception as err:
        st.error(f"⚠️ Error loading unit index: {err}")
        return

    # Apply filters
    filtered_df = unit_df.copy()
    is_filtered = False

    if f_dist != "All Districts":
        filtered_df = filtered_df[filtered_df["district_name"] == f_dist]
        is_filtered = True

    if f_band != "All Priority Bands":
        band_map = {
            "Band 1 — Critical Priority": "BAND_1_CRITICAL",
            "Band 2 — Specialized Readiness": "BAND_2_SPECIALIZED",
            "Band 3 — Moderate Priority": "BAND_3_MODERATE",
            "Band 4 — Adequate Baseline": "BAND_4_ADEQUATE",
        }
        target_band = band_map.get(f_band)
        col_b = "provisional_priority_band" if "provisional_priority_band" in filtered_df.columns else "priority_band"
        filtered_df = filtered_df[filtered_df[col_b] == target_band]
        is_filtered = True

    if f_tier != "All Preparedness Tiers":
        tier_map = {
            "High Preparedness (>60)": "High",
            "Medium Preparedness (40–60)": "Medium",
            "Low Preparedness (<40)": "Low",
        }
        target_tier = tier_map.get(f_tier)
        filtered_df = filtered_df[filtered_df["preparedness_tier"] == target_tier]
        is_filtered = True

    # Active filter notification banner
    if is_filtered:
        n_filtered = len(filtered_df)
        pct_filtered = (n_filtered / total_units * 100) if total_units else 0
        st.markdown(
            f"""
            <div class="filter-banner">
                🔍 <b>Filtered Scope Active:</b> Displaying <b>{n_filtered:,}</b> of <b>{total_units:,}</b> units ({pct_filtered:.1f}%)
                matching: District: <i>{f_dist}</i> &nbsp;|&nbsp; Band: <i>{f_band}</i> &nbsp;|&nbsp; Tier: <i>{f_tier}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Empty State Handling
    if filtered_df.empty:
        st.warning("⚠️ No administrative units match the active filter criteria. Please adjust or reset the filters.")
        return

    st.markdown("---")

    # 3. District-Level Comparative Vulnerability
    st.markdown("### 🏛️ District Comparison")
    st.markdown(
        """
        Comparison of administrative unit counts and priority-band distribution across the three coastal districts:
        <b>Dakshina Kannada</b> (369 units), <b>Udupi</b> (253 units), and <b>Uttara Kannada</b> (1,257 units).
        """,
        unsafe_allow_html=True,
    )

    tab_count, tab_norm = st.tabs(["📊 Unit Counts by Priority Band", "📈 Normalized Proportion (%)"])
    with tab_count:
        chart_counts = generate_district_band_chart(filtered_df, normalized=False)
        st.altair_chart(chart_counts, use_container_width=True)
    with tab_norm:
        chart_norm = generate_district_band_chart(filtered_df, normalized=True)
        st.altair_chart(chart_norm, use_container_width=True)

    # District Cross-Tabulation Table
    col_b = "provisional_priority_band" if "provisional_priority_band" in filtered_df.columns else "priority_band"
    dist_table_rows = []
    for d_name in ["Dakshina Kannada", "Udupi", "Uttara Kannada"]:
        sub = filtered_df[filtered_df["district_name"] == d_name]
        d_tot = len(sub)
        d_b1 = int((sub[col_b] == "BAND_1_CRITICAL").sum())
        d_b2 = int((sub[col_b] == "BAND_2_SPECIALIZED").sum())
        d_b3 = int((sub[col_b] == "BAND_3_MODERATE").sum())
        d_b4 = int((sub[col_b] == "BAND_4_ADEQUATE").sum())
        high_prio_share = ((d_b1 + d_b2) / d_tot * 100) if d_tot else 0.0
        dist_table_rows.append(
            {
                "District": d_name,
                "Total Units": d_tot,
                "Band 1 (Critical)": d_b1,
                "Band 2 (Specialized)": d_b2,
                "Band 3 (Moderate)": d_b3,
                "Band 4 (Adequate)": d_b4,
                "High Urgency Share (Band 1+2)": f"{high_prio_share:.1f}%",
            }
        )
    dist_summary_df = pd.DataFrame(dist_table_rows)
    st.dataframe(dist_summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. Preparedness Tier Analytics (AHP vs Priority Bands)
    st.markdown("### 🎯 AHP Preparedness Tier Distribution")
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.88rem; line-height: 1.45;">
            <b>Distinction Between AHP Tiers and Priority Bands:</b><br>
            • <b>AHP Preparedness Tiers</b> (<code style="color:#ef4444;">Low &lt; 40</code>, <code style="color:#f59e0b;">Medium 40–60</code>, <code style="color:#10b981;">High &gt; 60</code>)
            measure intrinsic multi-sector infrastructural and demographic capacity.<br>
            • <b>Priority Bands</b> operationalize these scores for disaster response. Medium-tier units with severe coastal exposure (&lt;5 km),
            shelter deficits (&lt;20), or medical deficits (&lt;35) are promoted to <b>Band 2 (Specialized Readiness)</b>.<br>
            • <i>Note: Logistic Regression predicted tiers are surrogate diagnostics and do not substitute for authoritative AHP evaluation.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tier_col1, tier_col2 = st.columns(2)
    with tier_col1:
        st.markdown("**Overall AHP Tier Breakdown:**")
        chart_tier_overall = generate_tier_distribution_chart(filtered_df, by_district=False)
        st.altair_chart(chart_tier_overall, use_container_width=True)
    with tier_col2:
        st.markdown("**District-Wise Tier Breakdown:**")
        chart_tier_dist = generate_tier_distribution_chart(filtered_df, by_district=True)
        st.altair_chart(chart_tier_dist, use_container_width=True)

    st.markdown("---")

    # 5. Vulnerability Archetype Analytics
    st.markdown("### 🧬 Vulnerability Archetype Profiling")
    st.markdown(
        """
        Unsupervised K-Means clustering (k=4) was fitted on standardized demographic and infrastructure dimensions
        for <b>Dakshina Kannada and Udupi</b> (622 units). <b>Uttara Kannada</b> (1,257 units) is unassigned in clustering
        and evaluated via deterministic three-sector deficit fallback triggers.
        """,
        unsafe_allow_html=True,
    )

    n_assigned = int(filtered_df["vulnerability_archetype_label"].notna().sum())
    n_unassigned = int(filtered_df["vulnerability_archetype_label"].isna().sum())

    arch_m1, arch_m2, arch_m3 = st.columns(3)
    with arch_m1:
        st.markdown(
            render_kpi_card(
                "K-Means Assigned Units",
                f"{n_assigned:,}",
                caption="Dakshina Kannada & Udupi",
                color="#38bdf8",
            ),
            unsafe_allow_html=True,
        )
    with arch_m2:
        st.markdown(
            render_kpi_card(
                "Sector Fallback Units",
                f"{n_unassigned:,}",
                caption="Uttara Kannada (Unassigned)",
                color="#94a3b8",
            ),
            unsafe_allow_html=True,
        )
    with arch_m3:
        st.markdown(
            render_kpi_card(
                "Vulnerability Archetypes",
                "4 Clusters",
                caption="K-Means (k=4) Partitioning",
                color="#10b981",
            ),
            unsafe_allow_html=True,
        )

    chart_arch = generate_archetype_distribution_chart(filtered_df)
    st.altair_chart(chart_arch, use_container_width=True)

    # Archetype Profile Reference Tiles
    st.markdown("#### Discovered Cluster Archetype Reference:")
    arc1, arc2, arc3, arc4 = st.columns(4)
    with arc1:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.8rem; font-size: 0.82rem;">
                <b style="color: #10b981;">Cluster 0</b><br>
                <b>Urban & Peri-Urban Corridor</b><br>
                High road paving, reliable drainage, high treated water access. Lower overall physical deficit.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arc2:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 0.8rem; font-size: 0.82rem;">
                <b style="color: #f59e0b;">Cluster 1</b><br>
                <b>Remote Foothill Communities</b><br>
                Significant hospital distance, moderate elevation, low road connectivity. High healthcare isolation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arc3:
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 0.8rem; font-size: 0.82rem;">
                <b style="color: #8b5cf6;">Cluster 2</b><br>
                <b>Socioeconomically Vulnerable Rural</b><br>
                Higher SC/ST population share, kutcha housing prevalence, drainage deficits in rural interior.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arc4:
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 0.8rem; font-size: 0.82rem;">
                <b style="color: #ef4444;">Cluster 3</b><br>
                <b>High-Exposure Coastal Belt</b><br>
                Within 5 km of shoreline, vulnerable to storm surge and coastal flooding. Evacuation priority.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top: 1rem;">
            <b>Scientific Precaution:</b> Archetypes characterize structural demographic and infrastructural
            baselines from 2011 decennial census records. They do NOT predict meteorological storm trajectories,
            cyclonic wind speeds, or real-time storm surge water levels.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: Prioritization Map & Policy Studio (Legacy - Preserved)
# ==============================================================================
def render_page_prioritization_map(service: DecisionSupportService, district_filter: Optional[str]) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Prioritization Map & Policy Studio</h1>
            <p class="hero-subtitle">
                Simulate emergency resource allocation policies across 622 coastal Karnataka revenue villages and towns.
                Adjust dimension weights to recompute deterministic preparedness priority scores and identify underserved communities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Weight Preset Selection and Interactive Sliders
    st.markdown("### 🎛️ Policy Weighting Studio")
    col_preset, col_scale = st.columns([3, 1])

    with col_preset:
        preset = st.selectbox(
            "Load Policy Preset",
            options=[
                "Equal Weights (Analytical Baseline: 25% each)",
                "Rapid Landfall Hazard Focus (60% Coastal Proximity, 20% Infra, 10% Health, 10% Demo)",
                "Rural Healthcare Isolation (50% Healthcare Deficit, 25% Demo, 25% Infra, 0% Coastal)",
                "Inland Resilience Focus (0% Coastal, 35% Healthcare, 35% Infra, 30% Demo)",
                "Custom Weight Sliders",
            ],
            index=0,
        )

    with col_scale:
        include_pop = st.checkbox(
            "Scale by Population",
            value=False,
            help="When checked, factors absolute Census population scale into the demographic vulnerability dimension.",
        )

    # Preset values mapping
    if "Equal Weights" in preset:
        w_health, w_demo, w_infra, w_coast = 25, 25, 25, 25
    elif "Rapid Landfall" in preset:
        w_health, w_demo, w_infra, w_coast = 10, 10, 20, 60
    elif "Rural Healthcare" in preset:
        w_health, w_demo, w_infra, w_coast = 50, 25, 25, 0
    elif "Inland Resilience" in preset:
        w_health, w_demo, w_infra, w_coast = 35, 30, 35, 0
    else:
        w_health, w_demo, w_infra, w_coast = 30, 20, 20, 30

    # Render interactive sliders
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        s_health = st.slider("Healthcare Deficit Weight", 0, 100, w_health, key="sl_h")
    with col2:
        s_demo = st.slider("Demographic Vulnerability Weight", 0, 100, w_demo, key="sl_d")
    with col3:
        s_infra = st.slider("Infrastructure Deficit Weight", 0, 100, w_infra, key="sl_i")
    with col4:
        s_coast = st.slider("Coastal Hazard Weight", 0, 100, w_coast, key="sl_c")

    # Safe normalization calculation
    total_raw = s_health + s_demo + s_infra + s_coast
    if total_raw == 0:
        st.error("⚠️ At least one dimension weight must be greater than zero.")
        return

    norm_h = s_health / total_raw
    norm_d = s_demo / total_raw
    norm_i = s_infra / total_raw
    norm_c = s_coast / total_raw

    # Display effective normalized weights
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 1.2rem; font-size: 0.88rem;">
            <b>Effective Normalized Weights:</b> &nbsp;
            🏥 Healthcare: <code style="color:#38bdf8;">{norm_h:.1%}</code> &nbsp;|&nbsp;
            👥 Demographic: <code style="color:#38bdf8;">{norm_d:.1%}</code> &nbsp;|&nbsp;
            🏗️ Infrastructure: <code style="color:#38bdf8;">{norm_i:.1%}</code> &nbsp;|&nbsp;
            🌊 Coastal Proximity: <code style="color:#38bdf8;">{norm_c:.1%}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Recompute scenario rankings dynamically
    req_weights = WeightConfiguration(
        healthcare=float(s_health),
        demographic=float(s_demo),
        infrastructure=float(s_infra),
        coastal=float(s_coast),
        auto_normalize=True,
        include_population_scale=include_pop,
    )
    req = PrioritizationScenarioRequest(
        scenario_name=preset if "Custom" not in preset else "Custom Scenario",
        weights=req_weights,
        top_n=622,
        district_filter=district_filter,
    )

    scenario_res = service.compute_prioritization_scenario(req)

    eval_df = service.df.copy()
    w_applied = scenario_res.weights_applied
    custom_score = (
        w_applied["healthcare"] * eval_df["dim_healthcare_deficit"]
        + w_applied["demographic"] * eval_df["dim_demographic_vulnerability"]
        + w_applied["infrastructure"] * eval_df["dim_infrastructure_deficit"]
        + w_applied["coastal"] * eval_df["dim_coastal_exposure"]
    ).round(4)

    eval_df["priority_score"] = custom_score
    eval_df["priority_rank"] = custom_score.rank(ascending=False, method="min").astype(int)

    if district_filter:
        display_df = eval_df[eval_df["district_name"].str.lower() == district_filter.lower()].copy()
    else:
        display_df = eval_df.copy()

    is_baseline = (
        abs(norm_h - 0.25) < 1e-4
        and abs(norm_d - 0.25) < 1e-4
        and abs(norm_i - 0.25) < 1e-4
        and abs(norm_c - 0.25) < 1e-4
        and not include_pop
    )

    if is_baseline:
        st.info("ℹ️ Displaying **Analytical Baseline: Equal Weights (25% each)** across all dimensions.")
    else:
        st.success(f"⚡ Displaying **Dynamic Policy Scenario**: *{req.scenario_name}*")

    # Key Statistical Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Units Evaluated</div>
                <div class="metric-value">{len(display_df)}</div>
                <div class="metric-caption">{district_filter or "Both Districts"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        top_unit = display_df.sort_values("priority_score", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Top Priority Unit</div>
                <div class="metric-value" style="font-size: 1.25rem;">{top_unit['unit_name']}</div>
                <div class="metric-caption">Score: {top_unit['priority_score']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Median Score</div>
                <div class="metric-value">{display_df['priority_score'].median():.4f}</div>
                <div class="metric-caption">Interquartile baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Max Priority Score</div>
                <div class="metric-value" style="color: #ef4444;">{display_df['priority_score'].max():.4f}</div>
                <div class="metric-caption">Highest composite deficit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Min Priority Score</div>
                <div class="metric-value" style="color: #10b981;">{display_df['priority_score'].min():.4f}</div>
                <div class="metric-caption">Lowest composite deficit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Folium Map Layer & Overlay Controls
    st.markdown("### 🗺️ Geospatial Layer Exploration")
    map_col1, map_col2, map_col3 = st.columns([2, 1, 1])

    with map_col1:
        selected_metric = st.selectbox(
            "Choropleth Color Layer",
            options=[
                ("priority_score", "Composite Priority Score (Active Scenario)"),
                ("dim_healthcare_deficit", "Healthcare Access Deficit (0 to 1)"),
                ("dim_demographic_vulnerability", "Demographic Vulnerability (0 to 1)"),
                ("dim_infrastructure_deficit", "Infrastructure Deficit (0 to 1)"),
                ("dim_coastal_exposure", "Coastal Hazard Proximity (0 to 1)"),
                ("vulnerability_archetype_id", "Discovered Vulnerability Archetypes (4 Clusters)"),
            ],
            format_func=lambda x: x[1],
            index=0,
        )[0]

    with map_col2:
        show_coast = st.checkbox("Overlay 10m Coastline", value=True)

    with map_col3:
        show_hosp = st.checkbox("Overlay Healthcare Facilities", value=False)

    with st.spinner("Rendering geospatial choropleth map..."):
        fmap = build_folium_choropleth_map(
            gdf=display_df,
            metric_col=selected_metric,
            show_coastline=show_coast,
            show_healthcare=show_hosp,
        )
        st_folium(fmap, use_container_width=True, height=520, returned_objects=[])

    # Top-20 Prioritized Communities Table
    st.markdown("### 📋 Top-20 Prioritized Communities")
    top_20 = display_df.sort_values("priority_score", ascending=False).head(20).copy()

    table_df = pd.DataFrame(
        {
            "Rank": range(1, len(top_20) + 1),
            "Unit Name": top_20["unit_name"],
            "Type": top_20["unit_type"],
            "District": top_20["district_name"],
            "Taluk": top_20.get("subdistrict_name", "N/A"),
            "Priority Score": top_20["priority_score"].round(4),
            "Coastal Dist (km)": top_20["dist_to_coastline_km"].round(2),
            "Hospital Dist (km)": top_20["dist_to_nearest_hospital_km"].round(2),
            "Archetype": top_20["vulnerability_archetype_label"],
        }
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Download Table CSV
    csv_data = top_20.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Top Prioritized Units (CSV)",
        data=csv_data,
        file_name=f"cyclone_prioritization_{preset.split()[0].lower()}.csv",
        mime="text/csv",
    )


# ==============================================================================
# PAGE 2: Community Deep-Dive & Inspection Tool (Legacy - Preserved)
# ==============================================================================
def render_page_community_deep_dive(service: DecisionSupportService) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Community Deep-Dive & Inspection Tool</h1>
            <p class="hero-subtitle">
                Inspect granular multidimensional indicators, radar profiles, demographic vulnerability metrics,
                and tailored disaster response intervention checklists for any of the 622 revenue villages and towns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_units = service.df.sort_values("unit_name")
    unit_options = [
        f"{row['unit_name']} ({row['unit_type']}, {row['district_name']}) — PC11: {row['pc11_id']}"
        for _, row in all_units.iterrows()
    ]
    shrid_map = {opt: row["shrid2"] for opt, (_, row) in zip(unit_options, all_units.iterrows())}

    st.markdown("### 🔎 Select Administrative Unit")
    quick_col, sel_col = st.columns([1, 3])

    with quick_col:
        st.markdown("**Quick Picks:**")
        q1, q2 = st.columns(2)
        with q1:
            pick_tallur = st.button("Tallur (Coast)")
            pick_edmoge = st.button("Edmoge (Foothill)")
        with q2:
            pick_mang = st.button("Mangaluru (Urban)")
            pick_shim = st.button("Shimanthur (Rural)")

    default_idx = 0
    if pick_tallur:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Tallur" in opt), 0)
    elif pick_edmoge:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Edmoge" in opt), 0)
    elif pick_mang:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Mangaluru" in opt), 0)
    elif pick_shim:
        default_idx = next((i for i, opt in enumerate(unit_options) if "Shimanthur" in opt), 0)

    with sel_col:
        selected_option = st.selectbox(
            "Search by Unit Name, Census PC11 ID, or District",
            options=unit_options,
            index=default_idx,
        )

    selected_shrid = shrid_map[selected_option]
    unit_detail = service.get_unit_detail(selected_shrid)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Composite Priority Rank</div>
                <div class="metric-value">#{unit_detail.priority_rank} <span style="font-size:0.9rem; color:#64748b;">/ 622</span></div>
                <div class="metric-caption">Baseline equal weighting</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Deficit Score</div>
                <div class="metric-value">{unit_detail.priority_score:.4f}</div>
                <div class="metric-caption">Higher indicates greater vulnerability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        arch_id = unit_detail.archetype["archetype_id"]
        arch_meta = ARCHETYPE_COLORS.get(arch_id, {"hex": "#38bdf8"})
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Vulnerability Archetype</div>
                <div class="metric-value" style="font-size: 1.15rem; color: {arch_meta['hex']};">
                    Cluster {arch_id}
                </div>
                <div class="metric-caption">{unit_detail.archetype['label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Distance to Coastline</div>
                <div class="metric-value">{unit_detail.raw_indicators['dist_to_coastline_km']:.1f} km</div>
                <div class="metric-caption">Direct geodesic distance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("### 🕸️ Multidimensional Radar Profile")
        dist_df = service.df[service.df["district_name"] == unit_detail.district_name]
        dist_medians = {
            "healthcare_deficit": float(dist_df["dim_healthcare_deficit"].median()),
            "demographic_vulnerability": float(dist_df["dim_demographic_vulnerability"].median()),
            "infrastructure_deficit": float(dist_df["dim_infrastructure_deficit"].median()),
            "coastal_exposure": float(dist_df["dim_coastal_exposure"].median()),
        }

        radar_fig = generate_radar_chart(
            unit_scores=unit_detail.dimension_scores,
            district_medians=dist_medians,
            unit_name=unit_detail.unit_name,
            district_name=unit_detail.district_name,
        )
        st.pyplot(radar_fig, use_container_width=True)
        plt.close(radar_fig)

    with right_col:
        st.markdown("### 🗺️ Location & Boundary Context")
        unit_gdf = service.df[service.df["shrid2"] == selected_shrid]
        fmap_unit = build_folium_choropleth_map(
            gdf=unit_gdf,
            metric_col="priority_score",
            show_coastline=True,
            show_healthcare=True,
            selected_unit_shrid=selected_shrid,
        )
        st_folium(fmap_unit, use_container_width=True, height=360, returned_objects=[])

    st.markdown("### 📊 Granular Sector Indicators")
    ind_c1, ind_c2, ind_c3 = st.columns(3)
    raw = unit_detail.raw_indicators

    with ind_c1:
        st.markdown(
            f"""
            **🏥 Healthcare Accessibility**
            - Nearest Hospital: `{raw['nearest_hospital_name']}` ({raw['nearest_hospital_type']})
            - Dist to Hospital: **{raw['dist_to_nearest_hospital_km']:.2f} km**
            - Nearest PHC/CHC: `{raw['nearest_phc_chc_name']}`
            - Dist to PHC/CHC: **{raw['dist_to_nearest_phc_chc_km']:.2f} km**
            """
        )
    with ind_c2:
        st.markdown(
            f"""
            **👥 Demographic Profile**
            - Total Population: **{raw['total_population']:,}**
            - Total Households: **{raw['total_households']:,}**
            - SC/ST Ratio: **{raw['sc_st_ratio']:.1%}**
            - Sex Ratio: **{raw['sex_ratio_female_per_1000_male']:.0f}** F / 1000 M
            """
        )
    with ind_c3:
        st.markdown(
            f"""
            **🏗️ Civil Infrastructure**
            - All-Weather Road: `{'✅ Yes' if raw['has_all_weather_road'] else '❌ No'}`
            - Treated Tap Water: `{'✅ Yes' if raw['has_treated_tap_water'] else '❌ No'}`
            - Drainage System: `{raw['drainage_system']}`
            """
        )

    st.markdown("### 📋 Tailored Disaster Response Interventions")
    st.markdown(f"**Primary Archetype Strategy ({unit_detail.archetype['label']}):**")
    st.info(f"💡 {unit_detail.archetype['recommended_interventions']}")

    st.markdown("**Prioritized Action Checklist:**")
    if raw["dist_to_coastline_km"] < 5.0:
        st.markdown(
            """
            <div class="checklist-item">
                <b>🌊 Storm Surge Evacuation Priority:</b> Community is located &lt; 5 km from open coastline.
                Pre-designate cyclone shelters in inland elevated sectors and inspect evacuation bus routes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if raw["dist_to_nearest_hospital_km"] > 15.0:
        st.markdown(
            f"""
            <div class="checklist-item">
                <b>🚑 Pre-position Mobile Medical Team:</b> Distance to tertiary hospital is {raw['dist_to_nearest_hospital_km']:.1f} km.
                Establish temporary emergency medical post with 72-hour trauma stabilization kit.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not raw["has_all_weather_road"]:
        st.markdown(
            """
            <div class="checklist-item">
                <b>🚜 Road Ingress Hardening:</b> Community lacks confirmed all-weather pucca road.
                High risk of culvert washout; deploy earthmoving JCB machinery in taluk staging area.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="disclaimer-box">
            <b>Data Provenance & Operational Caveat:</b> {unit_detail.provenance_and_limitations['operational_limitations']}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 3: Scenario Comparison Studio (Legacy - Preserved)
# ==============================================================================
def render_page_scenario_comparison(service: DecisionSupportService) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Scenario Comparison & Rank Sensitivity Studio</h1>
            <p class="hero-subtitle">
                Audit how emergency resource allocation priorities diverge between conflicting disaster management policies.
                Identify communities whose priority rank surges or drops dramatically under different weighting schemes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        st.markdown("### 🅰️ Policy Scenario A")
        s_a_name = st.selectbox(
            "Select Baseline Policy (A)",
            options=[
                "Equal Weights (25% each)",
                "Rapid Landfall Focus (60% Coast, 20% Infra, 10% Health, 10% Demo)",
                "Healthcare Isolation Focus (50% Health, 25% Demo, 25% Infra, 0% Coast)",
            ],
            index=0,
        )
        if "Equal Weights" in s_a_name:
            wa_h, wa_d, wa_i, wa_c = 25, 25, 25, 25
        elif "Rapid Landfall" in s_a_name:
            wa_h, wa_d, wa_i, wa_c = 10, 10, 20, 60
        else:
            wa_h, wa_d, wa_i, wa_c = 50, 25, 25, 0

    with sc_col2:
        st.markdown("### 🅱️ Policy Scenario B")
        s_b_name = st.selectbox(
            "Select Counterfactual Policy (B)",
            options=[
                "Rapid Landfall Focus (60% Coast, 20% Infra, 10% Health, 10% Demo)",
                "Healthcare Isolation Focus (50% Health, 25% Demo, 25% Infra, 0% Coast)",
                "Inland Infrastructure Focus (0% Coast, 35% Health, 35% Infra, 30% Demo)",
                "Equal Weights (25% each)",
            ],
            index=0,
        )
        if "Rapid Landfall" in s_b_name:
            wb_h, wb_d, wb_i, wb_c = 10, 10, 20, 60
        elif "Healthcare Isolation" in s_b_name:
            wb_h, wb_d, wb_i, wb_c = 50, 25, 25, 0
        elif "Inland Infrastructure" in s_b_name:
            wb_h, wb_d, wb_i, wb_c = 35, 30, 35, 0
        else:
            wb_h, wb_d, wb_i, wb_c = 25, 25, 25, 25

    req_compare = ScenarioComparisonRequest(
        scenario_a_name=s_a_name,
        scenario_a_weights=WeightConfiguration(healthcare=wa_h, demographic=wa_d, infrastructure=wa_i, coastal=wa_c),
        scenario_b_name=s_b_name,
        scenario_b_weights=WeightConfiguration(healthcare=wb_h, demographic=wb_d, infrastructure=wb_i, coastal=wb_c),
        top_divergent_count=15,
    )

    res_compare = service.compare_scenarios(req_compare)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Spearman Rank Correlation</div>
                <div class="metric-value">{res_compare.spearman_rank_correlation:.3f}</div>
                <div class="metric-caption">1.0 = identical rank ordering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Kendall's Tau</div>
                <div class="metric-value">{res_compare.kendall_tau_correlation:.3f}</div>
                <div class="metric-caption">Pairwise concordance index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mean Rank Shift</div>
                <div class="metric-value">±{res_compare.mean_absolute_rank_shift:.1f}</div>
                <div class="metric-caption">Average positions displaced</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Max Divergence</div>
                <div class="metric-value" style="color: #ef4444;">{res_compare.max_rank_shift}</div>
                <div class="metric-caption">Maximum single unit rank displacement</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🌌 Rank Divergence Scatter Plot")
    wa = req_compare.scenario_a_weights.get_normalized_weights()
    wb = req_compare.scenario_b_weights.get_normalized_weights()

    score_a = (
        wa["healthcare"] * service.df["dim_healthcare_deficit"]
        + wa["demographic"] * service.df["dim_demographic_vulnerability"]
        + wa["infrastructure"] * service.df["dim_infrastructure_deficit"]
        + wa["coastal"] * service.df["dim_coastal_exposure"]
    )
    score_b = (
        wb["healthcare"] * service.df["dim_healthcare_deficit"]
        + wb["demographic"] * service.df["dim_demographic_vulnerability"]
        + wb["infrastructure"] * service.df["dim_infrastructure_deficit"]
        + wb["coastal"] * service.df["dim_coastal_exposure"]
    )

    plot_df = pd.DataFrame(
        {
            "unit_name": service.df["unit_name"],
            "district_name": service.df["district_name"],
            "vulnerability_archetype_label": service.df["vulnerability_archetype_label"],
            "rank_a": score_a.rank(ascending=False, method="min").astype(int),
            "rank_b": score_b.rank(ascending=False, method="min").astype(int),
        }
    )

    chart = generate_rank_scatter_chart(
        df=plot_df,
        col_x="rank_a",
        col_y="rank_b",
        label_x=f"Rank in {s_a_name.split('(')[0]}",
        label_y=f"Rank in {s_b_name.split('(')[0]}",
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### ⚡ Top Divergent Communities (Largest Priority Shift)")
    shift_table = pd.DataFrame([item.model_dump() for item in res_compare.top_divergent_units])
    st.dataframe(shift_table, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 4: Archetypes & Methodology Reference (Legacy - Preserved)
# ==============================================================================
def render_page_archetypes_methodology(service: DecisionSupportService) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">Vulnerability Archetypes & Methodology</h1>
            <p class="hero-subtitle">
                Scientific documentation on the unsupervised K-Means partitioning (k=4),
                dimensional indicators, normalization procedures, and operational limitations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    catalog = service.get_archetype_catalog()

    st.markdown("### 🧩 Discovered Vulnerability Archetypes (k=4)")
    for arch in catalog.archetypes:
        color = ARCHETYPE_COLORS.get(arch.archetype_id, {"hex": "#38bdf8"})["hex"]
        with st.expander(f"Cluster {arch.archetype_id}: {arch.short_label} ({arch.unit_count} Units, {arch.percentage_of_total}%)", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Description:** {arch.detailed_description}")
                st.markdown(f"**Primary Intervention Strategy:** `{arch.primary_intervention_need}`")
                st.markdown(
                    f"**District Breakdown:** Dakshina Kannada: **{arch.count_dakshina_kannada}** | Udupi: **{arch.count_udupi}**"
                )
            with col2:
                st.markdown("**Dimension Deficit Means (0 to 1):**")
                for dim, val in arch.dimension_means.items():
                    st.progress(val, text=f"{dim.replace('dim_', '').replace('_', ' ').title()}: {val:.3f}")

    st.markdown("---")
    st.markdown("### 📐 Normalization & MCDA Scoring Formulation")
    st.markdown(
        r"""
        Preparedness scores are formulated using deterministic Multi-Criteria Decision Analysis (MCDA):
        $$S_i = \sum_{k=1}^4 w_k \cdot D_{ik}$$
        where $D_{ik} \in [0, 1]$ represents the min-max normalized deficit for dimension $k$, and $\sum w_k = 1.0$.
        """
    )

    st.markdown(
        f"""
        <div class="disclaimer-box">
            <b>Clustering Disclaimer:</b> {ARCHETYPE_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN APPLICATION ROUTING
# ==============================================================================
def main():
    client = get_client()
    page, district_filter = render_sidebar(client)

    if "Regional Decision Dashboard" in page:
        render_page_regional_dashboard(client)
    elif "Prioritization Map" in page:
        render_page_prioritization_map(client.service, district_filter)
    elif "Community Deep-Dive" in page:
        render_page_community_deep_dive(client.service)
    elif "Scenario Comparison" in page:
        render_page_scenario_comparison(client.service)
    elif "Archetypes & Methodology" in page:
        render_page_archetypes_methodology(client.service)


if __name__ == "__main__":
    main()
