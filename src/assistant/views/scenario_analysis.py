"""Scenario Analysis: Counterfactual Policy Simulation and Sensitivity Auditing.

Provides:
- Dynamic Multi-Criteria Decision Analysis (MCDA) weight simulation across dimensions.
- Policy presets (Equal weights, Rapid Landfall Hazard, Healthcare Isolation, Inland Resilience, Custom).
- Baseline vs Counterfactual Scenario Comparison studio.
- Sensitivity metrics: Spearman Rank Correlation, Kendall Tau concordance, Mean Rank Shift, Max Divergence.
- Altair Rank Divergence Scatter Plot with 45-degree parity reference line.
- Top Divergent Communities table identifying communities with extreme priority rank displacements.
- Clear distinction between simulated policy weights and observed scientific baselines.
"""

from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.api.schemas import (
    PrioritizationScenarioRequest,
    ScenarioComparisonRequest,
    WeightConfiguration,
)
from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.charts import generate_rank_scatter_chart
from src.assistant.theme import (
    render_disclaimer,
    render_header,
    render_kpi_card,
)


def render_scenario_analysis_page(client: DecisionSupportApiClient) -> None:
    """Render the scenario analysis and rank sensitivity studio page."""
    st.markdown(
        render_header(
            title="Scenario Analysis and Policy Studio",
            subtitle="Simulate emergency resource allocation policies, test counterfactual dimension weightings, and audit rank divergence between conflicting disaster management strategies.",
            badge="Policy Simulation",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_disclaimer(
            text="Simulations in this studio recompute deterministic Multi-Criteria Decision Analysis (MCDA) priority rankings based on user-adjusted policy weights. They reflect hypothetical resource allocation focus and do NOT alter underlying census observations, measured healthcare facility coordinates, or authoritative AHP baseline tiers.",
            title="Simulation Scope and Analytical Precaution",
        ),
        unsafe_allow_html=True,
    )

    tab_sim, tab_compare = st.tabs(["Dynamic Policy Studio", "Scenario Comparison and Sensitivity"])

    # ==========================================================================
    # TAB 1: Dynamic Policy Studio
    # ==========================================================================
    with tab_sim:
        st.markdown("### Policy Weighting Studio")
        col_preset, col_scale, col_dist = st.columns([3, 1, 1.5])

        with col_preset:
            preset = st.selectbox(
                "Load Disaster Response Policy Preset",
                options=[
                    "Equal Weights (Analytical Baseline: 25% each)",
                    "Rapid Landfall Hazard Focus (60% Coastal Proximity, 20% Infra, 10% Health, 10% Demo)",
                    "Rural Healthcare Isolation (50% Healthcare Deficit, 25% Demo, 25% Infra, 0% Coastal)",
                    "Inland Resilience Focus (0% Coastal, 35% Healthcare, 35% Infra, 30% Demo)",
                    "Custom Weight Sliders",
                ],
                index=0,
                key="sa_preset_select",
            )

        with col_scale:
            include_pop = st.checkbox(
                "Scale by Population",
                value=False,
                help="When checked, factors absolute Census population scale into the demographic vulnerability dimension.",
                key="sa_scale_pop",
            )

        with col_dist:
            dist_filter_choice = st.selectbox(
                "District Focus (622 Units)",
                options=["All Coastal Units (622)", "Dakshina Kannada (369)", "Udupi (253)"],
                index=0,
                key="sa_dist_filter",
            )
            dist_filter = None
            if "Dakshina Kannada" in dist_filter_choice:
                dist_filter = "Dakshina Kannada"
            elif "Udupi" in dist_filter_choice:
                dist_filter = "Udupi"

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
            s_health = st.slider("Healthcare Deficit Weight", 0, 100, w_health, key="sa_sl_h")
        with col2:
            s_demo = st.slider("Demographic Vulnerability Weight", 0, 100, w_demo, key="sa_sl_d")
        with col3:
            s_infra = st.slider("Infrastructure Deficit Weight", 0, 100, w_infra, key="sa_sl_i")
        with col4:
            s_coast = st.slider("Coastal Hazard Weight", 0, 100, w_coast, key="sa_sl_c")

        # Normalization
        total_raw = s_health + s_demo + s_infra + s_coast
        if total_raw == 0:
            st.error("At least one dimension weight must be greater than zero.")
            return

        norm_h = s_health / total_raw
        norm_d = s_demo / total_raw
        norm_i = s_infra / total_raw
        norm_c = s_coast / total_raw

        st.markdown(
            f"""
            <div style="background: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 0.75rem 1.1rem; margin-bottom: 1.2rem; font-size: 0.88rem; color: #BDAE9C;">
                <b style="color: #F3E9DC;">Effective Normalized Weights:</b> &nbsp;
                Healthcare: <code style="color:#C18B62; background:#342820; border:1px solid #514033; padding:1px 5px; border-radius:3px;">{norm_h:.1%}</code> &nbsp;|&nbsp;
                Demographic: <code style="color:#C18B62; background:#342820; border:1px solid #514033; padding:1px 5px; border-radius:3px;">{norm_d:.1%}</code> &nbsp;|&nbsp;
                Infrastructure: <code style="color:#C18B62; background:#342820; border:1px solid #514033; padding:1px 5px; border-radius:3px;">{norm_i:.1%}</code> &nbsp;|&nbsp;
                Coastal Proximity: <code style="color:#C18B62; background:#342820; border:1px solid #514033; padding:1px 5px; border-radius:3px;">{norm_c:.1%}</code>
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
            district_filter=dist_filter,
        )

        scenario_res = client.service.compute_prioritization_scenario(req)

        eval_df = client.service.df.copy()
        w_applied = scenario_res.weights_applied
        custom_score = (
            w_applied["healthcare"] * eval_df["dim_healthcare_deficit"]
            + w_applied["demographic"] * eval_df["dim_demographic_vulnerability"]
            + w_applied["infrastructure"] * eval_df["dim_infrastructure_deficit"]
            + w_applied["coastal"] * eval_df["dim_coastal_exposure"]
        ).round(4)

        eval_df["priority_score"] = custom_score
        eval_df["priority_rank"] = custom_score.rank(ascending=False, method="min").astype(int)

        if dist_filter:
            display_df = eval_df[eval_df["district_name"].str.lower() == dist_filter.lower()].copy()
        else:
            display_df = eval_df.copy()

        # Key Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                render_kpi_card("Units Evaluated", f"{len(display_df):,}", caption=dist_filter or "Both Districts", color="#F3E9DC"),
                unsafe_allow_html=True,
            )
        with m2:
            top_unit = display_df.sort_values("priority_score", ascending=False).iloc[0]
            st.markdown(
                render_kpi_card("Top Priority Unit", top_unit["unit_name"], caption=f"Score: {top_unit['priority_score']:.4f}", color="#C95D4A"),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                render_kpi_card("Median Deficit", f"{display_df['priority_score'].median():.4f}", caption="Interquartile baseline", color="#D6B28C"),
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                render_kpi_card("Max Priority Score", f"{display_df['priority_score'].max():.4f}", caption="Highest composite deficit", color="#C95D4A"),
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                render_kpi_card("Min Priority Score", f"{display_df['priority_score'].min():.4f}", caption="Lowest composite deficit", color="#8D9A78"),
                unsafe_allow_html=True,
            )

        # Top-20 Prioritized Communities Table
        st.markdown("### Top-20 Prioritized Communities Under Scenario")
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

        csv_data = top_20.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Top Prioritized Units (CSV)",
            data=csv_data,
            file_name=f"cyclone_prioritization_{preset.split()[0].lower()}.csv",
            mime="text/csv",
        )

    # ==========================================================================
    # TAB 2: Scenario Comparison and Sensitivity
    # ==========================================================================
    with tab_compare:
        st.markdown("### Policy Divergence and Sensitivity Audit")
        sc_col1, sc_col2 = st.columns(2)

        with sc_col1:
            st.markdown("#### Policy Scenario A")
            s_a_name = st.selectbox(
                "Select Baseline Policy (A)",
                options=[
                    "Equal Weights (25% each)",
                    "Rapid Landfall Focus (60% Coast, 20% Infra, 10% Health, 10% Demo)",
                    "Healthcare Isolation Focus (50% Health, 25% Demo, 25% Infra, 0% Coast)",
                ],
                index=0,
                key="sa_compare_a",
            )
            if "Equal Weights" in s_a_name:
                wa_h, wa_d, wa_i, wa_c = 25, 25, 25, 25
            elif "Rapid Landfall" in s_a_name:
                wa_h, wa_d, wa_i, wa_c = 10, 10, 20, 60
            else:
                wa_h, wa_d, wa_i, wa_c = 50, 25, 25, 0

        with sc_col2:
            st.markdown("#### Policy Scenario B")
            s_b_name = st.selectbox(
                "Select Counterfactual Policy (B)",
                options=[
                    "Rapid Landfall Focus (60% Coast, 20% Infra, 10% Health, 10% Demo)",
                    "Healthcare Isolation Focus (50% Health, 25% Demo, 25% Infra, 0% Coast)",
                    "Inland Infrastructure Focus (0% Coast, 35% Health, 35% Infra, 30% Demo)",
                    "Equal Weights (25% each)",
                ],
                index=0,
                key="sa_compare_b",
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

        res_compare = client.service.compare_scenarios(req_compare)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                render_kpi_card("Spearman Rank Correlation", f"{res_compare.spearman_rank_correlation:.3f}", caption="1.0 = identical rank ordering", color="#C18B62"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_kpi_card("Kendall Tau", f"{res_compare.kendall_tau_correlation:.3f}", caption="Pairwise concordance index", color="#8D9A78"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_kpi_card("Mean Rank Shift", f"±{res_compare.mean_absolute_rank_shift:.1f}", caption="Average positions displaced", color="#D48B47"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                render_kpi_card("Max Divergence", f"{res_compare.max_rank_shift}", caption="Single unit maximum displacement", color="#C95D4A"),
                unsafe_allow_html=True,
            )

        st.markdown("### Rank Divergence Scatter Plot")
        wa = req_compare.scenario_a_weights.get_normalized_weights()
        wb = req_compare.scenario_b_weights.get_normalized_weights()

        score_a = (
            wa["healthcare"] * client.service.df["dim_healthcare_deficit"]
            + wa["demographic"] * client.service.df["dim_demographic_vulnerability"]
            + wa["infrastructure"] * client.service.df["dim_infrastructure_deficit"]
            + wa["coastal"] * client.service.df["dim_coastal_exposure"]
        )
        score_b = (
            wb["healthcare"] * client.service.df["dim_healthcare_deficit"]
            + wb["demographic"] * client.service.df["dim_demographic_vulnerability"]
            + wb["infrastructure"] * client.service.df["dim_infrastructure_deficit"]
            + wb["coastal"] * client.service.df["dim_coastal_exposure"]
        )

        plot_df = pd.DataFrame(
            {
                "unit_name": client.service.df["unit_name"],
                "district_name": client.service.df["district_name"],
                "vulnerability_archetype_label": client.service.df["vulnerability_archetype_label"],
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

        st.markdown("### Top Divergent Communities (Largest Priority Shift)")
        shift_table = pd.DataFrame([item.model_dump() for item in res_compare.top_divergent_units])
        st.dataframe(shift_table, use_container_width=True, hide_index=True)
