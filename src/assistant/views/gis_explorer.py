"""Regional GIS Explorer in Warm Chocolate Editorial Aesthetic.

Provides:
- Large interactive Folium geospatial map with administrative unit markers across all 1,879 units
  and choropleth polygons across 622 units.
- Multi-criteria geospatial filtering (District, Priority Band, AHP Tier).
- Instant village and town search by name, PC11 ID, or SHRID.
- Map layer controls in coordinated warm palette.
- Interactive administrative unit markers with rich metadata popups.
- Selected unit summary panel with direct navigation to Community Profiles.
- Filtered prioritized communities table and CSV export.
- Strictly zero emojis and zero em dashes.
"""

from typing import Optional
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.maps import build_folium_choropleth_map, build_regional_gis_map
from src.assistant.theme import (
    BAND_COLORS,
    TIER_COLORS,
    render_disclaimer,
    render_empty_state,
    render_filter_banner,
    render_header,
    render_kpi_card,
)


def render_gis_explorer_page(client: DecisionSupportApiClient) -> None:
    """Render the regional GIS explorer page."""
    st.markdown(
        render_header(
            title="Regional Geospatial Intelligence Explorer",
            subtitle="Interactive spatial decision support map exploring administrative units, priority bands, intrinsic AHP tiers, and coastal hazard vulnerability across coastal Karnataka.",
            badge="GIS Studio",
        ),
        unsafe_allow_html=True,
    )

    # Load unit index
    try:
        unit_df = client.get_unit_index()
    except Exception as err:
        st.error(f"Error loading unit spatial index: {err}")
        return

    # 1. Top Filters & Controls
    col_dist, col_band, col_tier = st.columns(3)
    with col_dist:
        gis_dist = st.selectbox(
            "District Scope",
            options=["All Districts (1,879 Units)", "Dakshina Kannada (369 Units)", "Udupi (253 Units)", "Uttara Kannada (1,257 Units)"],
            index=0,
            key="gis_filter_district",
        )
    with col_band:
        gis_band = st.selectbox(
            "Priority Band Scope",
            options=[
                "All Priority Bands",
                "Band 1: Critical Priority",
                "Band 2: Specialized Readiness",
                "Band 3: Moderate Priority",
                "Band 4: Adequate Baseline",
            ],
            index=0,
            key="gis_filter_band",
        )
    with col_tier:
        gis_tier = st.selectbox(
            "AHP Preparedness Tier Scope",
            options=[
                "All Preparedness Tiers",
                "High Preparedness (>60)",
                "Medium Preparedness (40-60)",
                "Low Preparedness (<40)",
            ],
            index=0,
            key="gis_filter_tier",
        )

    # Filter DataFrame
    filtered_df = unit_df.copy()
    if "Dakshina Kannada" in gis_dist:
        filtered_df = filtered_df[filtered_df["district_name"] == "Dakshina Kannada"]
    elif "Udupi" in gis_dist:
        filtered_df = filtered_df[filtered_df["district_name"] == "Udupi"]
    elif "Uttara Kannada" in gis_dist:
        filtered_df = filtered_df[filtered_df["district_name"] == "Uttara Kannada"]

    band_map = {
        "Band 1: Critical Priority": "BAND_1_CRITICAL",
        "Band 2: Specialized Readiness": "BAND_2_SPECIALIZED",
        "Band 3: Moderate Priority": "BAND_3_MODERATE",
        "Band 4: Adequate Baseline": "BAND_4_ADEQUATE",
    }
    if gis_band != "All Priority Bands":
        target_b = band_map.get(gis_band)
        col_b = "provisional_priority_band" if "provisional_priority_band" in filtered_df.columns else "priority_band"
        filtered_df = filtered_df[filtered_df[col_b] == target_b]

    tier_map = {
        "High Preparedness (>60)": "High",
        "Medium Preparedness (40-60)": "Medium",
        "Low Preparedness (<40)": "Low",
    }
    if gis_tier != "All Preparedness Tiers":
        target_t = tier_map.get(gis_tier)
        filtered_df = filtered_df[filtered_df["preparedness_tier"] == target_t]

    # Active filter banner
    n_filtered = len(filtered_df)
    n_total = len(unit_df)
    st.markdown(
        render_filter_banner(
            f"Displaying <b>{n_filtered:,}</b> of <b>{n_total:,}</b> administrative units in spatial viewport."
        ),
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.markdown(
            render_empty_state("No Units Found", "No administrative units match the active filter criteria. Please broaden your scope."),
            unsafe_allow_html=True,
        )
        return

    # 2. Map Layer & Visualization Settings
    col_mode, col_color, col_coast, col_hosp = st.columns([2, 2, 1, 1])

    with col_mode:
        map_mode = st.radio(
            "Visualization Mode",
            options=["Administrative Unit Markers", "Choropleth Polygons (622 Units)"],
            index=0,
            horizontal=True,
            help="Marker mode displays all 1,879 units with coordinates; Choropleth mode renders boundary polygons for Dakshina Kannada and Udupi.",
        )

    with col_color:
        if map_mode == "Administrative Unit Markers":
            color_metric = st.selectbox(
                "Marker Color Metric",
                options=[("priority_band", "Priority Band (Operational Urgency)"), ("preparedness_tier", "AHP Preparedness Tier (Structural)")],
                format_func=lambda x: x[1],
                index=0,
            )[0]
            choropleth_metric = "priority_score"
        else:
            choropleth_metric = st.selectbox(
                "Choropleth Metric Layer",
                options=[
                    ("priority_score", "Composite Priority Score"),
                    ("dim_healthcare_deficit", "Healthcare Access Deficit"),
                    ("dim_demographic_vulnerability", "Demographic Vulnerability"),
                    ("dim_infrastructure_deficit", "Infrastructure Deficit"),
                    ("dim_coastal_exposure", "Coastal Hazard Proximity"),
                    ("vulnerability_archetype_id", "Vulnerability Archetypes (4 Clusters)"),
                ],
                format_func=lambda x: x[1],
                index=0,
            )[0]
            color_metric = "priority_band"

    with col_coast:
        show_coast = st.checkbox("Overlay Coastline (10m)", value=True)

    with col_hosp:
        show_hosp = st.checkbox("Overlay Healthcare", value=False)

    # 3. Unit Search & Quick Selection
    st.markdown("### Community Search and Focus")
    search_col1, search_col2 = st.columns([1, 2])

    # Preset Quick picks
    with search_col1:
        st.markdown("**Regional Quick Picks:**")
        qp1, qp2, qp3 = st.columns(3)
        with qp1:
            pick_tallur = st.button("Tallur (Udupi)", key="btn_qp_tallur")
            pick_shim = st.button("Shimanthur (DK)", key="btn_qp_shim")
        with qp2:
            pick_edmoge = st.button("Edmoge (Udupi)", key="btn_qp_edmoge")
            pick_bhatkal = st.button("Mudbhatkal (UK)", key="btn_qp_bhatkal")
        with qp3:
            pick_siddapur = st.button("Siddapur (UK)", key="btn_qp_siddapur")
            pick_sirsi = st.button("Sirsi (UK)", key="btn_qp_sirsi")

    # Options for selectbox
    name_col = "village_name" if "village_name" in unit_df.columns else "unit_name"
    all_sorted_units = unit_df.sort_values(name_col)
    unit_search_options = [
        f"{row[name_col]} ({row['district_name']}) - PC11: {row['pc11_id']}"
        for _, row in all_sorted_units.iterrows()
    ]
    pc11_map = {opt: str(row["pc11_id"]) for opt, (_, row) in zip(unit_search_options, all_sorted_units.iterrows())}

    # Determine default selection
    default_search_idx = 0
    if pick_tallur:
        default_search_idx = next((i for i, opt in enumerate(unit_search_options) if "Tallur" in opt), 0)
    elif pick_shim:
        default_search_idx = next((i for i, opt in enumerate(unit_search_options) if "Shimanthur" in opt), 0)
    elif pick_edmoge:
        default_search_idx = next((i for i, opt in enumerate(unit_search_options) if "Edmoge" in opt), 0)
    elif pick_bhatkal:
        default_search_idx = next((i for i, opt in enumerate(unit_search_options) if "Mudbhatkal" in opt), 0)
    elif pick_siddapur:
        default_search_idx = next((i for i, opt in enumerate(unit_search_options) if "Siddapur" in opt), 0)
    elif pick_sirsi:
        default_search_idx = next((i for i, opt in enumerate(unit_search_options) if "Sirsi" in opt), 0)
    elif "selected_community_id" in st.session_state:
        cur_id = str(st.session_state["selected_community_id"])
        match_idx = next((i for i, opt in enumerate(unit_search_options) if cur_id in opt), None)
        if match_idx is not None:
            default_search_idx = match_idx

    with search_col2:
        selected_search_opt = st.selectbox(
            "Search Administrative Unit by Name or Census PC11 ID",
            options=unit_search_options,
            index=default_search_idx,
            key="gis_unit_search_box",
        )

    selected_pc11 = pc11_map[selected_search_opt]

    # Map Legend
    st.markdown(
        """
        <div style="background-color: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 0.65rem 1.0rem; margin-bottom: 0.8rem; font-size: 0.82rem; display: flex; flex-wrap: wrap; gap: 1.2rem; align-items: center;">
            <b style="color: #BDAE9C;">Priority Legend:</b>
            <span><span class="badge-pill badge-band-1">Band 1: Critical</span> (&lt;40 score)</span>
            <span><span class="badge-pill badge-band-2">Band 2: Specialized</span> (Targeted Hazards)</span>
            <span><span class="badge-pill badge-band-3">Band 3: Moderate</span> (Secondary)</span>
            <span><span class="badge-pill badge-band-4">Band 4: Adequate</span> (&gt;60 score)</span>
            <span style="color: #918170;">| Ivory Border: Focus Community</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Interactive Map Rendering
    with st.spinner("Rendering geospatial map..."):
        if map_mode == "Choropleth Polygons (622 Units)":
            legacy_gdf = client.service.df
            if "Dakshina Kannada" in gis_dist:
                display_gdf = legacy_gdf[legacy_gdf["district_name"] == "Dakshina Kannada"]
            elif "Udupi" in gis_dist:
                display_gdf = legacy_gdf[legacy_gdf["district_name"] == "Udupi"]
            else:
                display_gdf = legacy_gdf

            fmap = build_folium_choropleth_map(
                gdf=display_gdf,
                metric_col=choropleth_metric,
                show_coastline=show_coast,
                show_healthcare=show_hosp,
            )
        else:
            fmap = build_regional_gis_map(
                unit_df=filtered_df,
                polygon_gdf=client.service.df,
                color_by=color_metric,
                show_polygons=False,
                show_coastline=show_coast,
                show_healthcare=show_hosp,
                selected_pc11_id=selected_pc11,
                max_markers=1879,
            )

        st_folium(fmap, use_container_width=True, height=520, returned_objects=[])

    # 5. Selected Unit Summary Card & Jump Action
    selected_row_match = unit_df[unit_df["pc11_id"] == selected_pc11]
    if not selected_row_match.empty:
        sel_row = selected_row_match.iloc[0]
        u_name = sel_row.get(name_col, "Selected Unit")
        d_name = sel_row.get("district_name", "")
        t_name = sel_row.get("subdistrict_name", "N/A")
        band = sel_row.get("provisional_priority_band", "BAND_4_ADEQUATE")
        tier = sel_row.get("preparedness_tier", "Medium")
        score = float(sel_row.get("preparedness_score", 0.0))
        p_rank = int(sel_row.get("overall_priority_rank", 0))

        st.markdown("### Focus Community Snapshot")
        sc1, sc2, sc3, sc4, sc5 = st.columns([2, 1, 1, 1, 1])
        with sc1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Unit Identity</div>
                    <div class="metric-value" style="font-size: 1.35rem; color: #F3E9DC;">{u_name}</div>
                    <div class="metric-caption">District: <b>{d_name}</b> | Taluk: <b>{t_name}</b> | PC11: <code>{selected_pc11}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                render_kpi_card("AHP Score", f"{score:.1f}", caption=f"Tier: {tier}", color="#C18B62"),
                unsafe_allow_html=True,
            )
        with sc3:
            b_meta = BAND_COLORS.get(band, {"hex": "#D48B47", "short_label": "Band"})
            st.markdown(
                render_kpi_card("Priority Band", b_meta["short_label"], caption=f"Rank #{p_rank} of 1,879", color=b_meta["hex"]),
                unsafe_allow_html=True,
            )
        with sc4:
            arch_lbl = str(sel_row.get("vulnerability_archetype_label", "Uttara Kannada Fallback"))
            if pd.isna(arch_lbl) or not arch_lbl or arch_lbl.lower() == "nan":
                arch_lbl = "Sector Fallback"
            st.markdown(
                render_kpi_card("Archetype", arch_lbl.split()[0], caption=arch_lbl, color="#8D9A78"),
                unsafe_allow_html=True,
            )
        with sc5:
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            if st.button("Open Community Profile", use_container_width=True, type="primary"):
                st.session_state["selected_community_id"] = selected_pc11
                st.session_state["nav_page"] = "Community Profiles"
                st.rerun()

    st.markdown("---")

    # 6. Top Prioritized Communities Table & Export
    st.markdown("### Top Prioritized Communities in Scope")
    col_b = "provisional_priority_band" if "provisional_priority_band" in filtered_df.columns else "priority_band"
    top_units = filtered_df.sort_values("overall_priority_rank", ascending=True).head(20).copy()

    table_records = []
    for rank_idx, (_, r) in enumerate(top_units.iterrows(), 1):
        table_records.append(
            {
                "Overall Rank": int(r.get("overall_priority_rank", rank_idx)),
                "Unit Name": str(r.get(name_col, "")),
                "District": str(r.get("district_name", "")),
                "Subdistrict": str(r.get("subdistrict_name", "N/A")),
                "Priority Band": str(r.get(col_b, "")).replace("_", " ").title(),
                "AHP Score": round(float(r.get("preparedness_score", 0.0)), 2),
                "AHP Tier": str(r.get("preparedness_tier", "")),
                "PC11 ID": str(r.get("pc11_id", "")),
            }
        )
    st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)

    csv_data = top_units.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Top Prioritized Communities (CSV)",
        data=csv_data,
        file_name=f"cyclone_gis_prioritization_{gis_dist.split()[0].lower()}.csv",
        mime="text/csv",
    )
