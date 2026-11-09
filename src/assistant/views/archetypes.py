"""Vulnerability Archetypes: Unsupervised Demographic and Structural Partitioning.

Provides:
- Comprehensive analytical profile of the four discovered K-Means vulnerability archetypes (k=4).
- Exact accounting: 622 units assigned in Dakshina Kannada & Udupi, 1,257 unassigned in Uttara Kannada.
- Detailed cluster cards with operational descriptions, primary intervention needs, and district distributions.
- Comparative dimension deficit bar chart across all four clusters.
- Mathematical formulation of MCDA score calculation and standardization.
- Scientific governance disclaimers distinguishing structural baseline clustering from cyclone hazard prediction.
"""

import pandas as pd
import streamlit as st

from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.charts import generate_archetype_feature_chart
from src.assistant.theme import (
    ARCHETYPE_COLORS,
    render_disclaimer,
    render_header,
    render_kpi_card,
)
from src.models.clustering import ARCHETYPE_DISCLAIMER


def render_archetypes_page(client: DecisionSupportApiClient) -> None:
    """Render the vulnerability archetypes and methodology reference page."""
    st.markdown(
        render_header(
            title="Vulnerability Archetypes and Clustering Methodology",
            subtitle="Scientific documentation and multidimensional profiling of the four unsupervised K-Means archetypes discovered across coastal Karnataka communities.",
            badge="Methodology Reference",
        ),
        unsafe_allow_html=True,
    )

    catalog = client.service.get_archetype_catalog()

    # 1. Macro Breakdown KPIs
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            render_kpi_card("Total Archetypes", "4 Clusters", caption="K-Means (k=4) Partitioning", color="#C18B62"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            render_kpi_card("Assigned Units", "622 Units", caption="Dakshina Kannada & Udupi", color="#8D9A78"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            render_kpi_card("Fallback Units", "1,257 Units", caption="Uttara Kannada (Unassigned)", color="#918170"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            render_kpi_card("Feature Dimensions", "4 Core Sectors", caption="Health, Demo, Infra, Coast", color="#D6B28C"),
            unsafe_allow_html=True,
        )

    # 2. Comparative Dimension Profiles Chart
    st.markdown("### Cross-Archetype Dimension Deficit Comparison")
    st.markdown(
        """
        <div style="color: #BDAE9C; font-size: 0.9rem; margin-bottom: 0.8rem;">
            Mean normalized deficit scores (0.0 = minimal deficit, 1.0 = maximum vulnerability) across the four
            discovered clusters for Healthcare Access, Demographic Vulnerability, Civil Infrastructure, and Coastal Proximity.
        </div>
        """,
        unsafe_allow_html=True,
    )
    dim_means_by_cluster = {arch.archetype_id: arch.dimension_means for arch in catalog.archetypes}
    comp_chart = generate_archetype_feature_chart(dim_means_by_cluster)
    st.altair_chart(comp_chart, use_container_width=True)

    # 3. Discovered Archetype Deep-Dive Cards
    st.markdown("### Discovered Vulnerability Archetypes (k=4)")
    for arch in catalog.archetypes:
        arch_meta = ARCHETYPE_COLORS.get(arch.archetype_id, {"hex": "#C18B62"})
        color = arch_meta["hex"]

        with st.expander(
            f"Cluster {arch.archetype_id}: {arch.short_label} : {arch.unit_count} Units ({arch.percentage_of_total:.1f}% of assigned)",
            expanded=True,
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Detailed Profile:** {arch.detailed_description}")
                st.markdown(
                    f"**Primary Intervention Protocol:** <code style='color:{color}; background:#2B211B; padding:2px 6px; border-radius:3px; border:1px solid #514033;'>{arch.primary_intervention_need}</code>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**District Distribution:** Dakshina Kannada: <b>{arch.count_dakshina_kannada}</b> ({arch.count_dakshina_kannada/arch.unit_count*100:.1f}%) | Udupi: <b>{arch.count_udupi}</b> ({arch.count_udupi/arch.unit_count*100:.1f}%)"
                )
            with col2:
                st.markdown("**Mean Sector Deficit Indicators (0 to 1):**")
                for dim_name, val in arch.dimension_means.items():
                    clean_name = dim_name.replace("dim_", "").replace("_", " ").title()
                    st.progress(val, text=f"{clean_name}: {val:.3f}")

    st.markdown("---")

    # 4. Fallback Architecture for Uttara Kannada
    st.markdown("### Uttara Kannada Sector Fallback Architecture")
    st.markdown(
        """
        <div style="background: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 1.0rem 1.2rem; font-size: 0.88rem; line-height: 1.6; color: #BDAE9C;">
            <b style="color: #F3E9DC;">Why is Uttara Kannada Unassigned in K-Means Clustering?</b><br><br>
            Unsupervised clustering was originally trained and validated exclusively on the 622 units in <b>Dakshina Kannada and Udupi</b> during baseline research.<br><br>
            When expanding to the complete coastal study area (1,879 units), the 1,257 revenue villages and census towns of <b>Uttara Kannada</b> were intentionally evaluated using <b>deterministic three-sector fallback rules</b> rather than forcing artificial cluster labels.<br><br>
            Operational priority bands (Band 1 through Band 4) for Uttara Kannada are assigned rigorously via deterministic triggers:
            coastal proximity (&lt;5 km), emergency shelter capacity deficits (&lt;20), and tertiary medical distance (&lt;35).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 5. MCDA Mathematical Formulation
    st.markdown("### Multi-Criteria Decision Analysis (MCDA) Formulation")
    st.markdown(
        r"""
        Preparedness priority scores are formulated using deterministic Multi-Criteria Decision Analysis (MCDA):
        $$S_i = \sum_{k=1}^4 w_k \cdot D_{ik}$$
        where $D_{ik} \in [0, 1]$ represents the min-max normalized deficit for dimension $k$, and $\sum_{k=1}^4 w_k = 1.0$.

        Dimensions are formulated from raw physical and demographic proxies:
        - **$D_{i, \text{health}}$ (Healthcare Access Deficit):** Geodesic distance to nearest hospital, PHC/CHC facilities, and density of health infrastructure.
        - **$D_{i, \text{demo}}$ (Demographic Vulnerability):** Proportions of vulnerable demographic groups (SC/ST population shares, sex ratio, elderly/child dependency).
        - **$D_{i, \text{infra}}$ (Infrastructure Deficit):** Unpaved road connectivity, lack of treated tap water, and unlined drainage systems.
        - **$D_{i, \text{coast}}$ (Coastal Hazard Exposure):** Geodesic proximity to the Natural Earth 10m coastline and mean digital elevation (SRTM).
        """
    )

    st.markdown(
        render_disclaimer(
            text=ARCHETYPE_DISCLAIMER,
            title="Scientific Clustering Disclaimer",
        ),
        unsafe_allow_html=True,
    )
