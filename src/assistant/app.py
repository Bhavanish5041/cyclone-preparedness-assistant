"""Cyclone Preparedness Intelligence Platform: Spatial Decision Support System.

Streamlit multi-page platform for coastal Karnataka (Dakshina Kannada, Udupi, and Uttara Kannada).
Provides:
- Section 1: Overview: Macroscopic regional preparedness overview across all 1,879 units.
- Section 2: Regional GIS Explorer: Spatial map, priority-band layers, filters, and community search.
- Section 3: Community Profiles: Granular multidimensional indicators, AHP scores, and response directives.
- Section 4: Scenario Analysis: Dynamic MCDA weight simulation and counterfactual sensitivity auditing.
- Section 5: Vulnerability Archetypes: K-Means (k=4) cluster profiling and Uttara Kannada fallback reference.
- Section 6: Reports and Export: Standardized executive dossiers and downloadable priority action lists.
- Section 7: AI Readiness: Architectural blueprint for future LLM decision support integration.

Architecture:
- Utilizes DecisionSupportApiClient in Direct Mode for low-latency standalone single-process execution.
- Strictly adheres to the 512 MB RAM budget of the Render Free-Tier deployment environment.
- Preserves full scientific and functional integrity across all analytical modules.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import streamlit as st

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.assistant.api_client import DecisionSupportApiClient, get_api_client
from src.assistant.theme import get_custom_css
from src.assistant.views import (
    render_ai_readiness_page,
    render_archetypes_page,
    render_assistant_page,
    render_community_profile_page,
    render_gis_explorer_page,
    render_overview_page,
    render_reports_page,
    render_scenario_analysis_page,
)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Cyclone Preparedness Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject refined custom CSS styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Cyclone Preparedness Client...")
def get_client() -> DecisionSupportApiClient:
    """Initialize and cache the DecisionSupportApiClient singleton."""
    adapter_mode = os.getenv("CYCLONE_ADAPTER_MODE", "direct")
    return get_api_client(mode=adapter_mode)


def render_sidebar(client: DecisionSupportApiClient) -> str:
    """Render the sidebar with brand identity, structured navigation, and architecture telemetry."""
    st.sidebar.markdown(
        """
        <div style="text-align: left; padding: 0.6rem 0.2rem 1.0rem 0.2rem; border-bottom: 1px solid #514033; margin-bottom: 1.0rem;">
            <div style="color: #F3E9DC; font-family: 'DM Serif Display', Georgia, serif; font-size: 1.35rem; font-weight: normal; letter-spacing: 0.01em;">Cyclone Intelligence</div>
            <div style="color: #918170; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;">Preparedness Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "Overview",
        "Regional GIS Explorer",
        "Community Profiles",
        "Scenario Analysis",
        "Vulnerability Archetypes",
        "Reports and Export",
        "Cyclone Assistant",
        "AI Readiness (Preview)",
    ]

    # Check URL query parameters for direct page linking
    query_page = st.query_params.get("page") or st.query_params.get("nav")
    if query_page:
        clean_qp = str(query_page).lower().replace("-", "_")
        for opt in nav_options:
            clean_opt = opt.lower().replace(" ", "_")
            if clean_qp in clean_opt:
                st.session_state["nav_page"] = opt
                break

    # Initialize or synchronize session state for navigation
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = nav_options[0]

    current_idx = 0
    if st.session_state["nav_page"] in nav_options:
        current_idx = nav_options.index(st.session_state["nav_page"])

    selected_page = st.sidebar.radio(
        "Navigation",
        options=nav_options,
        index=current_idx,
        key="platform_nav_radio",
    )

    # Update session state if user clicks radio
    st.session_state["nav_page"] = selected_page

    st.sidebar.markdown("---")

    # Platform telemetry and study area context
    with st.sidebar.expander("Platform Telemetry", expanded=True):
        st.markdown(
            f"""
            **Integration Adapter**
            - Mode: `{client.mode.value.upper()}` (In-Process)
            - Total Study Area: `1,879 Units`
            - Districts: `Dakshina Kannada`, `Udupi`, `Uttara Kannada`

            **Scientific Precedence**
            - Authoritative: `Deterministic AHP`
            - Surrogate ML: `Logistic Reg (v0)`
            - Archetypes: `K-Means (k=4)`
            """
        )

    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; color: #918170; margin-top: 1.8rem; padding: 0.5rem; border-top: 1px solid #514033;">
            RV College of Engineering &bull; Dept. of CSE (AI & ML)<br>
            Coastal Karnataka Disaster Resilience Initiative
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_page


def main():
    """Main application controller and view router."""
    client = get_client()
    selected_page = render_sidebar(client)

    if "Overview" in selected_page:
        render_overview_page(client)
    elif "Regional GIS Explorer" in selected_page:
        render_gis_explorer_page(client)
    elif "Community Profiles" in selected_page:
        render_community_profile_page(client)
    elif "Scenario Analysis" in selected_page:
        render_scenario_analysis_page(client)
    elif "Vulnerability Archetypes" in selected_page:
        render_archetypes_page(client)
    elif "Reports and Export" in selected_page or "Reports" in selected_page:
        render_reports_page(client)
    elif "Cyclone Assistant" in selected_page or "Assistant" in selected_page:
        render_assistant_page(client)
    elif "AI Readiness" in selected_page:
        render_ai_readiness_page(client)


if __name__ == "__main__":
    main()
