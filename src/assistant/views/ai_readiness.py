"""AI Readiness and Future Assistant Architecture Interface.

Provides:
- Architectural specification for future Large Language Model (LLM) decision support integration.
- Clear separation between authoritative scientific results (AHP, Census, MoHFW) and generative AI synthesis.
- Retrieval-Augmented Generation (RAG) context payload blueprint.
- Context-aware question builder demonstrating how incident commanders will prompt the model.
- Strictly adheres to Phase 3.1 boundaries: zero external LLM dependencies, zero mock chats, zero API keys.
"""

import streamlit as st

from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.theme import (
    render_disclaimer,
    render_header,
    render_kpi_card,
)


def render_ai_readiness_page(client: DecisionSupportApiClient) -> None:
    """Render the future AI assistant architecture and readiness preview page."""
    st.markdown(
        render_header(
            title="AI Decision Support Assistant: Architecture and Readiness",
            subtitle="Architectural blueprint and integration interface for future LLM decision-support integration, grounding natural language guidance in authoritative scientific metrics.",
            badge="Architecture Preview",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_disclaimer(
            text="This section outlines the architectural readiness for LLM decision support integration. In accordance with strict scientific governance, all generative capabilities are strictly bounded by verified AHP criteria, deterministic priority bands, and historical Census provenance.",
            title="Scientific Governance and Integration Guardrails",
        ),
        unsafe_allow_html=True,
    )

    if st.button("Launch Cyclone Assistant", key="btn_launch_assistant_from_readiness"):
        st.session_state["nav_page"] = "Cyclone Assistant"
        st.rerun()

    # 1. Architecture Pillars
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            render_kpi_card(
                "Authoritative RAG",
                "Deterministic Core",
                caption="Grounded in 1,879 Verified Units",
                color="#C18B62",
                badge_label="Pillar 1",
            ),
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            render_kpi_card(
                "Scientific Guardrails",
                "Strict Boundaries",
                caption="No Invented Real-Time Forecasts",
                color="#8D9A78",
                badge_label="Pillar 2",
            ),
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            render_kpi_card(
                "Explainable Guidance",
                "Source-Backed",
                caption="Explicit Citation of AHP & Census",
                color="#D6B28C",
                badge_label="Pillar 3",
            ),
            unsafe_allow_html=True,
        )

    # 2. RAG Architecture Blueprint
    st.markdown("### Retrieval-Augmented Decision Support Pipeline")
    st.markdown(
        """
        <div style="background: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 1.2rem; margin-bottom: 1.2rem; font-size: 0.88rem; line-height: 1.7; color: #BDAE9C;">
            <b style="color: #F3E9DC;">Phase 3.2 Integration Blueprint:</b><br>
            <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">1. Incident Commander Query</code> &rarr;
            <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">2. Spatial and Sector Context Extraction (SHRID / PC11 / District)</code> &rarr;
            <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">3. Authoritative Metric Retrieval (AHP Score, Priority Band, Healthcare Distance, Archetype)</code> &rarr;
            <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">4. Scientific Guardrail Injection (Precedence of AHP, Vintage Caveats, Non-Real-time Boundaries)</code> &rarr;
            <code style="color: #D6B28C; background: #342820; border: 1px solid #514033; padding: 2px 6px; border-radius: 3px;">5. Grounded Natural Language Action Directives</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Context-Aware Query Inspector (Preview)
    st.markdown("### Context-Aware Query Inspector (Architecture Preview)")
    st.markdown(
        """
        <div style="color: #BDAE9C; font-size: 0.9rem; margin-bottom: 0.8rem;">
            Select a sample incident command inquiry to inspect the exact authoritative payload that will ground future AI synthesis:
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_query = st.selectbox(
        "Select Operational Scenario Template",
        options=[
            "Scenario 1: Why is Edmoge classified as Band 2 Specialized Readiness despite inland location?",
            "Scenario 2: What makes Tallur highly vulnerable despite an overall High AHP score (>70)?",
            "Scenario 3: How are the 1,257 units of Uttara Kannada evaluated without K-Means cluster assignments?",
            "Scenario 4: What is the operational distinction between AHP Preparedness tiers and Logistic Regression probabilities?",
        ],
        index=0,
    )

    col_ctx1, col_ctx2 = st.columns([1, 1])

    with col_ctx1:
        st.markdown("#### Retrieved Scientific Context Payload")
        if "Edmoge" in sample_query:
            unit_eval = client.get_unit("Edmoge")
            st.json(
                {
                    "unit_name": unit_eval.village_name,
                    "district": unit_eval.district_name,
                    "ahp_score": unit_eval.preparedness_score,
                    "priority_band": unit_eval.provisional_priority_band,
                    "triggers": {
                        "hospital_distance_km": 15.2,
                        "pucca_road_access": False,
                        "archetype": "Remote Foothill & Healthcare-Isolated Communities",
                    },
                    "scientific_precaution": "AHP score authoritative. 2011 Census baseline.",
                }
            )
        elif "Tallur" in sample_query:
            unit_eval = client.get_unit("Tallur")
            st.json(
                {
                    "unit_name": unit_eval.village_name,
                    "district": unit_eval.district_name,
                    "ahp_score": unit_eval.preparedness_score,
                    "priority_band": unit_eval.provisional_priority_band,
                    "triggers": {
                        "dist_coast_km": 1.68,
                        "coastal_surge_hazard": True,
                        "archetype": "High-Exposure Coastal Rural Belt",
                    },
                    "scientific_precaution": "AHP High (>60), but coastal proximity <5km promotes to Band 2 readiness.",
                }
            )
        elif "Uttara Kannada" in sample_query:
            st.json(
                {
                    "study_area": "Uttara Kannada (1,257 Units)",
                    "evaluation_mechanism": "Deterministic Three-Sector Fallback",
                    "band_breakdown": {"Band 1": 153, "Band 2": 686, "Band 3": 211, "Band 4": 207},
                    "clustering_status": "Unassigned in K-Means (Trained on DK & Udupi)",
                    "trigger_rules": "Coastal <5km, Shelter <20, Medical <35",
                }
            )
        else:
            st.json(
                {
                    "ahp_preparedness_tier": "Authoritative MCDA baseline based on infrastructure and demographics",
                    "logistic_regression_tier": "Diagnostic surrogate approximating AHP tiers from 26 features",
                    "model_probabilities": "Simplex likelihoods; NOT real-world cyclone strike or damage probabilities",
                    "precedence_rule": "Deterministic AHP strictly supersedes surrogate ML in operational decision-making",
                }
            )

    with col_ctx2:
        st.markdown("#### Scientific Grounding Verification")
        st.markdown(
            """
            <div style="background: #2B211B; border: 1px solid #514033; border-radius: 4px; padding: 1.0rem; font-size: 0.85rem; line-height: 1.6; color: #BDAE9C;">
                <b style="color: #F3E9DC;">Pre-Response Guardrail Checks:</b><br><br>
                <b style="color: #8D9A78;">[Verified]</b> <b style="color: #F3E9DC;">Authority Check:</b> Response strictly constrained to AHP decision matrix.<br>
                <b style="color: #8D9A78;">[Verified]</b> <b style="color: #F3E9DC;">Vintage Verification:</b> 2011 Census disclaimer automatically affixed.<br>
                <b style="color: #8D9A78;">[Verified]</b> <b style="color: #F3E9DC;">Hazard Demarcation:</b> Prohibits simulating unmeasured cyclone wind speeds or live water depths.<br>
                <b style="color: #8D9A78;">[Verified]</b> <b style="color: #F3E9DC;">Operational Directive:</b> Mandates DDMA ground reconnaissance before dispatch.<br>
                <hr style="border-color: #514033; margin: 0.7rem 0;">
                <span style="color: #918170; font-size: 0.8rem;">
                    In Phase 3.2, the LLM will generate natural language operational guidance formatted directly for
                    Incident Response Teams (IRTs), citing specific PC11 and MoHFW data sources.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
