"""Reports and Data Export Center: Structured Administrative and Incident Dossiers.

Provides:
- Lightweight, memory-efficient data export center utilizing native Python and Pandas.
- Regional Preparedness Summary Report (Executive Markdown Dossier and CSV summary).
- Operational Priority Action Lists (CSV exports filtered by Band 1 Critical, Band 2 Specialized, etc.).
- District Preparedness Profiles for Dakshina Kannada, Udupi, and Uttara Kannada.
- Individual Community Briefing Dossier generator (printable/downloadable markdown incident brief).
- Zero external reporting dependencies.
"""

from io import StringIO
import pandas as pd
import streamlit as st

from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.theme import (
    BAND_COLORS,
    TIER_COLORS,
    render_disclaimer,
    render_header,
    render_kpi_card,
)


def render_reports_page(client: DecisionSupportApiClient) -> None:
    """Render the reports and data export page."""
    st.markdown(
        render_header(
            title="Reports and Data Export Center",
            subtitle="Generate standardized regional preparedness summaries, priority action lists, and downloadable incident briefing dossiers for district disaster management authorities.",
            badge="Export Center",
        ),
        unsafe_allow_html=True,
    )

    try:
        unit_df = client.get_unit_index()
        summary = client.get_regional_summary()
    except Exception as err:
        st.error(f"Error accessing platform data for reporting: {err}")
        return

    rep_tab1, rep_tab2, rep_tab3, rep_tab4 = st.tabs(
        [
            "Regional Preparedness Dossier",
            "Priority Band Action Lists",
            "District Profiles",
            "Single Community Briefing Sheet",
        ]
    )

    # ==========================================================================
    # TAB 1: Regional Preparedness Dossier
    # ==========================================================================
    with rep_tab1:
        st.markdown("### Executive Regional Preparedness Summary")
        st.markdown(
            """
            <div style="color: #BDAE9C; font-size: 0.9rem; margin-bottom: 0.8rem;">
                Synthesizes structural capacity across 1,879 administrative units, authoritative AHP scores, and priority bands.
            </div>
            """,
            unsafe_allow_html=True,
        )

        b1 = summary.band_counts.get("BAND_1_CRITICAL", 0)
        b2 = summary.band_counts.get("BAND_2_SPECIALIZED", 0)
        b3 = summary.band_counts.get("BAND_3_MODERATE", 0)
        b4 = summary.band_counts.get("BAND_4_ADEQUATE", 0)
        tot = summary.total_units

        # Generate markdown dossier
        dossier_text = f"""# CYCLONE PREPAREDNESS INTELLIGENCE DOSSIER
**Regional Study Area:** Coastal Karnataka (Dakshina Kannada, Udupi, Uttara Kannada)
**Total Administrative Units:** {tot:,}
**Baseline Vintage:** 2011 Decennial Census and MoHFW Healthcare Directory
**Authority:** Multi-Criteria Decision Analysis (AHP/MCDA)

---

## 1. OPERATIONAL PRIORITY BAND ALLOCATION
- **Band 1 (Critical Priority):** {b1:,} Units ({b1/tot*100:.1f}%) : Urgent evacuation and infrastructure hardening.
- **Band 2 (Specialized Readiness):** {b2:,} Units ({b2/tot*100:.1f}%) : Targeted coastal surge, shelter, or medical deficits.
- **Band 3 (Moderate Priority):** {b3:,} Units ({b3/tot*100:.1f}%) : Secondary baseline readiness.
- **Band 4 (Adequate Baseline):** {b4:,} Units ({b4/tot*100:.1f}%) : Standard vigilance and operational continuity.

---

## 2. DISTRICT CROSS-TABULATION
| District | Total Units | Band 1 (Critical) | Band 2 (Specialized) | Band 3 (Moderate) | Band 4 (Adequate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dakshina Kannada | 369 | 2 | 123 | 45 | 199 |
| Udupi | 253 | 0 | 20 | 43 | 190 |
| Uttara Kannada | 1,257 | 153 | 686 | 211 | 207 |
| **Total Study Area** | **1,879** | **155** | **829** | **299** | **596** |

---

## 3. SCIENTIFIC GOVERNANCE AND LIMITATIONS
- AHP Preparedness scores measure intrinsic multi-sector infrastructural capacity.
- Surrogate Logistic Regression probabilities are uncalibrated diagnostics and do NOT predict physical cyclone paths.
- Census statistics represent 2011 baselines; local ground verification is mandatory.
"""

        st.text_area("Executive Summary Preview (Markdown)", dossier_text, height=220)

        c_down1, c_down2 = st.columns(2)
        with c_down1:
            st.download_button(
                label="Download Executive Dossier (Markdown .md)",
                data=dossier_text.encode("utf-8"),
                file_name="cyclone_regional_preparedness_dossier.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c_down2:
            # Regional summary CSV
            reg_csv_df = pd.DataFrame(
                [
                    {"Metric": "Total Units", "Value": tot},
                    {"Metric": "Band 1 (Critical)", "Value": b1, "Share": f"{b1/tot*100:.1f}%"},
                    {"Metric": "Band 2 (Specialized)", "Value": b2, "Share": f"{b2/tot*100:.1f}%"},
                    {"Metric": "Band 3 (Moderate)", "Value": b3, "Share": f"{b3/tot*100:.1f}%"},
                    {"Metric": "Band 4 (Adequate)", "Value": b4, "Share": f"{b4/tot*100:.1f}%"},
                ]
            )
            st.download_button(
                label="Download Regional Summary (CSV)",
                data=reg_csv_df.to_csv(index=False).encode("utf-8"),
                file_name="cyclone_regional_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ==========================================================================
    # TAB 2: Priority Band Action Lists
    # ==========================================================================
    with rep_tab2:
        st.markdown("### Operational Priority Band Action Lists")
        st.markdown(
            """
            <div style="color: #BDAE9C; font-size: 0.9rem; margin-bottom: 0.8rem;">
                Export complete rosters of administrative units filtered by priority band.
            </div>
            """,
            unsafe_allow_html=True,
        )

        sel_band = st.selectbox(
            "Select Priority Band for Export",
            options=[
                "Band 1 : Critical Priority (155 Units)",
                "Band 2 : Specialized Readiness (829 Units)",
                "Band 3 : Moderate Priority (299 Units)",
                "Band 4 : Adequate Baseline (596 Units)",
            ],
            index=0,
            key="rep_band_select",
        )
        band_key = (
            "BAND_1_CRITICAL"
            if "Band 1" in sel_band
            else (
                "BAND_2_SPECIALIZED"
                if "Band 2" in sel_band
                else ("BAND_3_MODERATE" if "Band 3" in sel_band else "BAND_4_ADEQUATE")
            )
        )

        col_b = "provisional_priority_band" if "provisional_priority_band" in unit_df.columns else "priority_band"
        band_units = unit_df[unit_df[col_b] == band_key].sort_values("overall_priority_rank")

        st.dataframe(
            band_units[
                [
                    "overall_priority_rank",
                    "village_name" if "village_name" in band_units.columns else "unit_name",
                    "district_name",
                    "subdistrict_name",
                    "preparedness_score",
                    "preparedness_tier",
                    "pc11_id",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label=f"Download {sel_band.split('(')[0].strip()} Roster (CSV)",
            data=band_units.to_csv(index=False).encode("utf-8"),
            file_name=f"cyclone_action_list_{band_key.lower()}.csv",
            mime="text/csv",
        )

    # ==========================================================================
    # TAB 3: District Profiles
    # ==========================================================================
    with rep_tab3:
        st.markdown("### District Preparedness Profiles")
        st.markdown(
            """
            <div style="color: #BDAE9C; font-size: 0.9rem; margin-bottom: 0.8rem;">
                Extract district-specific dataset profiles for localized disaster management units.
            </div>
            """,
            unsafe_allow_html=True,
        )

        sel_dist = st.selectbox(
            "Select District",
            options=["Dakshina Kannada (369 Units)", "Udupi (253 Units)", "Uttara Kannada (1,257 Units)"],
            index=0,
            key="rep_dist_select",
        )
        clean_dist = sel_dist.split("(")[0].strip()
        dist_units = unit_df[unit_df["district_name"] == clean_dist].sort_values("overall_priority_rank")

        st.dataframe(
            dist_units[
                [
                    "overall_priority_rank",
                    "village_name" if "village_name" in dist_units.columns else "unit_name",
                    "subdistrict_name",
                    "provisional_priority_band",
                    "preparedness_score",
                    "preparedness_tier",
                    "pc11_id",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label=f"Download {clean_dist} Complete Roster (CSV)",
            data=dist_units.to_csv(index=False).encode("utf-8"),
            file_name=f"cyclone_profile_{clean_dist.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # ==========================================================================
    # TAB 4: Single Community Briefing Sheet
    # ==========================================================================
    with rep_tab4:
        st.markdown("### Individual Community Inspection Dossier")
        st.markdown(
            """
            <div style="color: #BDAE9C; font-size: 0.9rem; margin-bottom: 0.8rem;">
                Generate and print an operational incident briefing sheet for any selected unit.
            </div>
            """,
            unsafe_allow_html=True,
        )

        name_c = "village_name" if "village_name" in unit_df.columns else "unit_name"
        sorted_units = unit_df.sort_values(name_c)
        comm_options = [
            f"{r[name_c]} ({r['district_name']}) | PC11: {r['pc11_id']}"
            for _, r in sorted_units.iterrows()
        ]
        pc11_lookup = {opt: str(r["pc11_id"]) for opt, (_, r) in zip(comm_options, sorted_units.iterrows())}

        sel_comm_opt = st.selectbox("Select Community", options=comm_options, index=0, key="rep_comm_select")
        target_pc11 = pc11_lookup[sel_comm_opt]

        try:
            eval_unit = client.get_unit(target_pc11)
            brief_text = f"""# DISASTER INCIDENT BRIEFING SHEET
**Unit Name:** {eval_unit.village_name} ({eval_unit.unit_type or 'VILLAGE'})
**District:** {eval_unit.district_name} | **Taluk:** {eval_unit.subdistrict_name}
**Census PC11 ID:** {eval_unit.pc11_id} | **SHRID:** {eval_unit.shrid2}
**Coordinates:** {eval_unit.lat:.4f}° N, {eval_unit.lon:.4f}° E

---

## 1. PREPAREDNESS AND PRIORITY CLASSIFICATION
- **AHP Preparedness Score:** {eval_unit.preparedness_score:.2f} / 100.0 (Tier: {eval_unit.preparedness_tier})
- **Operational Priority Band:** {eval_unit.provisional_priority_band}
- **Regional Priority Rank:** #{eval_unit.overall_priority_rank} of 1,879 (Band Rank: #{eval_unit.band_rank})
- **Vulnerability Archetype:** {eval_unit.vulnerability_archetype_label or 'Uttara Kannada Sector Fallback'}

## 2. SURROGATE MACHINE LEARNING DIAGNOSTICS
- **Predicted Tier:** {eval_unit.predicted_preparedness_tier}
- **Simplex Class Likelihoods:** Low: {eval_unit.class_probabilities.get('prob_low', 0.0):.1%}, Medium: {eval_unit.class_probabilities.get('prob_medium', 0.0):.1%}, High: {eval_unit.class_probabilities.get('prob_high', 0.0):.1%}
- **Agreement Status:** {eval_unit.model_agreement_status} | **Boundary Case:** {eval_unit.is_boundary_case}

## 3. OPERATIONAL RATIONALE AND DIRECTIVES
**Primary Rationale:**
{eval_unit.priority_explanation}

**Action Directives:**
"""
            for a in eval_unit.recommended_advisories:
                brief_text += f"- **[{a.directive_id}] {a.advisory_focus}:** {a.operational_recommendation} (Lead: {a.lead_coordinating_agency})\n"

            brief_text += f"\n---\n*Notice:* {eval_unit.advisory_disclaimer}\n"

            st.text_area("Incident Briefing Preview", brief_text, height=220)

            st.download_button(
                label=f"Download {eval_unit.village_name} Briefing Sheet (.md)",
                data=brief_text.encode("utf-8"),
                file_name=f"briefing_{eval_unit.village_name.lower().replace(' ', '_')}_{eval_unit.pc11_id}.md",
                mime="text/markdown",
            )
        except Exception as err:
            st.error(f"Error preparing community briefing dossier: {err}")
