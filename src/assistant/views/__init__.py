"""Views package for Cyclone Preparedness Decision Support Platform."""

from src.assistant.views.overview import render_overview_page
from src.assistant.views.gis_explorer import render_gis_explorer_page
from src.assistant.views.community_profile import render_community_profile_page
from src.assistant.views.scenario_analysis import render_scenario_analysis_page
from src.assistant.views.archetypes import render_archetypes_page
from src.assistant.views.reports import render_reports_page
from src.assistant.views.ai_readiness import render_ai_readiness_page
from src.assistant.views.chat_assistant import render_assistant_page

__all__ = [
    "render_overview_page",
    "render_gis_explorer_page",
    "render_community_profile_page",
    "render_scenario_analysis_page",
    "render_archetypes_page",
    "render_reports_page",
    "render_ai_readiness_page",
    "render_assistant_page",
]
