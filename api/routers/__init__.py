"""API routers package."""

from api.routers.analytics import router as analytics_router
from api.routers.assistant import router as assistant_router
from api.routers.communities import router as communities_router
from api.routers.dashboard import router as dashboard_router
from api.routers.districts import router as districts_router
from api.routers.health import router as health_router
from api.routers.maps import router as maps_router
from api.routers.models import router as models_router
from api.routers.simulation import router as simulation_router

__all__ = [
    "analytics_router",
    "assistant_router",
    "communities_router",
    "dashboard_router",
    "districts_router",
    "health_router",
    "maps_router",
    "models_router",
    "simulation_router",
]
