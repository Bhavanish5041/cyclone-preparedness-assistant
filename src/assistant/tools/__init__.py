"""Assistant tool suite package for Cyclone Preparedness Intelligence Platform.

Exposes:
- `BaseTool`, `ToolResult`, and tool error exceptions.
- The seven deterministic tools:
  1. `lookup_community` (LookupCommunityTool)
  2. `explain_preparedness_assessment` (ExplainPreparednessAssessmentTool)
  3. `compare_districts` (CompareDistrictsTool)
  4. `filter_and_rank_communities` (FilterAndRankCommunitiesTool)
  5. `analyze_infrastructure_and_exposure` (AnalyzeInfrastructureAndExposureTool)
  6. `explain_vulnerability_archetype` (ExplainVulnerabilityArchetypeTool)
  7. `get_intervention_recommendations` (GetInterventionRecommendationsTool)
- Centralized `ToolRegistry` and `get_default_registry`.
"""

from src.assistant.tools.archetypes import (
    ExplainVulnerabilityArchetypeInput,
    ExplainVulnerabilityArchetypeTool,
)
from src.assistant.tools.base import (
    AmbiguousCommunityError,
    BaseTool,
    CommunityNotFoundError,
    ToolError,
    ToolNotFoundError,
    ToolResult,
    ToolValidationError,
)
from src.assistant.tools.community import (
    AnalyzeInfrastructureAndExposureInput,
    AnalyzeInfrastructureAndExposureTool,
    LookupCommunityInput,
    LookupCommunityTool,
    resolve_community_row,
)
from src.assistant.tools.decision import (
    ExplainPreparednessAssessmentInput,
    ExplainPreparednessAssessmentTool,
    GetInterventionRecommendationsInput,
    GetInterventionRecommendationsTool,
)
from src.assistant.tools.regional import (
    CompareDistrictsInput,
    CompareDistrictsTool,
    FilterAndRankCommunitiesInput,
    FilterAndRankCommunitiesTool,
)
from src.assistant.tools.argument_extractor import (
    ArgumentExtractor,
    ExtractionResult,
)
from src.assistant.tools.registry import (
    ToolRegistry,
    get_default_registry,
    sanitize_gemini_schema,
)

__all__ = [
    # Base contracts
    "BaseTool",
    "ToolResult",
    "ToolError",
    "ToolValidationError",
    "ToolNotFoundError",
    "CommunityNotFoundError",
    "AmbiguousCommunityError",
    # Tools
    "LookupCommunityTool",
    "AnalyzeInfrastructureAndExposureTool",
    "ExplainPreparednessAssessmentTool",
    "GetInterventionRecommendationsTool",
    "CompareDistrictsTool",
    "FilterAndRankCommunitiesTool",
    "ExplainVulnerabilityArchetypeTool",
    # Schemas
    "LookupCommunityInput",
    "AnalyzeInfrastructureAndExposureInput",
    "ExplainPreparednessAssessmentInput",
    "GetInterventionRecommendationsInput",
    "CompareDistrictsInput",
    "FilterAndRankCommunitiesInput",
    "ExplainVulnerabilityArchetypeInput",
    # Utilities & Registry
    "resolve_community_row",
    "ToolRegistry",
    "get_default_registry",
    "sanitize_gemini_schema",
]
