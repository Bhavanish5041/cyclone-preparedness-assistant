"""Tool-calling conversational orchestrator for the Cyclone Preparedness Assistant.

Manages:
- The full multi-turn conversation turn cycle.
- Gemini API request formatting with closed-domain scientific guardrails.
- Tool invocation detection, server-side argument validation, and controlled execution.
- Prevention of infinite tool loops (strictly capped at 4 rounds and 8 executions per turn).
- Automatic deterministic fallback when Gemini API is unconfigured, unreachable, or rate-limited.
"""

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.client import GeminiClient, GeminiClientError
from src.assistant.llm.prompts import (
    SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    SYSTEM_INSTRUCTION,
    format_ambiguous_community_response,
    format_band_filter_response,
    format_community_assessment_response,
    format_deterministic_fallback_response,
    format_district_comparison_response,
    format_unsupported_query_response,
    format_uttara_kannada_fallback_response,
)
from src.assistant.llm.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    ToolExecutionRecord,
)
from src.assistant.tools.registry import ToolRegistry, get_default_registry

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
MAX_TOTAL_TOOL_CALLS = 8


class ToolCallingOrchestrator:
    """Orchestrates multi-turn conversation and deterministic tool-calling with Gemini."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        registry: Optional[ToolRegistry] = None,
        service: Optional[DecisionSupportService] = None,
        max_rounds: int = MAX_TOOL_ROUNDS,
        max_tool_calls: int = MAX_TOTAL_TOOL_CALLS,
    ) -> None:
        """Initialize orchestrator with tool registry and LLM client."""
        if service is None:
            from src.assistant.decision_support import DecisionSupportService
            self.service = DecisionSupportService()
        else:
            self.service = service
        self.registry = registry or get_default_registry(service=self.service)
        self.client = client or GeminiClient()
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls

    def _build_initial_contents(self, request: ChatRequest) -> List[Dict[str, Any]]:
        """Construct the Gemini contents array from conversation history and user prompt."""
        contents: List[Dict[str, Any]] = []

        # 1. Process validated prior turns
        for msg in request.conversation_history or []:
            role_str = "user" if msg.role == MessageRole.USER else "model"
            contents.append({"role": role_str, "parts": [{"text": msg.content}]})

        # 2. Prepare user prompt with optional community context injection
        user_text = request.message.strip()
        if request.community_identifier and request.community_identifier.lower() not in user_text.lower():
            user_text = (
                f"[Selected Community Context: identifier = '{request.community_identifier}']\n"
                f"{user_text}"
            )

        contents.append({"role": "user", "parts": [{"text": user_text}]})
        return contents

    async def _execute_deterministic_fallback(
        self,
        request: ChatRequest,
        reason: str,
        tool_records: List[ToolExecutionRecord],
        tools_executed: List[str],
        start_time: float,
    ) -> ChatResponse:
        """Execute deterministic fallback by identifying intent and executing verified tools."""
        logger.info(f"Activating deterministic fallback. Reason: {reason}")
        structured_evidence: Dict[str, Any] = {}
        query_text = request.message.strip()
        query_lower = query_text.lower()

        def _call_tool(tool_name: str, args: Dict[str, Any]) -> Any:
            """Execute a tool through registry and record timing and telemetry."""
            t0 = time.perf_counter()
            res = self.registry.execute(tool_name, args)
            dur = round((time.perf_counter() - t0) * 1000, 2)
            summary = res.metadata.get("summary") or f"Status: {'ok' if res.success else 'error'}"
            tool_records.append(
                ToolExecutionRecord(
                    tool_name=tool_name,
                    arguments=args,
                    success=res.success,
                    execution_time_ms=dur,
                    result_summary=summary,
                    error=res.error,
                    error_type=res.error_type,
                )
            )
            tools_executed.append(tool_name)
            return res

        # Multi-turn context resolution from conversation history
        prior_districts: List[str] = []
        prior_community: Optional[str] = None

        df_villages = self.service.ranked_df
        all_village_names = set(df_villages["village_name"].str.strip().str.lower().unique())

        # Step 1: Scan user messages first for explicitly mentioned districts/communities
        for prior_msg in reversed(request.conversation_history or []):
            p_content = prior_msg.content or ""
            p_lower = p_content.lower()

            if prior_msg.role == MessageRole.USER:
                if len(prior_districts) < 2:
                    for d_name in ["Dakshina Kannada", "Udupi", "Uttara Kannada"]:
                        if (
                            d_name.lower() in p_lower
                            or (d_name == "Dakshina Kannada" and re.search(r"\bdk\b", p_lower))
                            or (d_name == "Uttara Kannada" and re.search(r"\buk\b", p_lower))
                        ) and d_name not in prior_districts:
                            prior_districts.append(d_name)

                if not prior_community:
                    pc11_m = re.search(r"\b\d{6}\b", p_content)
                    shrid_m = re.search(r"\b\d{2}-\d{2}-\d{3}-\d{5}-\d{6}\b", p_content)
                    if pc11_m:
                        prior_community = pc11_m.group(0)
                    elif shrid_m:
                        prior_community = shrid_m.group(0)
                    else:
                        for v_name in sorted(all_village_names, key=len, reverse=True):
                            if len(v_name) >= 3 and re.search(r"\b" + re.escape(v_name) + r"\b", p_lower):
                                prior_community = v_name
                                break

        # Step 2: Fallback to structured headers in assistant messages if user messages were generic
        if len(prior_districts) < 2 or not prior_community:
            for prior_msg in reversed(request.conversation_history or []):
                p_content = prior_msg.content or ""
                if len(prior_districts) < 2:
                    m_comp = re.search(r"### District Comparative Analysis:\s*([A-Za-z\s]+)\s+vs\.\s+([A-Za-z\s]+)", p_content)
                    if m_comp:
                        prior_districts = [m_comp.group(1).strip(), m_comp.group(2).strip()]
                if not prior_community:
                    m_prof = re.search(r"#### Community Profile:\s*([A-Za-z\s\(\)]+)", p_content)
                    if m_prof:
                        raw_cname = m_prof.group(1).split("(")[0].strip()
                        if raw_cname.lower() in all_village_names:
                            prior_community = raw_cname

        # INTENT DETECTION 1: District Comparison
        # Detect if two districts are mentioned
        districts_found = []
        if "dakshina kannada" in query_lower or re.search(r"\bdk\b", query_lower):
            districts_found.append("Dakshina Kannada")
        if "udupi" in query_lower:
            districts_found.append("Udupi")
        if "uttara kannada" in query_lower or re.search(r"\buk\b", query_lower):
            districts_found.append("Uttara Kannada")

        # Multi-turn district inheritance: if current turn doesn't specify two districts, inherit from prior turn
        if len(districts_found) < 2 and len(prior_districts) == 2:
            is_comp_followup = any(
                w in query_lower
                for w in [
                    "compare", "comparison", "versus", "vs", "difference", "between",
                    "which district", "more band", "higher score", "lower score",
                    "average ahp", "ahp score", "average score", "band 1"
                ]
            )
            if is_comp_followup:
                districts_found = list(prior_districts)

        is_compare_query = (
            any(w in query_lower for w in ["compare", "comparison", "versus", "vs", "difference", "between", "which district", "more band", "average ahp"])
            and len(districts_found) >= 2
        ) or len(districts_found) == 2

        if is_compare_query and len(districts_found) >= 2:
            dist_a = districts_found[0]
            dist_b = districts_found[1]
            res_comp = _call_tool("compare_districts", {"district_a": dist_a, "district_b": dist_b})
            if res_comp.success and res_comp.data:
                structured_evidence["compare_districts"] = res_comp.data
                answer_text = format_district_comparison_response(res_comp.data, query_text)
            else:
                answer_text = f"Failed to compare districts: {res_comp.error}"

        # INTENT DETECTION 2: Priority Band Filtering & Ranking
        elif any(w in query_lower for w in ["find", "list", "show", "rank", "filter", "display", "get"]) and any(
            b in query_lower for b in ["band 1", "band 2", "band 3", "band 4", "critical", "specialized", "moderate", "adequate"]
        ):
            target_band = None
            if "band 1" in query_lower or "critical" in query_lower:
                target_band = "BAND_1_CRITICAL"
            elif "band 2" in query_lower or "specialized" in query_lower:
                target_band = "BAND_2_SPECIALIZED"
            elif "band 3" in query_lower or "moderate" in query_lower:
                target_band = "BAND_3_MODERATE"
            elif "band 4" in query_lower or "adequate" in query_lower:
                target_band = "BAND_4_ADEQUATE"

            target_dist = districts_found[0] if len(districts_found) == 1 else None
            args_filter: Dict[str, Any] = {"limit": 10}
            if target_band:
                args_filter["priority_band"] = target_band
            if target_dist:
                args_filter["district"] = target_dist

            res_filter = _call_tool("filter_and_rank_communities", args_filter)
            if res_filter.success and res_filter.data:
                structured_evidence["filter_and_rank_communities"] = res_filter.data
                answer_text = format_band_filter_response(res_filter.data, query_text)
            else:
                answer_text = f"Failed to filter communities: {res_filter.error}"

        # INTENT DETECTION 3: Uttara Kannada Fallback Methodology
        elif ("uttara kannada" in query_lower or re.search(r"\buk\b", query_lower)) and any(
            w in query_lower for w in ["fallback", "methodology", "unassigned", "clustering", "k-means", "sector", "three-sector", "3-sector"]
        ):
            res_arch = _call_tool("explain_vulnerability_archetype", {"community_identifier": "Bappalgundi"})
            reg_summary = self.service.get_regional_decision_summary()
            if res_arch.success and res_arch.data:
                structured_evidence["explain_vulnerability_archetype"] = res_arch.data
                answer_text = format_uttara_kannada_fallback_response(res_arch.data, query_text, reg_summary)
            else:
                answer_text = f"Failed to retrieve fallback methodology: {res_arch.error}"

        # INTENT DETECTION 4: Community Lookup & Preparedness Assessment
        else:
            target_community = request.community_identifier
            if not target_community:
                # 1. Check for 6-digit PC11 ID or SHRID
                pc11_match = re.search(r"\b\d{6}\b", query_text)
                shrid_match = re.search(r"\b\d{2}-\d{2}-\d{3}-\d{5}-\d{6}\b", query_text)
                if pc11_match:
                    target_community = pc11_match.group(0)
                elif shrid_match:
                    target_community = shrid_match.group(0)
                else:
                    # 2. Extract potential village name from query by matching against known dataset villages
                    for v_name in sorted(all_village_names, key=len, reverse=True):
                        if len(v_name) >= 3 and re.search(r"\b" + re.escape(v_name) + r"\b", query_lower):
                            target_community = v_name
                            break

            # Multi-turn community context inheritance
            if not target_community and prior_community:
                is_comm_followup = any(
                    w in query_lower
                    for w in [
                        "infrastructure", "indicator", "score", "contribute", "why", "intervention",
                        "recommend", "advisory", "action", "hazard", "shelter", "hospital", "phc",
                        "road", "water", "archetype", "cluster", "band", "rank", "access", "transit",
                        "health", "demographic", "exposure", "assess"
                    ]
                )
                if is_comm_followup:
                    target_community = prior_community

            if target_community:
                # Call lookup_community to verify existence and check for ambiguities
                res_lookup = _call_tool("lookup_community", {"query": target_community, "limit": 10})
                if res_lookup.success and res_lookup.data:
                    matches = res_lookup.data.get("matches", [])
                    total_matches = res_lookup.data.get("total_matches", len(matches))

                    if total_matches == 0:
                        answer_text = (
                            f"Administrative unit '{target_community}' was not found in the coastal Karnataka study area "
                            "(1,879 surveyed units). Please verify the spelling or provide a Census PC11 ID (e.g. `608691`)."
                        )
                    elif total_matches > 1:
                        # Ambiguous village name matching multiple units across taluks/districts
                        answer_text = format_ambiguous_community_response(target_community, matches)
                    else:
                        # Exactly one matching community resolved
                        match_row = matches[0]
                        resolved_id = str(match_row.get("pc11_id") or match_row.get("shrid2") or target_community)

                        res_eval = _call_tool("explain_preparedness_assessment", {"identifier": resolved_id})
                        if res_eval.success and res_eval.data:
                            structured_evidence["explain_preparedness_assessment"] = res_eval.data

                        res_infra = _call_tool("analyze_infrastructure_and_exposure", {"identifier": resolved_id})
                        if res_infra.success and res_infra.data:
                            structured_evidence["analyze_infrastructure_and_exposure"] = res_infra.data

                        res_adv = _call_tool("get_intervention_recommendations", {"identifier": resolved_id})
                        if res_adv.success and res_adv.data:
                            structured_evidence["get_intervention_recommendations"] = res_adv.data

                        eval_data = res_eval.data if res_eval.success else None
                        infra_data = res_infra.data if res_infra.success else None
                        adv_data = res_adv.data.get("advisories", []) if res_adv.success and res_adv.data else None
                        answer_text = format_community_assessment_response(eval_data, adv_data, query_text, reason, infra_data)
                else:
                    if res_lookup.error_type == "AMBIGUOUS_COMMUNITY":
                        candidates = res_lookup.data.get("candidates", []) if res_lookup.data else []
                        answer_text = format_ambiguous_community_response(target_community, candidates)
                    else:
                        answer_text = f"Community resolution error: {res_lookup.error}"
            else:
                # No supported intent, district, band, or community identifier could be resolved
                answer_text = format_unsupported_query_response(query_text)

        total_duration = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatResponse(
            answer=answer_text,
            tools_executed=tools_executed,
            tool_records=tool_records,
            structured_evidence=structured_evidence if structured_evidence else None,
            is_fallback=True,
            fallback_reason=reason,
            conversation_status=ConversationStatus.FALLBACK,
            model_name="deterministic-fallback",
            total_duration_ms=total_duration,
            scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
        )

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a complete conversational turn with Gemini and deterministic tools.

        Lifecycle:
        1. Validates Gemini API configuration; if unconfigured, triggers immediate fallback.
        2. Packages conversation history and user query into Gemini v1beta schema.
        3. Exposes registered deterministic tools as function declarations.
        4. Manages multi-turn tool calling loop up to 4 rounds / 8 total calls.
        5. Validates and executes tool calls via ToolRegistry.
        6. Obtains final grounded natural language briefing from Gemini.
        7. Returns complete ChatResponse with tool audit trails and precedence notices.
        """
        start_time = time.perf_counter()
        tool_records: List[ToolExecutionRecord] = []
        tools_executed: List[str] = []
        structured_evidence: Dict[str, Any] = {}

        # 1. Check client configuration
        if not self.client.is_configured():
            return await self._execute_deterministic_fallback(
                request=request,
                reason="GEMINI_API_KEY is not configured",
                tool_records=tool_records,
                tools_executed=tools_executed,
                start_time=start_time,
            )

        # 2. Build initial contents
        contents = self._build_initial_contents(request)

        # 3. Prepare Gemini tool declarations
        gemini_tools = [{"functionDeclarations": self.registry.export_gemini_tool_declarations()}]

        current_round = 0
        total_calls = 0

        # 4. Multi-round tool loop
        while current_round < self.max_rounds and total_calls < self.max_tool_calls:
            current_round += 1

            # Dispatch request to Gemini API
            try:
                response_json = await self.client.generate_content(
                    contents=contents,
                    tools=gemini_tools,
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={"temperature": request.temperature or 0.0},
                )
            except GeminiClientError as client_err:
                logger.warning(f"Gemini API invocation failed during round {current_round}: {client_err}")
                return await self._execute_deterministic_fallback(
                    request=request,
                    reason=f"Gemini API failure: {str(client_err)}",
                    tool_records=tool_records,
                    tools_executed=tools_executed,
                    start_time=start_time,
                )
            except Exception as unhandled_err:
                logger.error(f"Unexpected error querying Gemini API: {unhandled_err}")
                return await self._execute_deterministic_fallback(
                    request=request,
                    reason=f"Unexpected engine error: {type(unhandled_err).__name__}",
                    tool_records=tool_records,
                    tools_executed=tools_executed,
                    start_time=start_time,
                )

            # Check if Gemini returned function calls
            function_calls = self.client.extract_function_calls(response_json)

            if not function_calls:
                # No further tools requested; model formulated final answer
                final_answer = self.client.extract_text_response(response_json) or (
                    "Assessment complete. Deterministic evidence retrieved."
                )
                total_duration = round((time.perf_counter() - start_time) * 1000, 2)

                return ChatResponse(
                    answer=final_answer,
                    tools_executed=tools_executed,
                    tool_records=tool_records,
                    structured_evidence=structured_evidence if structured_evidence else None,
                    is_fallback=False,
                    fallback_reason=None,
                    conversation_status=ConversationStatus.COMPLETED,
                    model_name=self.client.model,
                    total_duration_ms=total_duration,
                    scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
                )

            # Function calls present: append model turn to contents
            candidates = response_json.get("candidates", [])
            if candidates:
                contents.append(candidates[0].get("content", {}))

            # Execute requested tools server-side
            function_response_parts: List[Dict[str, Any]] = []

            for fc in function_calls:
                fn_name = fc.get("name", "")
                fn_args = fc.get("args") or {}

                if total_calls >= self.max_tool_calls:
                    logger.warning(f"Total tool execution limit ({self.max_tool_calls}) reached.")
                    break

                total_calls += 1
                tools_executed.append(fn_name)

                # Controlled deterministic execution
                t_tool_start = time.perf_counter()
                tool_res = self.registry.execute(fn_name, fn_args)
                tool_dur = round((time.perf_counter() - t_tool_start) * 1000, 2)

                tool_records.append(
                    ToolExecutionRecord(
                        tool_name=fn_name,
                        arguments=fn_args,
                        success=tool_res.success,
                        execution_time_ms=tool_dur,
                        error=tool_res.error,
                        error_type=tool_res.error_type,
                    )
                )

                if tool_res.success and tool_res.data:
                    structured_evidence[fn_name] = tool_res.data
                    result_body = tool_res.data
                else:
                    result_body = {
                        "error": tool_res.error or "Tool execution failed",
                        "error_type": tool_res.error_type or "TOOL_ERROR",
                    }

                function_response_parts.append(
                    {
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"result": result_body},
                        }
                    }
                )

            # Append tool responses under role "user" per Gemini v1beta schema
            contents.append({"role": "user", "parts": function_response_parts})

        # 5. If loop exited due to round or call limits, make final request to synthesize answer
        logger.info(f"Tool calling budget reached (rounds: {current_round}, calls: {total_calls}). Requesting final answer.")
        try:
            final_resp = await self.client.generate_content(
                contents=contents,
                tools=None,  # Disable tools to force text generation
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={"temperature": request.temperature or 0.0},
            )
            final_answer = self.client.extract_text_response(final_resp) or (
                "Assessment formulated based on executed tool findings."
            )
        except Exception as err:
            logger.warning(f"Final synthesis request failed: {err}. Falling back to deterministic summary.")
            return await self._execute_deterministic_fallback(
                request=request,
                reason=f"Synthesis timeout/error: {err}",
                tool_records=tool_records,
                tools_executed=tools_executed,
                start_time=start_time,
            )

        total_duration = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatResponse(
            answer=final_answer,
            tools_executed=tools_executed,
            tool_records=tool_records,
            structured_evidence=structured_evidence if structured_evidence else None,
            is_fallback=False,
            fallback_reason=None,
            conversation_status=ConversationStatus.COMPLETED,
            model_name=self.client.model,
            total_duration_ms=total_duration,
            scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
        )
