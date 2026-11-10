"""Tool-calling conversational orchestrator for the Cyclone Preparedness Assistant.

Manages:
- The full multi-turn conversation turn cycle across OpenRouter and Gemini providers.
- OpenAI-compatible and Gemini v1beta REST payload formatting with closed-domain scientific guardrails.
- Tool invocation detection, server-side argument validation, and controlled execution.
- Prevention of infinite tool loops (strictly capped at 4 rounds and 8 executions per turn).
- Automatic deterministic fallback when LLM API is unconfigured, unreachable, or rate-limited.
"""

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    from src.assistant.decision_support import DecisionSupportService
    from src.cyclonelm.classification.argument_extractor import ArgumentExtractor
    from src.cyclonelm.classification.refusal_gate import RefusalGate
from src.assistant.llm.client import (
    GeminiClient,
    GeminiClientError,
    OllamaClient,
    OllamaClientError,
    OpenRouterClient,
    OpenRouterClientError,
    get_default_llm_client,
)
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
    QwenIntentParse,
    ToolExecutionRecord,
)
from src.assistant.tools.registry import ToolRegistry, get_default_registry

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
MAX_TOTAL_TOOL_CALLS = 8


def audit_and_guard_answer(answer: str, structured_evidence: Optional[Dict[str, Any]]) -> str:
    """Audit LLM-generated prose against deterministic ground truth.

    If an authoritative preparedness assessment was executed during this turn,
    verifies that the model does not omit or contradict the authoritative AHP score
    or priority band. If omission or contradiction is detected, appends an explicit
    deterministic reconciliation block to prevent operational misrepresentation.
    """
    if not structured_evidence or not isinstance(structured_evidence, dict):
        return answer

    assessment = structured_evidence.get("explain_preparedness_assessment")
    if not isinstance(assessment, dict):
        return answer

    ahp_data = assessment.get("deterministic_ahp")
    prio_data = assessment.get("operational_prioritization")
    score = ahp_data.get("preparedness_score") if isinstance(ahp_data, dict) else None
    band = prio_data.get("priority_band") if isinstance(prio_data, dict) else None
    c_name = assessment.get("community_name") or assessment.get("name", "Community")

    if score is not None and band is not None:
        score_2dec = f"{score:.2f}"
        score_1dec = f"{score:.1f}"
        band_raw = str(band).strip()
        band_clean = band_raw.replace("BAND_", "Band ").replace("_", " ")

        score_in_answer = score_2dec in answer or score_1dec in answer
        band_in_answer = band_raw.lower() in answer.lower() or band_clean.lower() in answer.lower()

        if not (score_in_answer and band_in_answer):
            reconciliation = (
                f"\n\n---\n**Authoritative Metric Verification (Deterministic Ground Truth):**\n"
                f"- **Community:** {c_name}\n"
                f"- **Authoritative AHP Score:** `{score_2dec} / 100.0`\n"
                f"- **Priority Classification:** `{band_raw}`\n"
                f"*Governance Note: Deterministic AHP scores and priority bands maintain absolute operational precedence.*"
            )
            return answer + reconciliation

    return answer


def _clean_evidence_for_prompt(evidence: Any) -> Any:
    """Strip repetitive legal boilerplate from evidence dictionary for prompt brevity."""
    if isinstance(evidence, dict):
        return {
            k: _clean_evidence_for_prompt(v)
            for k, v in evidence.items()
            if k not in ("scientific_precedence_mandate", "disclaimer", "eligibility_rule")
        }
    elif isinstance(evidence, list):
        return [_clean_evidence_for_prompt(x) for x in evidence]
    return evidence


QWEN_SYSTEM_INSTRUCTION = (
    "You are CycloneLM, an AI technical assistant for coastal Karnataka cyclone preparedness intelligence. "
    "Your duty is to assist District Disaster Management Authorities (DDMAs) and emergency planners "
    "by formulating objective, grounded briefing memos using EXCLUSIVELY the provided deterministic evidence.\n\n"
    "STRICT INVARIANTS:\n"
    "1. Base all statements, scores, ranks, and priority tiers strictly on the provided evidence.\n"
    "2. Always cite the exact authoritative AHP preparedness score (0.0 to 100.0) and exact priority band "
    "(Band 1 Critical, Band 2 Specialized, Band 3 Moderate, or Band 4 Adequate) from the evidence. Never alter or contradict them.\n"
    "3. Never invent live weather forecasts, wind speeds, shelter vacancy, hospital bed counts, or road conditions.\n"
    "4. Logistic Regression tiers are empirical surrogate classifications approximating AHP boundaries on Census 2011 indicators, "
    "not physical cyclone hazard risk or casualty forecasts.\n"
    "5. If information is unrecorded in the evidence, explicitly state that it is unrecorded."
)


class ToolCallingOrchestrator:
    """Orchestrates multi-turn conversation and deterministic tool-calling with LLMs."""

    def __init__(
        self,
        client: Optional[Union[OpenRouterClient, GeminiClient]] = None,
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
        self.client = client or get_default_llm_client()
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self._refusal_gate: Optional["RefusalGate"] = None
        self._argument_extractor: Optional["ArgumentExtractor"] = None

    @property
    def refusal_gate(self) -> "RefusalGate":
        """Lazy-loaded RefusalGate for dynamic live telemetry checks."""
        if self._refusal_gate is None:
            from src.cyclonelm.classification.refusal_gate import RefusalGate
            self._refusal_gate = RefusalGate()
        return self._refusal_gate

    @property
    def argument_extractor(self) -> "ArgumentExtractor":
        """Lazy-loaded ArgumentExtractor for entity resolution and Pydantic validation."""
        if self._argument_extractor is None:
            from src.cyclonelm.classification.argument_extractor import ArgumentExtractor
            self._argument_extractor = ArgumentExtractor(service=self.service)
        return self._argument_extractor

    def _build_initial_gemini_contents(self, request: ChatRequest) -> List[Dict[str, Any]]:
        """Construct the Gemini contents array from conversation history and user prompt."""
        contents: List[Dict[str, Any]] = []

        for msg in request.conversation_history or []:
            if msg.role == MessageRole.USER:
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == MessageRole.TOOL:
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": msg.name or "tool", "response": {"result": msg.content}}}],
                })
            else:
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        user_text = request.message.strip()
        if request.community_identifier and request.community_identifier.lower() not in user_text.lower():
            user_text = (
                f"[Selected Community Context: identifier = '{request.community_identifier}']\n"
                f"{user_text}"
            )

        contents.append({"role": "user", "parts": [{"text": user_text}]})
        return contents

    def _build_initial_openrouter_messages(self, request: ChatRequest) -> List[Dict[str, Any]]:
        """Construct the OpenAI-compatible messages array from conversation history and user prompt."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]

        for msg in request.conversation_history or []:
            if msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.TOOL:
                tool_msg: Dict[str, Any] = {"role": "tool", "content": msg.content}
                if msg.tool_call_id:
                    tool_msg["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    tool_msg["name"] = msg.name
                messages.append(tool_msg)
            elif msg.role == MessageRole.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
            else:
                messages.append({"role": "assistant", "content": msg.content})

        user_text = request.message.strip()
        if request.community_identifier and request.community_identifier.lower() not in user_text.lower():
            user_text = (
                f"[Selected Community Context: identifier = '{request.community_identifier}']\n"
                f"{user_text}"
            )

        messages.append({"role": "user", "content": user_text})
        return messages

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
        districts_found = []
        if "dakshina kannada" in query_lower or re.search(r"\bdk\b", query_lower):
            districts_found.append("Dakshina Kannada")
        if "udupi" in query_lower:
            districts_found.append("Udupi")
        if "uttara kannada" in query_lower or re.search(r"\buk\b", query_lower):
            districts_found.append("Uttara Kannada")

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
                pc11_match = re.search(r"\b\d{6}\b", query_text)
                shrid_match = re.search(r"\b\d{2}-\d{2}-\d{3}-\d{5}-\d{6}\b", query_text)
                if pc11_match:
                    target_community = pc11_match.group(0)
                elif shrid_match:
                    target_community = shrid_match.group(0)
                else:
                    for v_name in sorted(all_village_names, key=len, reverse=True):
                        if len(v_name) >= 3 and re.search(r"\b" + re.escape(v_name) + r"\b", query_lower):
                            target_community = v_name
                            break

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
                        answer_text = format_ambiguous_community_response(target_community, matches)
                    else:
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

    async def _process_chat_gemini(
        self,
        request: ChatRequest,
        start_time: float,
        tool_records: List[ToolExecutionRecord],
        tools_executed: List[str],
    ) -> ChatResponse:
        """Execute conversational turn using Google Gemini generateContent REST API."""
        structured_evidence: Dict[str, Any] = {}
        contents = self._build_initial_gemini_contents(request)
        gemini_tools = [{"functionDeclarations": self.registry.export_gemini_tool_declarations()}]

        current_round = 0
        total_calls = 0

        while current_round < self.max_rounds and total_calls < self.max_tool_calls:
            current_round += 1

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

            function_calls = GeminiClient.extract_function_calls(response_json)

            if not function_calls:
                final_answer = GeminiClient.extract_text_response(response_json) or (
                    "Assessment complete. Deterministic evidence retrieved."
                )
                total_duration = round((time.perf_counter() - start_time) * 1000, 2)
                guarded_answer = audit_and_guard_answer(final_answer, structured_evidence)

                return ChatResponse(
                    answer=guarded_answer,
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

            candidates = response_json.get("candidates", [])
            if candidates:
                contents.append(candidates[0].get("content", {}))

            function_response_parts: List[Dict[str, Any]] = []

            for fc in function_calls:
                fn_name = fc.get("name", "")
                fn_args = fc.get("args") or {}

                if total_calls >= self.max_tool_calls:
                    logger.warning(f"Total tool execution limit ({self.max_tool_calls}) reached.")
                    break

                total_calls += 1
                tools_executed.append(fn_name)

                t_tool_start = time.perf_counter()
                tool_res = self.registry.execute(fn_name, fn_args)
                tool_dur = round((time.perf_counter() - t_tool_start) * 1000, 2)
                summary = tool_res.metadata.get("summary") or f"Status: {'ok' if tool_res.success else 'error'}"

                tool_records.append(
                    ToolExecutionRecord(
                        tool_name=fn_name,
                        arguments=fn_args,
                        success=tool_res.success,
                        execution_time_ms=tool_dur,
                        result_summary=summary,
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

                # Safe compaction of large structured tool results
                raw_json_str = json.dumps(result_body, default=str)
                if len(raw_json_str) > 4000:
                    from src.assistant.llm.history import compact_json_structure
                    safe_gemini_result = compact_json_structure(result_body)
                else:
                    safe_gemini_result = result_body

                function_response_parts.append(
                    {
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"result": safe_gemini_result},
                        }
                    }
                )

            contents.append({"role": "user", "parts": function_response_parts})

        logger.info(f"Tool calling budget reached (rounds: {current_round}, calls: {total_calls}). Requesting final answer.")
        try:
            final_resp = await self.client.generate_content(
                contents=contents,
                tools=None,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={"temperature": request.temperature or 0.0},
            )
            final_answer = GeminiClient.extract_text_response(final_resp) or (
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
        guarded_answer = audit_and_guard_answer(final_answer, structured_evidence)

        return ChatResponse(
            answer=guarded_answer,
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

    def _deterministic_intent_parse(
        self,
        query: str,
        community_hint: Optional[str] = None,
    ) -> QwenIntentParse:
        """Rule-based fallback intent and argument parser when Qwen parsing is unavailable or malformed."""
        q_lower = query.lower()

        # 1. District comparison check
        if any(w in q_lower for w in ["compare", "versus", "vs", "difference"]):
            da, db = None, None
            if "dakshina" in q_lower:
                da = "Dakshina Kannada"
            if "udupi" in q_lower:
                if da is None:
                    da = "Udupi"
                else:
                    db = "Udupi"
            if "uttara" in q_lower:
                if da is None:
                    da = "Uttara Kannada"
                else:
                    db = "Uttara Kannada"
            if da and db:
                return QwenIntentParse(
                    intents=["compare_districts"],
                    arguments={"district_a": da, "district_b": db},
                )
            elif da and not db:
                return QwenIntentParse(
                    intents=["compare_districts"],
                    arguments={"district_a": da},
                    needs_clarification=True,
                    clarification_question=f"Comparison requires two districts. Please specify the second district to compare with {da}.",
                )

        # 2. Priority Band filter check
        for band_num in [1, 2, 3, 4]:
            if f"band {band_num}" in q_lower or (band_num == 1 and "critical" in q_lower):
                dist = None
                if "dakshina" in q_lower:
                    dist = "Dakshina Kannada"
                elif "udupi" in q_lower:
                    dist = "Udupi"
                elif "uttara" in q_lower:
                    dist = "Uttara Kannada"
                return QwenIntentParse(
                    intents=["filter_and_rank_communities"],
                    arguments={"priority_band": band_num, "district": dist},
                )

        # 3. Uttara Kannada fallback methodology check
        if "uttara kannada" in q_lower and any(w in q_lower for w in ["fallback", "methodology", "unassigned", "heuristic"]):
            return QwenIntentParse(
                intents=["explain_vulnerability_archetype"],
                arguments={"community_identifier": "Bappalgundi"},
            )

        # 4. Community search/lookup/assessment check
        comm_cand = community_hint
        if not comm_cand:
            words = re.findall(r"\b[A-Za-z]{3,}\b", query)
            for w in words:
                if w.lower() in self.argument_extractor.village_index:
                    comm_cand = w
                    break

        if comm_cand:
            intents = []
            if any(w in q_lower for w in ["where", "locate", "lookup", "look up", "find"]):
                intents.append("lookup_community")
            if any(w in q_lower for w in ["vulnerability", "archetype", "cluster"]):
                intents.append("explain_vulnerability_archetype")
            if any(w in q_lower for w in ["infrastructure", "pucca", "road", "hospital", "exposure"]):
                intents.append("analyze_infrastructure_and_exposure")
            if any(w in q_lower for w in ["intervention", "recommend", "action", "priority"]):
                intents.append("get_intervention_recommendations")
            if not intents or any(w in q_lower for w in ["preparedness", "score", "assessment", "tier", "rank"]):
                intents.append("explain_preparedness_assessment")
                if "get_intervention_recommendations" not in intents:
                    intents.append("get_intervention_recommendations")

            return QwenIntentParse(
                intents=intents,
                arguments={"identifier": comm_cand, "query": comm_cand, "community_identifier": comm_cand},
            )

        return QwenIntentParse(
            intents=[],
            arguments={},
            needs_clarification=True,
            clarification_question="Query is ambiguous or missing community/district details.",
        )

    async def _parse_query_intent_with_qwen(
        self,
        query: str,
        community_hint: Optional[str] = None,
    ) -> QwenIntentParse:
        """Query Qwen via Ollama to identify required tools and structured arguments."""
        prompt = (
            "You are the CYCLONE INTEL intent understanding engine.\n"
            "Analyze the user query and identify the exact deterministic tools needed to answer it.\n\n"
            "Available Deterministic Tools:\n"
            "1. compare_districts: Use for comparing preparedness, vulnerabilities, or priority metrics between two districts. "
            "Arguments: district_a (str), district_b (str).\n"
            "2. explain_preparedness_assessment: Use for AHP composite scores and preparedness tiers for a specific community/village. "
            "Argument: identifier (str).\n"
            "3. analyze_infrastructure_and_exposure: Use for detailed infrastructure vulnerabilities and road access for a specific community/village. "
            "Argument: identifier (str).\n"
            "4. get_intervention_recommendations: Use for prioritized policy interventions for a specific community/village. "
            "Argument: identifier (str).\n"
            "5. filter_and_rank_communities: Use for searching or listing communities by priority band (1-4) or district. "
            "Arguments: priority_band (int 1-4, optional), district (str, optional).\n"
            "6. lookup_community: Use for geographic or demographic lookup, location, or finding a specific village (e.g. 'Where is Belapu', 'Look up Ullal'). "
            "Argument: query (str).\n"
            "7. explain_vulnerability_archetype: Use for K-Means cluster archetypes or regional fallback methodologies. "
            "Arguments: community_identifier (str, optional), archetype_id (int, optional).\n\n"
            "Disambiguation Rules:\n"
            "- If comparing two districts (e.g. 'Dakshina Kannada and Udupi'), select 'compare_districts' as the primary tool. Do NOT select single-community tools unless a specific village is named.\n"
            "- If asking where a community is located or asking to look up a community, choose 'lookup_community' with 'query': <village name>.\n"
            "- If asking about a specific community's assessment or score, select 'explain_preparedness_assessment' and/or 'get_intervention_recommendations'.\n"
            "- If asking for communities in a priority band (e.g. Band 1 Critical), select 'filter_and_rank_communities'.\n"
            "- If asking about Uttara Kannada fallback or unassigned units, select 'explain_vulnerability_archetype'.\n"
            "- If the query is completely outside coastal cyclone disaster preparedness, set 'intents': [] and 'needs_clarification': true.\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "intents": ["tool_name"],\n'
            '  "arguments": {"district_a": "...", "district_b": "...", "identifier": "...", "priority_band": 1},\n'
            '  "needs_clarification": false,\n'
            '  "clarification_question": null\n'
            "}\n\n"
            f'User Query: "{query}"\n'
        )
        if community_hint:
            prompt += f'Context Community: "{community_hint}"\n'

        try:
            raw = await self.client.generate(
                prompt=prompt,
                system="You are a strict JSON intent parsing engine. Return valid JSON only, without commentary.",
                temperature=0.0,
                num_predict=180,
                repeat_penalty=1.15,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            return QwenIntentParse.model_validate(parsed)
        except Exception as exc:
            logger.warning(f"Qwen intent parsing failed ({exc}); falling back to deterministic parser.")
            return self._deterministic_intent_parse(query, community_hint)

    async def _process_chat_ollama(
        self,
        request: ChatRequest,
        start_time: float,
        tool_records: List[ToolExecutionRecord],
        tools_executed: List[str],
    ) -> ChatResponse:
        """Execute conversational turn using local Ollama (Qwen) with structured intent orchestration."""
        query_text = request.message.strip()

        # 1. Refusal check (dynamic operational telemetry / live data)
        # Evaluated BEFORE LLM call for sub-millisecond safety gating
        refusal_res = self.refusal_gate.evaluate(query_text)
        if refusal_res.is_refused:
            total_duration = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"[ASSISTANT RUNTIME] MODE=DETERMINISTIC_REFUSAL | Query='{query_text[:50]}' | "
                f"Duration={total_duration}ms"
            )
            return ChatResponse(
                answer=refusal_res.reason or (
                    "Request involves live dynamic operational telemetry which CYCLONE INTEL does not track. "
                    "Please refer to IMD or district DDMA emergency bulletins."
                ),
                tools_executed=[],
                tool_records=[],
                structured_evidence=None,
                is_fallback=False,
                conversation_status=ConversationStatus.COMPLETED,
                model_name="deterministic-gate",
                total_duration_ms=total_duration,
                scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
            )

        # 2. Qwen Intent Understanding & Argument Extraction
        t_parse_start = time.perf_counter()
        qwen_parse = await self._parse_query_intent_with_qwen(
            query=query_text,
            community_hint=request.community_identifier,
        )
        parse_dur = round((time.perf_counter() - t_parse_start) * 1000, 2)
        logger.info(
            f"Qwen intent parse completed in {parse_dur}ms: intents={qwen_parse.intents}, "
            f"args={qwen_parse.arguments}, needs_clarification={qwen_parse.needs_clarification}"
        )

        # 3. Resolve & filter target tools against registered catalog
        allowed_tools = set(self.registry.get_tool_descriptions().keys())
        target_tools = [t for t in qwen_parse.intents if t in allowed_tools]

        # Domain heuristics fallback if Qwen returned no recognized tools:
        query_lower = query_text.lower()
        if not target_tools:
            if "uttara kannada" in query_lower and any(w in query_lower for w in ["fallback", "methodology", "unassigned", "heuristic"]):
                target_tools = ["explain_vulnerability_archetype"]
                qwen_parse.arguments["community_identifier"] = "Bappalgundi"
            elif any(w in query_lower for w in ["compare", "versus", "vs", "difference"]) and ("dakshina" in query_lower or "udupi" in query_lower):
                target_tools = ["compare_districts"]
            elif any(w in query_lower for w in ["band 1", "band 2", "band 3", "band 4", "critical", "moderate", "adequate"]):
                target_tools = ["filter_and_rank_communities"]

        # If still no valid tools and user needs clarification / out-of-scope:
        if not target_tools:
            total_duration = round((time.perf_counter() - start_time) * 1000, 2)
            if qwen_parse.needs_clarification and qwen_parse.clarification_question:
                msg = qwen_parse.clarification_question
                mode = "DETERMINISTIC_CLARIFICATION"
            else:
                msg = (
                    "Query is outside the scope of coastal Karnataka cyclone preparedness intelligence. "
                    "Supported tasks include community lookups, AHP evaluations, intervention recommendations, "
                    "district comparisons, priority band filtering, and vulnerability archetypes."
                )
                mode = "DETERMINISTIC_OUT_OF_SCOPE"
            logger.info(f"[ASSISTANT RUNTIME] MODE={mode} | Query='{query_text[:50]}' | Duration={total_duration}ms")
            return ChatResponse(
                answer=msg,
                tools_executed=[],
                tool_records=[],
                structured_evidence=None,
                is_fallback=False,
                conversation_status=ConversationStatus.COMPLETED,
                model_name="deterministic-gate",
                total_duration_ms=total_duration,
                scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
            )

        # 4. Multi-Intent Argument Validation & Execution
        structured_evidence: Dict[str, Any] = {}
        is_district_query = "compare_districts" in target_tools or ("district_a" in qwen_parse.arguments and "district_b" in qwen_parse.arguments)

        for tool_name in list(target_tools):
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue

            # Validate arguments through ArgumentExtractor
            ext_res = self.argument_extractor.extract_arguments(tool_name, query_text)

            # Check for ambiguous community matches across districts
            if ext_res.is_ambiguous and not is_district_query:
                total_duration = round((time.perf_counter() - start_time) * 1000, 2)
                candidates = ext_res.ambiguous_candidates
                cand_msg = format_ambiguous_community_response(query_text, candidates) if candidates else (
                    ext_res.validation_error or "Ambiguous community name. Please specify district."
                )
                logger.info(
                    f"[ASSISTANT RUNTIME] MODE=DETERMINISTIC_DISAMBIGUATION | Query='{query_text[:50]}' | "
                    f"Matches={len(candidates)} | Duration={total_duration}ms"
                )
                return ChatResponse(
                    answer=cand_msg,
                    tools_executed=[],
                    tool_records=[],
                    structured_evidence=None,
                    is_fallback=False,
                    conversation_status=ConversationStatus.COMPLETED,
                    model_name="deterministic-router",
                    total_duration_ms=total_duration,
                    scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
                )

            # Build final arguments: prioritize extractor parameters then qwen arguments
            final_args = dict(ext_res.parameters)
            for k, v in qwen_parse.arguments.items():
                if k not in final_args or not final_args[k]:
                    final_args[k] = v

            # Handle missing required arguments
            if tool_name == "compare_districts":
                da = final_args.get("district_a")
                db = final_args.get("district_b")
                if not da or not db:
                    total_duration = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.info(f"[ASSISTANT RUNTIME] MODE=DETERMINISTIC_MISSING_ARGUMENT | Query='{query_text[:50]}'")
                    return ChatResponse(
                        answer="Comparison requires two coastal districts (e.g., 'Compare Dakshina Kannada and Udupi'). Please specify both districts.",
                        tools_executed=[],
                        tool_records=[],
                        structured_evidence=None,
                        is_fallback=False,
                        conversation_status=ConversationStatus.COMPLETED,
                        model_name="deterministic-router",
                        total_duration_ms=total_duration,
                        scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
                    )

            elif tool_name in ("explain_preparedness_assessment", "analyze_infrastructure_and_exposure", "get_intervention_recommendations"):
                if is_district_query and not final_args.get("identifier"):
                    # Omit community tool in a broad district comparison
                    continue
                if not final_args.get("identifier"):
                    if request.community_identifier:
                        final_args["identifier"] = request.community_identifier
                    else:
                        for cand_k in ["community", "village", "query", "community_identifier"]:
                            if qwen_parse.arguments.get(cand_k):
                                final_args["identifier"] = qwen_parse.arguments[cand_k]
                                break

                if not final_args.get("identifier"):
                    total_duration = round((time.perf_counter() - start_time) * 1000, 2)
                    return ChatResponse(
                        answer=f"Please specify the community name to perform {tool_name.replace('_', ' ')}.",
                        tools_executed=[],
                        tool_records=[],
                        structured_evidence=None,
                        is_fallback=False,
                        conversation_status=ConversationStatus.COMPLETED,
                        model_name="deterministic-router",
                        total_duration_ms=total_duration,
                        scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
                    )

            # Execute tool authoritatively
            tools_executed.append(tool_name)
            t_tool_start = time.perf_counter()
            tool_res = self.registry.execute(tool_name, final_args)
            tool_dur = round((time.perf_counter() - t_tool_start) * 1000, 2)
            logger.info(f"[ASSISTANT RUNTIME] TOOL_EXECUTION | '{tool_name}' in {tool_dur}ms | Success: {tool_res.success}")

            summary = tool_res.metadata.get("summary") or f"Status: {'ok' if tool_res.success else 'error'}"
            tool_records.append(
                ToolExecutionRecord(
                    tool_name=tool_name,
                    arguments=final_args,
                    success=tool_res.success,
                    execution_time_ms=tool_dur,
                    result_summary=summary,
                    error=tool_res.error,
                    error_type=tool_res.error_type,
                )
            )

            if tool_res.success and tool_res.data:
                structured_evidence[tool_name] = tool_res.data

        # If explain_preparedness_assessment was executed, also ensure get_intervention_recommendations is retrieved
        if "explain_preparedness_assessment" in structured_evidence and "get_intervention_recommendations" not in structured_evidence:
            ident = structured_evidence["explain_preparedness_assessment"].get("village_name") or request.community_identifier
            if ident:
                adv_res = self.registry.execute("get_intervention_recommendations", {"identifier": ident})
                if adv_res.success and adv_res.data:
                    structured_evidence["get_intervention_recommendations"] = adv_res.data
                    tools_executed.append("get_intervention_recommendations")
                    tool_records.append(
                        ToolExecutionRecord(
                            tool_name="get_intervention_recommendations",
                            arguments={"identifier": ident},
                            success=True,
                            execution_time_ms=1.0,
                            result_summary="Intervention advisories retrieved",
                        )
                    )

        if not structured_evidence:
            total_duration = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer="No authoritative evidence could be retrieved for the specified parameters.",
                tools_executed=tools_executed,
                tool_records=tool_records,
                structured_evidence=None,
                is_fallback=True,
                fallback_reason="All tool executions failed or returned empty data.",
                conversation_status=ConversationStatus.ERROR,
                model_name=self.client.model,
                total_duration_ms=total_duration,
                scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
            )

        # 5. Natural-Language Synthesis via Local Qwen Conditioned on Evidence
        prompt_evidence = _clean_evidence_for_prompt(structured_evidence)
        prompt = (
            f"User Query: {query_text}\n\n"
            f"Authoritative Deterministic Evidence:\n"
            f"```json\n{json.dumps(prompt_evidence, indent=2, default=str)}\n```\n\n"
            f"Instruction: Formulate a concise, evidence-grounded briefing memo answering the user query. "
            f"Explicitly cite the exact authoritative composite scores, priority bands, metrics, and recommendations "
            f"from the evidence above."
        )

        t_gen_start = time.perf_counter()
        try:
            raw_synthesis = await self.client.generate(
                prompt=prompt,
                system=QWEN_SYSTEM_INSTRUCTION,
                temperature=request.temperature or 0.0,
                num_predict=350,
                repeat_penalty=1.15,
            )
            qwen_dur = round((time.perf_counter() - t_gen_start) * 1000, 2)
            final_answer = raw_synthesis or "Analysis complete based on deterministic evidence."
            is_fallback = False
            fallback_reason = None
            logger.info(
                f"[ASSISTANT RUNTIME] MODE=QWEN_SYNTHESIS | Model={self.client.model} | "
                f"GenerationLatency={qwen_dur}ms | Tools={tools_executed}"
            )
        except Exception as qwen_err:
            qwen_dur = round((time.perf_counter() - t_gen_start) * 1000, 2)
            logger.warning(
                f"[ASSISTANT RUNTIME] MODE=DETERMINISTIC_FALLBACK | Reason='{qwen_err}' | "
                f"Latency={qwen_dur}ms | Serving deterministic template briefing."
            )
            is_fallback = True
            fallback_reason = str(qwen_err)
            if "compare_districts" in structured_evidence:
                comp_data = structured_evidence["compare_districts"]
                final_answer = format_district_comparison_response(comp_data, query_text)
            elif "explain_preparedness_assessment" in structured_evidence:
                eval_data = structured_evidence["explain_preparedness_assessment"]
                adv_data = structured_evidence.get("get_intervention_recommendations", {}).get("advisories")
                final_answer = format_community_assessment_response(eval_data, adv_data, query_text, f"Deterministic mode (local LLM: {qwen_err})")
            elif "filter_and_rank_communities" in structured_evidence:
                filter_data = structured_evidence["filter_and_rank_communities"]
                final_answer = format_band_filter_response(filter_data, query_text)
            elif "explain_vulnerability_archetype" in structured_evidence and "uttara kannada" in query_text.lower():
                arch_data = structured_evidence["explain_vulnerability_archetype"]
                reg_summary = self.service.get_regional_decision_summary()
                final_answer = format_uttara_kannada_fallback_response(arch_data, query_text, reg_summary)
            else:
                final_answer = f"### Authoritative Tool Output\n\n```json\n{json.dumps(structured_evidence, indent=2, default=str)}\n```"

        guarded_answer = audit_and_guard_answer(final_answer, structured_evidence)
        total_duration = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatResponse(
            answer=guarded_answer,
            tools_executed=tools_executed,
            tool_records=tool_records,
            structured_evidence=structured_evidence if structured_evidence else None,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            conversation_status=ConversationStatus.COMPLETED if not is_fallback else ConversationStatus.FALLBACK,
            model_name=self.client.model if not is_fallback else "deterministic-fallback",
            total_duration_ms=total_duration,
            scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
        )

    async def _process_chat_openrouter(
        self,
        request: ChatRequest,
        start_time: float,
        tool_records: List[ToolExecutionRecord],
        tools_executed: List[str],
    ) -> ChatResponse:
        """Execute conversational turn using OpenRouter OpenAI-compatible REST API."""
        structured_evidence: Dict[str, Any] = {}
        messages = self._build_initial_openrouter_messages(request)
        openai_tools = self.registry.export_openai_tool_definitions()

        current_round = 0
        total_calls = 0

        while current_round < self.max_rounds and total_calls < self.max_tool_calls:
            current_round += 1

            try:
                response_json = await self.client.create_chat_completion(
                    messages=messages,
                    tools=openai_tools,
                    temperature=request.temperature or 0.0,
                )
            except OpenRouterClientError as client_err:
                logger.warning(f"OpenRouter API invocation failed during round {current_round}: {client_err}")
                return await self._execute_deterministic_fallback(
                    request=request,
                    reason=f"OpenRouter API failure: {str(client_err)}",
                    tool_records=tool_records,
                    tools_executed=tools_executed,
                    start_time=start_time,
                )
            except Exception as unhandled_err:
                logger.error(f"Unexpected error querying OpenRouter API: {unhandled_err}")
                return await self._execute_deterministic_fallback(
                    request=request,
                    reason=f"Unexpected engine error: {type(unhandled_err).__name__}",
                    tool_records=tool_records,
                    tools_executed=tools_executed,
                    start_time=start_time,
                )

            tool_calls = OpenRouterClient.extract_tool_calls(response_json)

            if not tool_calls:
                final_answer = OpenRouterClient.extract_text_response(response_json) or (
                    "Assessment complete. Deterministic evidence retrieved."
                )
                total_duration = round((time.perf_counter() - start_time) * 1000, 2)
                guarded_answer = audit_and_guard_answer(final_answer, structured_evidence)

                return ChatResponse(
                    answer=guarded_answer,
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

            choices = response_json.get("choices", [])
            asst_msg = choices[0].get("message", {}) if choices else {}
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": asst_msg.get("content") or None,
                "tool_calls": asst_msg.get("tool_calls", []),
            }
            messages.append(assistant_entry)

            for tc in tool_calls:
                fn_name = tc.get("name", "")
                fn_args = tc.get("args") or {}
                tc_id = tc.get("id") or f"call_{total_calls}"

                if total_calls >= self.max_tool_calls:
                    logger.warning(f"Total tool execution limit ({self.max_tool_calls}) reached.")
                    break

                total_calls += 1
                tools_executed.append(fn_name)

                t_tool_start = time.perf_counter()
                tool_res = self.registry.execute(fn_name, fn_args)
                tool_dur = round((time.perf_counter() - t_tool_start) * 1000, 2)
                summary = tool_res.metadata.get("summary") or f"Status: {'ok' if tool_res.success else 'error'}"

                tool_records.append(
                    ToolExecutionRecord(
                        tool_name=fn_name,
                        arguments=fn_args,
                        success=tool_res.success,
                        execution_time_ms=tool_dur,
                        result_summary=summary,
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

                # Safe compaction of large structured tool results preserving valid JSON
                raw_json_str = json.dumps(result_body, default=str)
                if len(raw_json_str) > 4000:
                    from src.assistant.llm.history import compact_message_content
                    safe_tool_content = compact_message_content(raw_json_str, max_chars=4000)
                else:
                    safe_tool_content = raw_json_str

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn_name,
                        "content": safe_tool_content,
                    }
                )

        logger.info(f"OpenRouter tool calling budget reached (rounds: {current_round}, calls: {total_calls}). Requesting final answer.")
        try:
            final_resp = await self.client.create_chat_completion(
                messages=messages,
                tools=None,
                temperature=request.temperature or 0.0,
            )
            final_answer = OpenRouterClient.extract_text_response(final_resp) or (
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
        guarded_answer = audit_and_guard_answer(final_answer, structured_evidence)

        return ChatResponse(
            answer=guarded_answer,
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

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a complete conversational turn with LLM and deterministic tools.

        Lifecycle:
        1. Validates LLM API configuration; if unconfigured, triggers immediate fallback.
        2. Packages conversation history and user query into provider-specific schema.
        3. Exposes registered deterministic tools as function declarations.
        4. Manages multi-turn tool calling loop up to 4 rounds / 8 total calls.
        5. Validates and executes tool calls via ToolRegistry.
        6. Obtains final grounded natural language briefing from the LLM.
        7. Returns complete ChatResponse with tool audit trails and precedence notices.
        """
        start_time = time.perf_counter()
        tool_records: List[ToolExecutionRecord] = []
        tools_executed: List[str] = []

        if not self.client.is_configured():
            provider_label = getattr(self.client, "provider_name", "LLM").upper()
            return await self._execute_deterministic_fallback(
                request=request,
                reason=f"{provider_label}_API_KEY is not configured",
                tool_records=tool_records,
                tools_executed=tools_executed,
                start_time=start_time,
            )

        provider = getattr(self.client, "provider_name", "ollama")
        if provider == "ollama":
            return await self._process_chat_ollama(
                request=request,
                start_time=start_time,
                tool_records=tool_records,
                tools_executed=tools_executed,
            )
        elif provider == "openrouter":
            return await self._process_chat_openrouter(
                request=request,
                start_time=start_time,
                tool_records=tool_records,
                tools_executed=tools_executed,
            )
        else:
            return await self._process_chat_gemini(
                request=request,
                start_time=start_time,
                tool_records=tool_records,
                tools_executed=tools_executed,
            )
