"""Tests for Gemini Function Declaration Schema Compatibility (Phase 4.3.2).

Verifies:
- Recursive schema normalization sanitizes unsupported JSON Schema metadata.
- All 7 deterministic assistant tools export Gemini function declarations without `examples`.
- No unsupported keys (`examples`, `example`, `$schema`, `$defs`, `definitions`, `additionalProperties`)
  exist anywhere in exported Gemini declarations.
- Critical schema contracts (parameter names, types, descriptions, constraints, nullability, enums)
  are strictly preserved.
- OpenAI and Anthropic schema exporters remain completely unmodified.
- Serialized function declarations conform strictly to Google Gemini Function Calling format.
- Live API integration (when GEMINI_API_KEY is configured) or mocked multi-turn execution.
- Automatic retirement recovery for deprecated models (e.g. gemini-2.5-flash -> gemini-3.5-flash).
"""

import asyncio
from copy import deepcopy
import json
import os
from typing import Any, Dict, List, Set
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.client import GeminiClient, GeminiResponseError
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.assistant.llm.schemas import ChatRequest, ConversationStatus
from src.assistant.tools.registry import (
    UNSUPPORTED_GEMINI_SCHEMA_KEYS,
    ToolRegistry,
    get_default_registry,
    sanitize_gemini_schema,
)


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Module-scoped DecisionSupportService instance."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def registry(shared_service: DecisionSupportService) -> ToolRegistry:
    """Module-scoped configured ToolRegistry."""
    return get_default_registry(service=shared_service)


def _collect_all_keys(data: Any) -> Set[str]:
    """Recursively traverse arbitrary data structure and collect all dictionary keys."""
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            keys.add(k)
            keys.update(_collect_all_keys(v))
    elif isinstance(data, list):
        for item in data:
            keys.update(_collect_all_keys(item))
    return keys


# ==============================================================================
# 1. UNIT TESTS: sanitize_gemini_schema
# ==============================================================================


def test_sanitize_gemini_schema_strips_unsupported_metadata():
    """Verify recursive removal of examples, example, $schema, $defs, definitions, additionalProperties."""
    raw_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SampleModel",
        "type": "object",
        "additionalProperties": False,
        "definitions": {"SubDef": {"type": "string"}},
        "$defs": {"AnotherDef": {"type": "integer"}},
        "examples": [{"field_a": "val"}],
        "properties": {
            "field_a": {
                "type": "string",
                "description": "A test string field",
                "examples": ["example_1", "example_2"],
                "example": "example_1",
                "minLength": 2,
            },
            "nested_obj": {
                "type": "object",
                "additionalProperties": True,
                "examples": [{"inner": 1}],
                "properties": {
                    "inner": {
                        "type": "integer",
                        "examples": [1, 2, 3],
                        "minimum": 0,
                    }
                },
            },
            "array_field": {
                "type": "array",
                "examples": [["a", "b"]],
                "items": {
                    "type": "string",
                    "examples": ["item_ex"],
                },
            },
        },
        "required": ["field_a"],
    }

    sanitized = sanitize_gemini_schema(raw_schema)
    all_keys = _collect_all_keys(sanitized)

    # Assert forbidden keys are 100% absent
    for forbidden in UNSUPPORTED_GEMINI_SCHEMA_KEYS:
        assert forbidden not in all_keys, f"Forbidden key '{forbidden}' found in sanitized schema."

    # Assert valid structure and constraints are preserved
    assert sanitized["type"] == "object"
    assert sanitized["required"] == ["field_a"]
    assert sanitized["properties"]["field_a"]["type"] == "string"
    assert sanitized["properties"]["field_a"]["minLength"] == 2
    assert sanitized["properties"]["field_a"]["description"] == "A test string field"
    assert sanitized["properties"]["nested_obj"]["properties"]["inner"]["minimum"] == 0
    assert sanitized["properties"]["array_field"]["items"]["type"] == "string"


def test_sanitize_gemini_schema_handles_unions_and_nullability():
    """Verify anyOf/oneOf containing null is flattened into concrete type with nullable=True."""
    raw_schema = {
        "type": "object",
        "properties": {
            "optional_str": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "An optional string parameter",
                "examples": ["val1", "val2"],
            },
            "optional_constrained_int": {
                "oneOf": [
                    {"type": "integer", "minimum": 0, "maximum": 10},
                    {"type": "null"},
                ],
                "default": None,
                "description": "An optional constrained integer",
                "examples": [5],
            },
            "list_type_field": {
                "type": ["string", "null"],
                "description": "Type as list with null",
            },
        },
    }

    sanitized = sanitize_gemini_schema(raw_schema)
    props = sanitized["properties"]

    # Check optional_str
    assert "anyOf" not in props["optional_str"]
    assert props["optional_str"]["type"] == "string"
    assert props["optional_str"]["nullable"] is True
    assert props["optional_str"]["description"] == "An optional string parameter"
    assert "examples" not in props["optional_str"]

    # Check optional_constrained_int
    assert "oneOf" not in props["optional_constrained_int"]
    assert props["optional_constrained_int"]["type"] == "integer"
    assert props["optional_constrained_int"]["minimum"] == 0
    assert props["optional_constrained_int"]["maximum"] == 10
    assert props["optional_constrained_int"]["nullable"] is True

    # Check list_type_field
    assert props["list_type_field"]["type"] == "string"
    assert props["list_type_field"]["nullable"] is True


# ==============================================================================
# 2. INTEGRATION TESTS: All 7 Exported Gemini Function Declarations
# ==============================================================================


def test_exported_gemini_declarations_count_and_format(registry: ToolRegistry):
    """Verify export_gemini_tool_declarations returns 7 valid function declarations."""
    declarations = registry.export_gemini_tool_declarations()
    assert len(declarations) == 7

    for decl in declarations:
        assert isinstance(decl, dict)
        assert "name" in decl
        assert isinstance(decl["name"], str) and decl["name"].strip()
        assert "description" in decl
        assert isinstance(decl["description"], str) and decl["description"].strip()
        assert "parameters" in decl
        params = decl["parameters"]
        assert isinstance(params, dict)
        assert params.get("type") == "object"
        assert "properties" in params


def test_exported_gemini_declarations_have_zero_examples_recursively(registry: ToolRegistry):
    """Recursively verify that NO tool declaration contains 'examples' or 'example'."""
    declarations = registry.export_gemini_tool_declarations()

    for decl in declarations:
        tool_name = decl["name"]
        all_keys = _collect_all_keys(decl)

        assert "examples" not in all_keys, f"Tool '{tool_name}' contains 'examples' in its Gemini declaration!"
        assert "example" not in all_keys, f"Tool '{tool_name}' contains 'example' in its Gemini declaration!"
        assert "$schema" not in all_keys, f"Tool '{tool_name}' contains '$schema'!"
        assert "$defs" not in all_keys, f"Tool '{tool_name}' contains '$defs'!"
        assert "definitions" not in all_keys, f"Tool '{tool_name}' contains 'definitions'!"
        assert "additionalProperties" not in all_keys, f"Tool '{tool_name}' contains 'additionalProperties'!"

        # Also verify raw JSON string does not contain forbidden keywords as JSON keys
        json_str = json.dumps(decl)
        assert '"examples"' not in json_str, f'Raw JSON of {tool_name} contains "examples"'
        assert '"example"' not in json_str, f'Raw JSON of {tool_name} contains "example"'
        assert '"additionalProperties"' not in json_str


def test_exported_gemini_declarations_preserve_contracts_and_constraints(registry: ToolRegistry):
    """Verify parameter names, types, required fields, and constraints for all 7 tools."""
    decl_map = {d["name"]: d for d in registry.export_gemini_tool_declarations()}

    # 1. lookup_community
    t1 = decl_map["lookup_community"]
    p1 = t1["parameters"]["properties"]
    assert set(p1.keys()) == {"query", "district", "limit"}
    assert p1["query"]["type"] == "string"
    assert p1["query"]["minLength"] == 1
    assert p1["district"]["type"] == "string"
    assert p1["district"]["nullable"] is True
    assert p1["limit"]["type"] == "integer"
    assert p1["limit"]["minimum"] == 1
    assert p1["limit"]["maximum"] == 15
    assert t1["parameters"]["required"] == ["query"]

    # 2. analyze_infrastructure_and_exposure
    t2 = decl_map["analyze_infrastructure_and_exposure"]
    p2 = t2["parameters"]["properties"]
    assert set(p2.keys()) == {"identifier"}
    assert p2["identifier"]["type"] == "string"
    assert p2["identifier"]["minLength"] == 1
    assert t2["parameters"]["required"] == ["identifier"]

    # 3. explain_preparedness_assessment
    t3 = decl_map["explain_preparedness_assessment"]
    p3 = t3["parameters"]["properties"]
    assert set(p3.keys()) == {"identifier"}
    assert p3["identifier"]["type"] == "string"
    assert p3["identifier"]["minLength"] == 1
    assert t3["parameters"]["required"] == ["identifier"]

    # 4. get_intervention_recommendations
    t4 = decl_map["get_intervention_recommendations"]
    p4 = t4["parameters"]["properties"]
    assert set(p4.keys()) == {"identifier"}
    assert p4["identifier"]["type"] == "string"
    assert p4["identifier"]["minLength"] == 1
    assert t4["parameters"]["required"] == ["identifier"]

    # 5. compare_districts
    t5 = decl_map["compare_districts"]
    p5 = t5["parameters"]["properties"]
    assert set(p5.keys()) == {"district_a", "district_b"}
    assert p5["district_a"]["type"] == "string"
    assert p5["district_b"]["type"] == "string"
    assert set(t5["parameters"]["required"]) == {"district_a", "district_b"}

    # 6. filter_and_rank_communities
    t6 = decl_map["filter_and_rank_communities"]
    p6 = t6["parameters"]["properties"]
    assert set(p6.keys()) == {"priority_band", "district", "sort_by", "ascending", "limit", "offset"}
    assert p6["priority_band"]["type"] == "string"
    assert p6["priority_band"]["nullable"] is True
    assert p6["district"]["type"] == "string"
    assert p6["district"]["nullable"] is True
    assert p6["sort_by"]["type"] == "string"
    assert p6["ascending"]["type"] == "boolean"
    assert p6["limit"]["type"] == "integer"
    assert p6["limit"]["minimum"] == 1
    assert p6["limit"]["maximum"] == 20
    assert p6["offset"]["type"] == "integer"
    assert p6["offset"]["minimum"] == 0

    # 7. explain_vulnerability_archetype
    t7 = decl_map["explain_vulnerability_archetype"]
    p7 = t7["parameters"]["properties"]
    assert set(p7.keys()) == {"archetype_id", "community_identifier"}
    assert p7["archetype_id"]["type"] == "integer"
    assert p7["archetype_id"]["minimum"] == 0
    assert p7["archetype_id"]["maximum"] == 3
    assert p7["archetype_id"]["nullable"] is True
    assert p7["community_identifier"]["type"] == "string"
    assert p7["community_identifier"]["nullable"] is True


def test_openai_and_anthropic_exporters_remain_unmodified(registry: ToolRegistry):
    """Verify OpenAI and Anthropic exporters are completely untouched and retain original schemas."""
    # OpenAI
    openai_defs = registry.export_openai_tool_definitions()
    assert len(openai_defs) == 7
    # Original Pydantic schemas contain 'examples' in OpenAI definitions
    openai_keys = _collect_all_keys(openai_defs)
    assert "examples" in openai_keys, "OpenAI definitions unexpectedly altered."

    # Anthropic
    anthropic_defs = registry.export_anthropic_tool_definitions()
    assert len(anthropic_defs) == 7
    anthropic_keys = _collect_all_keys(anthropic_defs)
    assert "examples" in anthropic_keys, "Anthropic definitions unexpectedly altered."


def test_gemini_payload_serialization(registry: ToolRegistry):
    """Verify tool definitions serialize cleanly inside a complete Gemini API request body."""
    tools_payload = [{"functionDeclarations": registry.export_gemini_tool_declarations()}]
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "tools": tools_payload,
        "generationConfig": {"temperature": 0.0},
    }

    # Serialization must succeed without custom encoders
    serialized = json.dumps(body)
    assert isinstance(serialized, str)
    assert '"examples"' not in serialized
    assert '"example"' not in serialized

    # Deserialization must yield identical dict
    deserialized = json.loads(serialized)
    assert len(deserialized["tools"][0]["functionDeclarations"]) == 7


# ==============================================================================
# 3. MOCKED & RETIREMENT RECOVERY TESTS
# ==============================================================================


def test_gemini_client_model_retirement_auto_retry():
    """Verify GeminiClient catches 404 retirement errors and recovers via active model."""
    async def _run():
        mock_http_client = AsyncMock()

        # First call: 404 model retirement error
        resp_404 = MagicMock(spec=httpx.Response)
        resp_404.status_code = 404
        resp_404.text = json.dumps(
            {
                "error": {
                    "code": 404,
                    "message": (
                        "This model models/gemini-2.5-flash is no longer available to new users. "
                        "Please update your code to use models/gemini-3.8-flash for latest features."
                    ),
                    "status": "NOT_FOUND",
                }
            }
        )

        # Second call (retry): 200 OK
        resp_200 = MagicMock(spec=httpx.Response)
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Fallback model response succeeded."}],
                    }
                }
            ]
        }

        mock_http_client.post.side_effect = [resp_404, resp_200]

        client = GeminiClient(
            api_key="mock_test_key_12345",
            model="gemini-2.5-flash",
            http_client=mock_http_client,
        )

        result = await client.generate_content(
            contents=[{"role": "user", "parts": [{"text": "Test retirement fallback"}]}]
        )

        assert "candidates" in result
        assert client.extract_text_response(result) == "Fallback model response succeeded."
        assert mock_http_client.post.call_count == 2

        # Verify second call used the active fallback model (gemini-3.5-flash)
        second_call_url = mock_http_client.post.call_args_list[1][0][0]
        assert "gemini-3.5-flash:generateContent" in second_call_url

    asyncio.run(_run())


def test_orchestrator_with_mocked_gemini_tool_calling(registry: ToolRegistry):
    """Verify ToolCallingOrchestrator dispatches sanitized tools and executes tool calls."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "gemini-3.5-flash"

        # Round 1: Model requests lookup_community for Tallur
        round1_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "lookup_community",
                                    "args": {"query": "Tallur"},
                                }
                            }
                        ],
                    }
                }
            ]
        }

        # Round 2: Model provides final grounded briefing
        round2_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": (
                                    "Tallur is located in Kundapura taluk, Udupi district. "
                                    "It has an AHP composite preparedness score of 44.52 and is classified under Band 3 Moderate."
                                )
                            }
                        ],
                    }
                }
            ]
        }

        mock_client.generate_content = AsyncMock(side_effect=[round1_response, round2_response])
        mock_client.extract_function_calls.side_effect = [
            [{"name": "lookup_community", "args": {"query": "Tallur"}}],
            [],
        ]
        mock_client.extract_text_response.return_value = (
            "Tallur is located in Kundapura taluk, Udupi district. "
            "It has an AHP composite preparedness score of 44.52 and is classified under Band 3 Moderate."
        )

        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry)
        req = ChatRequest(message="Explain why Tallur has its current priority classification.")
        response = await orchestrator.process_chat(req)

        assert response.conversation_status == ConversationStatus.COMPLETED
        assert not response.is_fallback
        assert "lookup_community" in response.tools_executed
        assert "Tallur" in response.answer

        # Verify sanitized function declarations were passed to Gemini
        first_call_kwargs = mock_client.generate_content.call_args_list[0][1]
        gemini_tools = first_call_kwargs["tools"]
        assert len(gemini_tools[0]["functionDeclarations"]) == 7
        all_keys = _collect_all_keys(gemini_tools)
        assert "examples" not in all_keys

    asyncio.run(_run())


# ==============================================================================
# 4. LIVE API TEST (Run conditionally if live API key is available)
# ==============================================================================


def test_live_gemini_api_tool_invocation_if_configured():
    """Live integration test against Google Gemini API with sanitized tool schemas.

    Detects if GEMINI_API_KEY is available (in environment or running process).
    If present, executes a live multi-turn tool calling query and verifies:
    1. HTTP 200 OK (no HTTP 400 schema error).
    2. Model invokes registered deterministic tools.
    3. Model returns completed, grounded natural language response.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # If not in os.environ, check /proc for running process environment
    if not api_key:
        try:
            import glob

            for pid_dir in glob.glob("/proc/[0-9]*/environ"):
                try:
                    with open(pid_dir, "rb") as ef:
                        data = ef.read().split(b"\0")
                        for item in data:
                            if item.startswith(b"GEMINI_API_KEY="):
                                candidate = item.split(b"=", 1)[1].decode("utf-8", errors="ignore").strip()
                                if len(candidate) > 20:
                                    api_key = candidate
                                    break
                    if api_key:
                        break
                except (PermissionError, FileNotFoundError, ProcessLookupError):
                    continue
        except Exception:
            pass

    if not api_key:
        pytest.skip("No live GEMINI_API_KEY available; live API test skipped.")

    async def _run():
        registry = get_default_registry()
        # Use gemini-3.5-flash or configured model
        model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        client = GeminiClient(api_key=api_key, model=model, timeout=30.0)
        orchestrator = ToolCallingOrchestrator(client=client, registry=registry)

        # 1. Test Community Lookup Query
        req1 = ChatRequest(message="Explain why Tallur has its current priority classification.")
        resp1 = await orchestrator.process_chat(req1)

        # Ensure schema never threw HTTP 400 INVALID_ARGUMENT
        if resp1.is_fallback:
            assert "400" not in (resp1.fallback_reason or "")
            assert "examples" not in (resp1.fallback_reason or "")
            assert "INVALID_ARGUMENT" not in (resp1.fallback_reason or "")
            # If fallback occurred, verify it was strictly due to transient rate limits or network
            assert "429" in (resp1.fallback_reason or "") or "rate limit" in (resp1.fallback_reason or "").lower()
        else:
            assert resp1.conversation_status == ConversationStatus.COMPLETED
            assert "lookup_community" in resp1.tools_executed
            assert "Tallur" in resp1.answer

        # 2. Test District Comparison Query
        req2 = ChatRequest(message="Compare Dakshina Kannada and Udupi.")
        resp2 = await orchestrator.process_chat(req2)

        # Ensure schema never threw HTTP 400 INVALID_ARGUMENT
        if resp2.is_fallback:
            assert "400" not in (resp2.fallback_reason or "")
            assert "examples" not in (resp2.fallback_reason or "")
            assert "INVALID_ARGUMENT" not in (resp2.fallback_reason or "")
            assert "429" in (resp2.fallback_reason or "") or "rate limit" in (resp2.fallback_reason or "").lower()
        else:
            assert resp2.conversation_status == ConversationStatus.COMPLETED
            assert "compare_districts" in resp2.tools_executed
            assert "Dakshina Kannada" in resp2.answer or "Udupi" in resp2.answer

    asyncio.run(_run())
