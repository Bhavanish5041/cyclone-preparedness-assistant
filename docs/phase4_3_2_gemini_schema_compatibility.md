# Phase 4.3.2: Gemini Function Declaration Schema Compatibility Fix

## 1. Executive Summary

During testing of the conversational Cyclone Preparedness Assistant with live Google Gemini API keys, API requests to `models/{model}:generateContent` failed with HTTP 400 `INVALID_ARGUMENT`:

```text
Invalid JSON payload received. Unknown name "examples" at 'tools[0].function_declarations[...].parameters.properties[...].value': Cannot find field.
```

Investigation traced the error to `src/assistant/tools/registry.py`, where `ToolRegistry.export_gemini_tool_declarations()` directly forwarded raw Pydantic v2 JSON Schemas (`tool.get_json_schema()`) into Gemini's `functionDeclarations`. Because the Google Gemini v1beta Generative Language REST API compiles function declaration parameters into a strict subset of OpenAPI 3.0 governed by the `google.ai.generativelanguage.v1beta.Schema` Protocol Buffer definition, fields such as `examples`, `example`, `$schema`, `$defs`, `definitions`, and `additionalProperties` are rejected at the protobuf deserialization layer prior to inference.

In Phase 4.3.2, we implemented a recursive schema normalization engine (`sanitize_gemini_schema`) tailored to Gemini's protobuf specification while strictly preserving all parameter names, types, descriptions, constraints, required fields, and enums. We also ensured that the OpenAI and Anthropic schema exporters remained unmodified.

---

## 2. Root Cause Analysis

### 2.1 Schema Layer Mismatch

1. **Pydantic v2 Behavior:**
   All 7 deterministic tools (`lookup_community`, `compare_districts`, `explain_preparedness_assessment`, `filter_and_rank_communities`, `analyze_infrastructure_and_exposure`, `explain_vulnerability_archetype`, `get_intervention_recommendations`) define input parameters using Pydantic `Field(..., examples=[...])` and `Optional[...] = None`. When `model_json_schema()` is generated:
   - Each property contains `"examples": [...]` (11 instances across the 7 tools).
   - Optional parameters generate `"anyOf": [{"type": "..."}, {"type": "null"}]`.
   - Root models include metadata such as `"$schema"`, `"$defs"`, and `"additionalProperties"`.

2. **Gemini Protocol Buffer Specification:**
   In `google.ai.generativelanguage.v1beta.Schema`:
   ```proto
   message Schema {
     Type type = 1;
     string format = 2;
     string description = 3;
     bool nullable = 4;
     repeated string enum = 5;
     double maximum = 6;
     double minimum = 7;
     int64 max_items = 8;
     int64 min_items = 9;
     map<string, Schema> properties = 10;
     Schema items = 11;
     repeated string required = 12;
     int64 max_length = 14;
     int64 min_length = 15;
     string pattern = 16;
     repeated string property_ordering = 18;
     string default = 23;
     string title = 24;
   }
   ```
   The Protobuf `Schema` message has **no** `examples`, `example`, `$schema`, `$defs`, `definitions`, or `additionalProperties` fields. The Google API gateway rejects JSON containing unrecognized keys with HTTP 400 `INVALID_ARGUMENT`.

---

## 3. Implementation Details

### 3.1 Recursive Normalizer (`sanitize_gemini_schema`)

Located in [`src/assistant/tools/registry.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/tools/registry.py):

```python
UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "examples",
    "example",
    "$schema",
    "$defs",
    "definitions",
    "additionalProperties",
}


def sanitize_gemini_schema(schema: Any) -> Any:
    """Recursively normalize a JSON Schema dictionary for Google Gemini Function Calling."""
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [sanitize_gemini_schema(item) for item in schema]
        return schema

    schema_copy = deepcopy(schema)

    # 1. Handle type as list (e.g. ['string', 'null'])
    raw_type = schema_copy.get("type")
    if isinstance(raw_type, list):
        has_null = "null" in raw_type
        non_null_types = [t for t in raw_type if t != "null"]
        if non_null_types:
            schema_copy["type"] = non_null_types[0]
        if has_null:
            schema_copy["nullable"] = True

    # 2. Normalize anyOf / oneOf unions into a single schema with nullable flag
    for union_key in ("anyOf", "oneOf"):
        if union_key in schema_copy and isinstance(schema_copy[union_key], list):
            variants = schema_copy[union_key]
            has_null = any(isinstance(v, dict) and v.get("type") == "null" for v in variants)
            non_null = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
            if non_null:
                primary = deepcopy(non_null[0])
                for pk, pv in primary.items():
                    if pk not in schema_copy:
                        schema_copy[pk] = pv
                if has_null:
                    schema_copy["nullable"] = True
            del schema_copy[union_key]

    # 3. Recursively clean remaining fields
    cleaned: Dict[str, Any] = {}
    for k, v in schema_copy.items():
        if k in UNSUPPORTED_GEMINI_SCHEMA_KEYS:
            continue
        cleaned[k] = sanitize_gemini_schema(v)

    # 4. Clean required array if present
    if "required" in cleaned and isinstance(cleaned["required"], list):
        cleaned["required"] = [str(r) for r in cleaned["required"]]

    return cleaned
```

### 3.2 Tool Declaration Export

In `ToolRegistry.export_gemini_tool_declarations`:

```python
    def export_gemini_tool_declarations(self) -> List[Dict[str, Any]]:
        declarations = []
        for name in self.list_tools():
            tool = self._tools[name]
            raw_schema = deepcopy(tool.get_json_schema())
            sanitized_parameters = sanitize_gemini_schema(raw_schema)
            declarations.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": sanitized_parameters,
                }
            )
        return declarations
```

### 3.3 Provider Isolation

OpenAI and Anthropic tool exporters were preserved without modification:
- `export_openai_tool_definitions()` continues to output full OpenAPI 3.0 function specifications including `examples`.
- `export_anthropic_tool_definitions()` continues to output standard Anthropic tool use declarations.

### 3.4 Model Retirement Auto-Recovery

In [`src/assistant/llm/client.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/llm/client.py), Google AI Studio has retired `gemini-2.5-flash` for new API accounts, returning:
`"This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash for the latest features..."`

We enhanced `GeminiClient.generate_content` with automatic retirement recovery:
- Intercepts HTTP 404 containing `"no longer available to new users"`.
- Logs an operational warning without credential leakage.
- Seamlessly dispatches the request to `gemini-3.5-flash` (or recommended active flash model).
- Increased default timeout from 15.0s to 30.0s for complex multi-turn tool invocation reasoning.

---

## 4. Verification & Testing

### 4.1 Test Suite (`tests/test_gemini_schema_compatibility.py`)

A comprehensive test suite of 10 automated tests was added:

1. `test_sanitize_gemini_schema_strips_unsupported_metadata`: Unit test verifying recursive elimination of `examples`, `example`, `$schema`, `$defs`, `definitions`, and `additionalProperties`.
2. `test_sanitize_gemini_schema_handles_unions_and_nullability`: Unit test verifying flattening of `anyOf`/`oneOf` unions to `nullable=True`.
3. `test_exported_gemini_declarations_count_and_format`: Integration test verifying all 7 tools export valid function declarations.
4. `test_exported_gemini_declarations_have_zero_examples_recursively`: Recursive tree traversal verifying 0 instances of `examples` or unsupported metadata across all 7 tool schemas.
5. `test_exported_gemini_declarations_preserve_contracts_and_constraints`: Asserts parameter names, types, descriptions, min/max limits, and required arrays for all 7 deterministic tools:
   - `lookup_community`: `query` (str, minLength 1, required), `district` (nullable str), `limit` (int, 1-15, default 5)
   - `compare_districts`: `district_a`, `district_b` (both required str)
   - `explain_preparedness_assessment`: `identifier` (str, minLength 1, required)
   - `get_intervention_recommendations`: `identifier` (str, minLength 1, required)
   - `filter_and_rank_communities`: `priority_band` (nullable str), `district` (nullable str), `sort_by` (str), `ascending` (bool), `limit` (int, 1-20), `offset` (int >= 0)
   - `explain_vulnerability_archetype`: `archetype_id` (nullable int, 0-3), `community_identifier` (nullable str)
   - `analyze_infrastructure_and_exposure`: `identifier` (str, minLength 1, required)
6. `test_openai_and_anthropic_exporters_remain_unmodified`: Asserts non-regression on OpenAI and Anthropic formats.
7. `test_gemini_payload_serialization`: Validates standard `json.dumps()` round-trip serialization.
8. `test_gemini_client_model_retirement_auto_retry`: Mocks HTTP 404 retirement response and verifies recovery.
9. `test_orchestrator_with_mocked_gemini_tool_calling`: Verifies multi-turn tool calling turn cycle with sanitized tool schemas.
10. `test_live_gemini_api_tool_invocation_if_configured`: Live API integration test verifying zero HTTP 400 errors and successful tool-assisted response generation.

### 4.2 Full Regression Suite

```text
======================== 364 passed, 13 warnings in 23.16s =========================
```

All 364 platform tests pass with 100% success rate.
