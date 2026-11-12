#!/usr/bin/env python3
"""Generate complete, authoritative API reference and inventory from canonical FastAPI app.

Inspects `api.main:app` directly, extracting:
- OpenAPI specification, paths, methods, tags, parameters, schemas, responses
- Python route endpoints, handlers, docstrings
- Frontend usage analysis and internal/operational categorization
- Generates `docs/api_inventory.json` and `docs/API_REFERENCE.md`
"""

import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
INVENTORY_JSON_PATH = DOCS_DIR / "api_inventory.json"
API_REFERENCE_MD_PATH = DOCS_DIR / "API_REFERENCE.md"

FRONTEND_ROUTES = {
    ("GET", "/health"),
    ("GET", "/api/health"),
    ("GET", "/api/dashboard/overview"),
    ("GET", "/api/districts"),
    ("GET", "/api/districts/{district_id}"),
    ("GET", "/api/communities"),
    ("GET", "/api/communities/{community_id}"),
    ("GET", "/api/maps/communities"),
    ("GET", "/api/analytics/preparedness"),
    ("GET", "/api/analytics/priority-bands"),
    ("GET", "/api/analytics/clusters"),
    ("GET", "/api/analytics/model-performance"),
    ("GET", "/api/assistant/tools"),
    ("POST", "/api/assistant/chat"),
}

INTERNAL_OR_OPS_ROUTES = {
    ("GET", "/metrics"),
    ("GET", "/ready"),
    ("GET", "/api/health/ready"),
    ("POST", "/api/models/predict"),
    ("GET", "/api/models/registry"),
    ("GET", "/api/models/registry/{model_id}"),
    ("POST", "/api/v1/model/predict"),
    ("GET", "/api/v1/model/health"),
}

LOGICAL_GROUPS = [
    {
        "id": "health_and_readiness",
        "title": "Health and Readiness",
        "description": "System operational status, dataset readiness verification, and orchestrator probes.",
        "match": lambda path, tag: path in ("/health", "/ready", "/api/health", "/api/health/ready", "/metrics"),
    },
    {
        "id": "dashboard",
        "title": "Dashboard",
        "description": "Regional command-center executive metrics, KPI summaries, and district overviews.",
        "match": lambda path, tag: path.startswith("/api/dashboard"),
    },
    {
        "id": "communities",
        "title": "Communities",
        "description": "Granular administrative unit records, full-text search, multi-criteria filtering, and detailed dossiers.",
        "match": lambda path, tag: path == "/api/communities" or path == "/api/communities/{community_id}",
    },
    {
        "id": "geospatial_and_maps",
        "title": "Geospatial and Maps",
        "description": "Spatial points, coordinate bounding boxes, and GIS filtering for interactive map visualization.",
        "match": lambda path, tag: path.startswith("/api/maps"),
    },
    {
        "id": "districts",
        "title": "Districts",
        "description": "Comparative district-level profiles, demographic totals, and cross-district preparedness differentials.",
        "match": lambda path, tag: path == "/api/districts" or path == "/api/districts/{district_id}",
    },
    {
        "id": "analytics",
        "title": "Analytics",
        "description": "Authoritative AHP distributions, surrogate model performance diagnostics, and operational distributions.",
        "match": lambda path, tag: path.startswith("/api/analytics"),
    },
    {
        "id": "priority_bands",
        "title": "Priority Bands",
        "description": "Four-tier operational resource prioritization (Band 1 Critical to Band 4 Adequate) distributions.",
        "match": lambda path, tag: path == "/api/analytics/priority-bands",
    },
    {
        "id": "vulnerability_clusters",
        "title": "Vulnerability Clusters",
        "description": "K-Means (k=4) unsupervised vulnerability archetypes and Uttara Kannada fallback documentation.",
        "match": lambda path, tag: path == "/api/analytics/clusters" or path == "/api/v1/archetypes",
    },
    {
        "id": "infrastructure_and_interventions",
        "title": "Infrastructure and Interventions",
        "description": "Domain-specific infrastructure deficit analysis and targeted physical intervention recommendations.",
        "match": lambda path, tag: "simulation" in path or "infrastructure" in path or "interventions" in path,
    },
    {
        "id": "model_inference",
        "title": "Model Inference",
        "description": "Empirical surrogate Logistic Regression classifier inference and probability output generation.",
        "match": lambda path, tag: path == "/api/models/predict" or path == "/api/v1/model/predict",
    },
    {
        "id": "model_registry",
        "title": "Model Registry",
        "description": "Local Git-compatible Model Registry catalog inspection, artifact provenance, and version metadata.",
        "match": lambda path, tag: path.startswith("/api/models/registry"),
    },
    {
        "id": "assistant",
        "title": "Assistant",
        "description": "Conversational spatial decision support with deterministic AHP tool grounding and refusal gates.",
        "match": lambda path, tag: path.startswith("/api/assistant") and not path.startswith("/api/v1/assistant"),
    },
    {
        "id": "legacy_v1",
        "title": "Legacy Compatibility Routes",
        "description": "Backward-compatible REST endpoints preserving contract compatibility with v1 clients.",
        "match": lambda path, tag: path.startswith("/api/v1/"),
    },
]


def resolve_ref(schema: dict, ref: str) -> dict:
    """Resolve a $ref pointer within OpenAPI components."""
    parts = ref.lstrip("#/").split("/")
    cur = schema
    for p in parts:
        cur = cur.get(p, {})
    return cur


def generate_inventory():
    openapi = app.openapi()
    paths = openapi.get("paths", {})
    components = openapi.get("components", {}).get("schemas", {})

    endpoints = []

    # Include special endpoints not in paths or schema
    special_routes = [
        {
            "method": "GET",
            "path": "/",
            "tag": "Navigation",
            "purpose": "Redirect root traffic to interactive Swagger UI documentation.",
            "path_parameters": [],
            "query_parameters": [],
            "request_body_schema": None,
            "response_schema": "307 Temporary Redirect -> /docs",
            "status_codes": [307],
            "authentication_required": False,
            "example_request": "curl -i http://localhost:8000/",
            "example_response": 'HTTP/1.1 307 Temporary Redirect\nLocation: /docs',
            "used_by_frontend": False,
            "internal_only": False,
            "group": "health_and_readiness",
        },
        {
            "method": "GET",
            "path": "/ready",
            "tag": "Health & Status",
            "purpose": "Root readiness probe for container orchestration confirming dataset and surrogate model are loaded.",
            "path_parameters": [],
            "query_parameters": [],
            "request_body_schema": None,
            "response_schema": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean"},
                    "dataset_loaded": {"type": "boolean"},
                    "model_loaded": {"type": "boolean"},
                    "version": {"type": "string"}
                }
            },
            "status_codes": [200],
            "authentication_required": False,
            "example_request": "curl http://localhost:8000/ready",
            "example_response": '{"ready": true, "dataset_loaded": true, "model_loaded": true, "version": "2.0.0"}',
            "used_by_frontend": False,
            "internal_only": True,
            "group": "health_and_readiness",
        },
        {
            "method": "GET",
            "path": "/metrics",
            "tag": "Observability",
            "purpose": "Prometheus operational metrics exposition endpoint with bounded cardinality.",
            "path_parameters": [],
            "query_parameters": [],
            "request_body_schema": None,
            "response_schema": "text/plain (Prometheus exposition format)",
            "status_codes": [200],
            "authentication_required": False,
            "example_request": "curl http://localhost:8000/metrics",
            "example_response": '# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{endpoint="/health",method="GET",status="200"} 12',
            "used_by_frontend": False,
            "internal_only": True,
            "group": "health_and_readiness",
        },
        {
            "method": "GET",
            "path": "/docs",
            "tag": "Documentation",
            "purpose": "Interactive Swagger UI for endpoint exploration and testing.",
            "path_parameters": [],
            "query_parameters": [],
            "request_body_schema": None,
            "response_schema": "text/html",
            "status_codes": [200],
            "authentication_required": False,
            "example_request": "curl http://localhost:8000/docs",
            "example_response": "<!DOCTYPE html><html><head><title>Swagger UI</title>...",
            "used_by_frontend": False,
            "internal_only": False,
            "group": "health_and_readiness",
        },
        {
            "method": "GET",
            "path": "/redoc",
            "tag": "Documentation",
            "purpose": "ReDoc responsive documentation browser.",
            "path_parameters": [],
            "query_parameters": [],
            "request_body_schema": None,
            "response_schema": "text/html",
            "status_codes": [200],
            "authentication_required": False,
            "example_request": "curl http://localhost:8000/redoc",
            "example_response": "<!DOCTYPE html><html><head><title>ReDoc</title>...",
            "used_by_frontend": False,
            "internal_only": False,
            "group": "health_and_readiness",
        },
        {
            "method": "GET",
            "path": "/openapi.json",
            "tag": "Documentation",
            "purpose": "Machine-readable OpenAPI 3.1 specification for tooling and client generation.",
            "path_parameters": [],
            "query_parameters": [],
            "request_body_schema": None,
            "response_schema": "application/json (OpenAPI 3.1.0)",
            "status_codes": [200],
            "authentication_required": False,
            "example_request": "curl http://localhost:8000/openapi.json",
            "example_response": '{"openapi": "3.1.0", "info": {"title": "Cyclone Preparedness Decision Intelligence API"}}',
            "used_by_frontend": False,
            "internal_only": False,
            "group": "health_and_readiness",
        },
    ]

    for item in special_routes:
        endpoints.append(item)

    for path, methods in sorted(paths.items()):
        for method, details in sorted(methods.items()):
            method_upper = method.upper()
            tags = details.get("tags", [])
            tag = tags[0] if tags else "General"
            summary = details.get("summary", "")
            description = details.get("description", "")
            purpose = summary or (description.split("\n")[0] if description else "API endpoint")

            # Parameters
            raw_params = details.get("parameters", [])
            path_params = []
            query_params = []
            for p in raw_params:
                p_in = p.get("in")
                p_info = {
                    "name": p.get("name"),
                    "required": p.get("required", False),
                    "schema": p.get("schema", {}),
                    "description": p.get("description", ""),
                }
                if p_in == "path":
                    path_params.append(p_info)
                elif p_in == "query":
                    query_params.append(p_info)

            # Request Body
            req_body_obj = details.get("requestBody", {})
            req_body_schema = None
            if req_body_obj:
                content = req_body_obj.get("content", {})
                json_content = content.get("application/json", {})
                req_body_schema = json_content.get("schema", {})
                if "$ref" in req_body_schema:
                    req_body_schema = resolve_ref(openapi, req_body_schema["$ref"])

            # Responses
            responses_obj = details.get("responses", {})
            status_codes = [int(sc) for sc in responses_obj.keys() if sc.isdigit()]
            resp_200 = responses_obj.get("200", responses_obj.get("201", {}))
            resp_content = resp_200.get("content", {}).get("application/json", {})
            resp_schema = resp_content.get("schema", {})
            if "$ref" in resp_schema:
                resp_schema = resolve_ref(openapi, resp_schema["$ref"])

            # Group matching
            matched_group = "legacy_v1" if path.startswith("/api/v1") else "communities"
            for g in LOGICAL_GROUPS:
                if g["match"](path, tag):
                    matched_group = g["id"]
                    break

            is_frontend = (method_upper, path) in FRONTEND_ROUTES
            is_internal = (method_upper, path) in INTERNAL_OR_OPS_ROUTES

            # Generate realistic example request and response
            example_req, example_resp = generate_examples(method_upper, path, path_params, query_params, req_body_schema)

            endpoints.append({
                "method": method_upper,
                "path": path,
                "tag": tag,
                "purpose": purpose,
                "path_parameters": path_params,
                "query_parameters": query_params,
                "request_body_schema": req_body_schema,
                "response_schema": resp_schema or "object",
                "status_codes": sorted(status_codes) or [200],
                "authentication_required": False,
                "example_request": example_req,
                "example_response": example_resp,
                "used_by_frontend": is_frontend,
                "internal_only": is_internal,
                "group": matched_group,
            })

    # Sort endpoints deterministically
    endpoints = sorted(endpoints, key=lambda e: (e["path"], e["method"]))

    inventory_data = {
        "metadata": {
            "title": openapi.get("info", {}).get("title"),
            "version": openapi.get("info", {}).get("version"),
            "description": openapi.get("info", {}).get("description"),
            "total_endpoints": len(endpoints),
            "generated_at_utc": "2026-11-11T12:00:00Z",
            "logical_groups": [g["id"] for g in LOGICAL_GROUPS],
        },
        "endpoints": endpoints,
    }

    with open(INVENTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(inventory_data, f, indent=2)
    print(f"Saved API inventory ({len(endpoints)} endpoints) to {INVENTORY_JSON_PATH}")

    # Generate Markdown Reference
    generate_markdown_reference(inventory_data)
    print(f"Saved API Markdown Reference to {API_REFERENCE_MD_PATH}")
    return inventory_data


def generate_examples(method: str, path: str, path_params: list, query_params: list, req_schema: dict):
    """Generate representative cURL commands and JSON responses."""
    if path == "/health" or path == "/api/health":
        req = f"curl -s http://localhost:8000{path}"
        resp = json.dumps({
            "status": "healthy",
            "dataset_loaded": True,
            "total_units": 1879,
            "total_villages": 1845,
            "total_census_towns": 34,
            "districts": ["Dakshina Kannada", "Udupi", "Uttara Kannada"],
            "archetypes_count": 4,
            "version": "2.0.0",
            "model_loaded": True,
            "model_artifact_name": "village-preparedness-classifier",
            "model_artifact_version": "v1.0.0",
            "model_feature_schema_status": "VALID_26_FEATURES",
            "model_provenance": "REPRODUCIBLE RECONSTRUCTION (70/15/15 experiment)"
        }, indent=2)
        return req, resp

    if path == "/api/dashboard/overview":
        req = "curl -s http://localhost:8000/api/dashboard/overview"
        resp = json.dumps({
            "total_units": 1879,
            "districts_count": 3,
            "assigned_clusters_count": 622,
            "unassigned_clusters_count": 1257,
            "priority_band_counts": {
                "BAND_1_CRITICAL": 155,
                "BAND_2_SPECIALIZED": 829,
                "BAND_3_MODERATE": 299,
                "BAND_4_ADEQUATE": 596
            },
            "preparedness_tier_counts": {"High": 596, "Medium": 1128, "Low": 155},
            "infrastructure_highlights": {
                "total_population": 4680000,
                "pucca_road_pct": 78.5,
                "coastal_units_under_5km": 249
            }
        }, indent=2)
        return req, resp

    if path == "/api/communities":
        req = "curl -s 'http://localhost:8000/api/communities?district=Udupi&priority_band=BAND_4_ADEQUATE&limit=2'"
        resp = json.dumps({
            "items": [
                {
                    "area_id": "608691",
                    "village_name": "Tallur",
                    "district_name": "Udupi",
                    "preparedness_score": 74.34,
                    "preparedness_tier": "High",
                    "priority_band": "BAND_4_ADEQUATE",
                    "vulnerability_archetype_id": 3
                }
            ],
            "total": 123,
            "page": 1,
            "limit": 2,
            "pages": 62
        }, indent=2)
        return req, resp

    if path == "/api/communities/{community_id}":
        req = "curl -s http://localhost:8000/api/communities/608691"
        resp = json.dumps({
            "area_id": "608691",
            "village_name": "Tallur",
            "district_name": "Udupi",
            "pc11_id": "608691",
            "preparedness_score": 74.34,
            "preparedness_tier": "High",
            "priority_band": "BAND_4_ADEQUATE",
            "vulnerability_archetype_id": 3,
            "domain_scores": {
                "medical": 82.5,
                "shelters": 45.0,
                "infrastructure": 85.0,
                "early_warning": 90.0,
                "socioeconomic": 88.0
            }
        }, indent=2)
        return req, resp

    if path == "/api/assistant/chat":
        req = """curl -X POST http://localhost:8000/api/assistant/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Explain why Tallur is classified as Band 4."}'"""
        resp = json.dumps({
            "answer": "Tallur (PC11: 608691, Udupi) achieves an authoritative AHP composite preparedness score of 74.34, categorizing it in the High preparedness tier and Priority Band 4 (Adequate).",
            "tools_executed": ["explain_preparedness_assessment"],
            "tool_records": [
                {
                    "tool_name": "explain_preparedness_assessment",
                    "success": True,
                    "execution_time_ms": 1.8
                }
            ],
            "is_fallback": False,
            "scientific_precedence_note": "AHP composite preparedness score maintains authoritative operational precedence."
        }, indent=2)
        return req, resp

    if path == "/api/models/predict":
        req = """curl -X POST http://localhost:8000/api/models/predict \\
  -H "Content-Type: application/json" \\
  -d '{"area_id": "608691"}'"""
        resp = json.dumps({
            "model_identifier": "village-preparedness-classifier",
            "artifact_version": "v1.0.0",
            "predicted_preparedness_tier": "High",
            "surrogate_probabilities": {
                "High": 0.892,
                "Medium": 0.104,
                "Low": 0.004
            },
            "methodology_note": "Surrogate model outputs represent empirical approximations of AHP preparedness tiers, NOT physical hazard probabilities."
        }, indent=2)
        return req, resp

    if method == "POST":
        req = f"""curl -X POST http://localhost:8000{path} \\
  -H "Content-Type: application/json" \\
  -d '{{}}'"""
        resp = json.dumps({"status": "success", "path": path}, indent=2)
    else:
        req = f"curl -s http://localhost:8000{path}"
        resp = json.dumps({"status": "success", "endpoint": path}, indent=2)

    return req, resp


def generate_markdown_reference(data: dict):
    endpoints = data["endpoints"]
    total = len(endpoints)
    lines = []

    lines.append("# CYCLONE INTEL: Canonical FastAPI API Reference")
    lines.append("")
    lines.append("> **Platform Version:** 2.0.0  ")
    lines.append("> **Specification:** OpenAPI 3.1.0  ")
    lines.append(f"> **Total Registered Endpoints:** {total}  ")
    lines.append("> **Base URL:** `/api` (Production canonical application `api.main:app`)  ")
    lines.append("")
    lines.append("## Overview and Architectural Principles")
    lines.append("")
    lines.append("1. **Authoritative AHP Precedence**: Deterministic Analytic Hierarchy Process (AHP) composite indices represent the single authoritative ground truth for coastal preparedness across all 1,879 administrative units. Machine Learning surrogates serve purely empirical diagnostic purposes and never supersede AHP.")
    lines.append("2. **Non-Downgrading Invariant**: The surrogate classifier is structurally barred from downgrading an authoritative AHP Low preparedness classification.")
    lines.append("3. **Descriptive Archetypes**: K-Means clustering (k=4) identifies descriptive structural vulnerability archetypes across Dakshina Kannada and Udupi. The 1,257 administrative units in Uttara Kannada are explicitly tracked under unassigned fallback heuristics.")
    lines.append("4. **Documentation Drift Detection**: This catalogue is verified programmatically by `tests/test_route_inventory.py` to ensure complete alignment with live registered FastAPI routers.")
    lines.append("")

    lines.append("## Interactive Documentation Interfaces")
    lines.append("")
    lines.append("- **Swagger UI (Local & Production)**: [`/docs`](http://localhost:8000/docs)")
    lines.append("- **ReDoc Interface**: [`/redoc`](http://localhost:8000/redoc)")
    lines.append("- **OpenAPI JSON Schema**: [`/openapi.json`](http://localhost:8000/openapi.json)")
    lines.append("")

    lines.append("## Logical Endpoint Groups")
    lines.append("")
    for g in LOGICAL_GROUPS:
        grp_endpoints = [e for e in endpoints if e["group"] == g["id"]]
        lines.append(f"- [{g['title']}](#{g['id'].replace('_', '-')}) ({len(grp_endpoints)} endpoints) — {g['description']}")
    lines.append("")

    # Render each group
    for g in LOGICAL_GROUPS:
        grp_endpoints = [e for e in endpoints if e["group"] == g["id"]]
        if not grp_endpoints:
            continue

        lines.append(f"---")
        lines.append(f"## {g['title']}")
        lines.append(f"*{g['description']}*")
        lines.append("")

        for ep in grp_endpoints:
            method = ep["method"]
            path = ep["path"]
            lines.append(f"### `{method} {path}`")
            lines.append("")
            lines.append(f"**Purpose**: {ep['purpose']}  ")
            lines.append(f"**API Tag**: `{ep['tag']}` | **Authentication**: `None` | **Status Codes**: `{', '.join(str(s) for s in ep['status_codes'])}`  ")
            lines.append(f"**Used by Frontend**: {'✅ Yes' if ep['used_by_frontend'] else '❌ No'} | **Internal / Ops Only**: {'🔒 Yes' if ep['internal_only'] else '🌐 Public'}  ")
            lines.append("")

            # Parameters
            if ep["path_parameters"]:
                lines.append("**Path Parameters**:")
                lines.append("| Name | Required | Type | Description |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for p in ep["path_parameters"]:
                    p_type = p.get("schema", {}).get("type", "string")
                    lines.append(f"| `{p['name']}` | `{'Yes' if p['required'] else 'No'}` | `{p_type}` | {p['description'] or '—'} |")
                lines.append("")

            if ep["query_parameters"]:
                lines.append("**Query Parameters**:")
                lines.append("| Name | Required | Type | Description |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for p in ep["query_parameters"]:
                    p_type = p.get("schema", {}).get("type", "string")
                    lines.append(f"| `{p['name']}` | `{'Yes' if p['required'] else 'No'}` | `{p_type}` | {p['description'] or '—'} |")
                lines.append("")

            # Request Body
            if ep["request_body_schema"]:
                lines.append("**Request Body Schema**:")
                lines.append("```json")
                if isinstance(ep["request_body_schema"], dict):
                    lines.append(json.dumps(ep["request_body_schema"].get("properties", ep["request_body_schema"]), indent=2)[:500])
                else:
                    lines.append(str(ep["request_body_schema"]))
                lines.append("```")
                lines.append("")

            # Example Request
            lines.append("**Example Request**:")
            lines.append("```bash")
            lines.append(ep["example_request"])
            lines.append("```")
            lines.append("")

            # Example Response
            lines.append("**Example Response**:")
            lines.append("```json")
            lines.append(ep["example_response"])
            lines.append("```")
            lines.append("")

    with open(API_REFERENCE_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    generate_inventory()
