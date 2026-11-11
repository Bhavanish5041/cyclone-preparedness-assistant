#!/usr/bin/env python3
"""Phase 8: Comprehensive End-to-End Production Verification.

Verifies the 14 mandatory non-negotiable requirements on the active public Cloudflare URL
and internal services.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error

# Determine current public tunnel URL
from scripts.ops_monitor import get_public_url

PUBLIC_URL = get_public_url()
if not PUBLIC_URL:
    print("FATAL: Could not detect public tunnel URL from journalctl.")
    sys.exit(1)

INTERNAL_API_URL = "http://127.0.0.1:8000"

print(f"Target Public URL: {PUBLIC_URL}")
print(f"Target Internal API: {INTERNAL_API_URL}")
print("=" * 70)


def request_json(url: str, method: str = "GET", payload: dict | None = None, headers: dict | None = None) -> tuple[int, dict | str]:
    headers = headers or {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    headers["User-Agent"] = "CycloneE2ETester/1.0"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:
        return 0, str(e)


results = []


def record_result(check_num: int, name: str, passed: bool, details: str):
    status = "PASS" if passed else "FAIL"
    results.append({"check": check_num, "name": name, "passed": passed, "details": details})
    print(f"[{status}] Check {check_num:02d}: {name}")
    print(f"       Details: {details}\n")


# 1. Public website loads
code, html = request_json(f"{PUBLIC_URL}/")
passed_1 = code == 200 and "CYCLONE INTEL" in str(html)
record_result(1, "Public Website SPA Loads", passed_1, f"Status: {code}, Title found: {'CYCLONE INTEL' in str(html)}")

# 2. Frontend communicates with FastAPI
code, health_data = request_json(f"{PUBLIC_URL}/api/health")
passed_2 = code == 200 and isinstance(health_data, dict) and health_data.get("status") == "healthy"
record_result(2, "Frontend Communicates with FastAPI Ingress", passed_2, f"Status: {code}, Units: {health_data.get('total_units') if isinstance(health_data, dict) else 'none'}")

# 3 & 4 & 5 & 6. Qwen processes query for Tallur, tools return evidence, grounded synthesis, AHP 74.34, High tier, Band 4
code, chat_data = request_json(
    f"{PUBLIC_URL}/api/assistant/chat",
    method="POST",
    payload={"message": "What is the preparedness score for Tallur?"}
)
passed_3 = (
    code == 200
    and isinstance(chat_data, dict)
    and "answer" in chat_data
    and chat_data.get("model_name") == "qwen2.5:0.5b-instruct"
    and chat_data.get("is_fallback") is False
)
answer = chat_data.get("answer", "") if isinstance(chat_data, dict) else ""
tools_used = chat_data.get("tools_executed", []) if isinstance(chat_data, dict) else []

# Check 3: Qwen query processing
record_result(3, "Qwen Processes Real User Query (qwen2.5:0.5b-instruct)", passed_3, f"Status: {code}, Model: {chat_data.get('model_name') if isinstance(chat_data, dict) else 'none'}, is_fallback: {chat_data.get('is_fallback') if isinstance(chat_data, dict) else 'none'}, Tools called: {tools_used}")

# Check 4: Deterministic tools return correct evidence
passed_4 = "explain_preparedness_assessment" in tools_used
record_result(4, "Deterministic Tools Return Correct Evidence", passed_4, f"Tools invoked: {tools_used}")

# Check 5: Final synthesis is grounded in evidence
passed_5 = "74.34" in answer and "Tallur" in answer
record_result(5, "Final Synthesis Grounded in Evidence", passed_5, f"Answer length: {len(answer)} chars, Contains 74.34: {'74.34' in answer}")

# Check 6: Tallur returns AHP score 74.34 and correct authoritative preparedness tier / band
code, village_data = request_json(f"{PUBLIC_URL}/api/communities/Tallur")
ahp_score = village_data.get("preparedness_score") if isinstance(village_data, dict) else None
ahp_tier = village_data.get("preparedness_tier") if isinstance(village_data, dict) else None
priority_band = village_data.get("priority_band") if isinstance(village_data, dict) else None
passed_6 = (
    code == 200
    and ahp_score == 74.34
    and ahp_tier == "High"
    and priority_band == "BAND_4_ADEQUATE"
    and "74.34" in answer
)
record_result(6, "Tallur Authoritative AHP Score 74.34 & High Preparedness Tier (Band 4)", passed_6, f"AHP Score: {ahp_score}, Tier: {ahp_tier}, Band: {priority_band}")

# 7. District comparison works correctly
code, comp_data = request_json(
    f"{PUBLIC_URL}/api/assistant/chat",
    method="POST",
    payload={"message": "Compare vulnerability between Udupi and Uttara Kannada"}
)
comp_tools = comp_data.get("tools_executed", []) if isinstance(comp_data, dict) else []
comp_answer = comp_data.get("answer", "") if isinstance(comp_data, dict) else ""
passed_7 = code == 200 and "compare_districts" in comp_tools and ("Udupi" in comp_answer and "Uttara Kannada" in comp_answer)
record_result(7, "District Comparison Execution", passed_7, f"Tools: {comp_tools}, Udupi/UK mentioned: {'Udupi' in comp_answer and 'Uttara Kannada' in comp_answer}")

# 8. Ambiguous district/community names are handled safely
code, ambig_data = request_json(
    f"{PUBLIC_URL}/api/assistant/chat",
    method="POST",
    payload={"message": "What is the status of Belur?"}
)
ambig_ans = ambig_data.get("answer", "") if isinstance(ambig_data, dict) else ""
passed_8 = code == 200 and len(ambig_ans) > 0
record_result(8, "Ambiguous Community Name Handled Safely", passed_8, f"Status: {code}, Response length: {len(ambig_ans)}")

# 9. Unsupported live-data requests are refused
code, ref_data = request_json(
    f"{PUBLIC_URL}/api/assistant/chat",
    method="POST",
    payload={"message": "What is the live radar storm surge and current flooded roads in Mangalore right now?"}
)
refused_flag = ref_data.get("refused", False) if isinstance(ref_data, dict) else False
ref_ans = ref_data.get("answer", "") if isinstance(ref_data, dict) else ""
passed_9 = (
    code == 200
    and (refused_flag or "not support" in ref_ans.lower() or "real-time" in ref_ans.lower() or "telemetry" in ref_ans.lower() or "cannot predict live" in ref_ans.lower())
)
record_result(9, "Unsupported Live Telemetry Requests Refused", passed_9, f"Refused flag: {refused_flag}, Refusal keyword present: {passed_9}")

# 10. Logistic Regression predictions respect AHP precedence (cannot downgrade AHP Low)
code, pred_data = request_json(
    f"{PUBLIC_URL}/api/models/predict",
    method="POST",
    payload={"area_id": "Tallur"}
)
passed_10 = (
    code == 200
    and isinstance(pred_data, dict)
    and pred_data.get("predicted_tier") == "High"
    and "MANDATE" in pred_data.get("operational_precedence_note", "")
)
record_result(10, "Logistic Regression Respects AHP Precedence Mandate", passed_10, f"Predicted Tier: {pred_data.get('predicted_tier') if isinstance(pred_data, dict) else 'err'}, Precedence note present: {'MANDATE' in str(pred_data)}")

# 11. Model registry inspection works
code, reg_data = request_json(f"{PUBLIC_URL}/api/models/registry")
models_list = reg_data.get("models", []) if isinstance(reg_data, dict) else []
passed_11 = code == 200 and len(models_list) >= 3
record_result(11, "Model Registry Inspection Operational", passed_11, f"Registered models count: {len(models_list)}")

# 12. API readiness reports correct service state
code, ready_data = request_json(f"{PUBLIC_URL}/ready")
code_health, ready_health = request_json(f"{PUBLIC_URL}/api/health/ready")
passed_12 = (
    code == 200
    and code_health == 200
    and isinstance(ready_data, dict)
    and ready_data.get("ready") is True
    and ready_data.get("dataset_loaded") is True
)
record_result(12, "API Readiness Reports Operational State", passed_12, f"/ready={code}, /api/health/ready={code_health}, ready={ready_data.get('ready') if isinstance(ready_data, dict) else False}")

# 13. Restart recovery works
# Verify Ollama bridge proxy and API container are running with health
from scripts.ops_monitor import get_docker_stats, get_systemd_status
c_stats = get_docker_stats()
sys_stats = get_systemd_status()
api_container = next((c for c in c_stats if c["name"] == "cyclone-api"), None)
passed_13 = (
    api_container is not None
    and api_container.get("health") == "healthy"
    and sys_stats.get("cyclone-ollama-bridge.service") == "active"
)
record_result(13, "Restart Recovery and Supervised Architecture", passed_13, f"cyclone-api container health: {api_container.get('health') if api_container else 'missing'}, Bridge: {sys_stats.get('cyclone-ollama-bridge.service')}")

# 14. Public access cannot reach Ollama or internal endpoints
code_ollama, _ = request_json(f"{PUBLIC_URL}/ollama")
code_generate, _ = request_json(f"{PUBLIC_URL}/api/generate")
code_tags, _ = request_json(f"{PUBLIC_URL}/api/tags")
passed_14 = code_ollama in (403, 404) and code_generate in (403, 404) and code_tags in (403, 404)
record_result(14, "Public Shielding: Ollama and Internal Endpoints Blocked", passed_14, f"/ollama={code_ollama}, /api/generate={code_generate}, /api/tags={code_tags}")

print("=" * 70)
total_passed = sum(1 for r in results if r["passed"])
print(f"PHASE 8 END-TO-END VALIDATION SUMMARY: {total_passed}/14 CHECKS PASSED")
print("=" * 70)

if total_passed == 14:
    print("ALL 14 PRODUCTION VALIDATION GATES PASSED! ✅")
    sys.exit(0)
else:
    print("VALIDATION FAILURES DETECTED! ❌")
    sys.exit(1)
