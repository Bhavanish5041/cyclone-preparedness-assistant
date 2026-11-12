# 08. Operational & Demonstration Commands

This reference contains copy-paste ready commands for verifying, testing, and presenting the CYCLONE INTEL platform.

---

## 1. Quick System Health Checks

```bash
# 1. Check Docker container runtime status
docker compose ps

# 2. Check API health endpoint
curl -s http://127.0.0.1:8000/health | jq .

# 3. Check Dashboard overview metrics and priority band counts
curl -s http://127.0.0.1:8000/api/dashboard/overview | jq .priority_band_counts

# 4. Verify local Ollama LLM daemon and models
curl -s http://127.0.0.1:11434/api/tags | jq '.models[] | {name: .name, size_mb: (.size / 1024 / 1024 | round)}'

# 5. Check Cloudflare Tunnel process status
pgrep -fl cloudflared || echo "Cloudflare tunnel is not running as a host process"
```

---

## 2. Practical API Queries for Demonstrations

### Query A: Search for Tallur (PC11: 608691)
```bash
curl -s "http://127.0.0.1:8000/api/communities?search=Tallur&limit=5" | jq '.items[] | {name: .village_name, district: .district_name, score: .preparedness_score, tier: .preparedness_tier, band: .priority_band}'
```

### Query B: Fetch Detailed Dossier for Edmoge (Udupi)
```bash
curl -s "http://127.0.0.1:8000/api/v1/decision-support/units/608720" | jq '{name: .village_name, score: .preparedness_score, band: .provisional_priority_band, archetype: .vulnerability_archetype_label, advisories: [.recommended_advisories[].directive_id]}'
```

### Query C: Fetch Out-of-Study Area Unit Bappalgundi (Uttara Kannada)
```bash
curl -s "http://127.0.0.1:8000/api/v1/decision-support/units/Bappalgundi" | jq '{name: .village_name, district: .district_name, band: .provisional_priority_band, cluster_status: .cluster_assignment_status, advisories: [.recommended_advisories[].directive_id]}'
```

### Query D: Compare Districts (Dakshina Kannada vs. Udupi)
```bash
curl -s "http://127.0.0.1:8000/api/districts/compare?district_a=Dakshina%20Kannada&district_b=Udupi" | jq .
```

### Query E: Test Assistant Chat Endpoint (Grounded Tool Execution)
```bash
curl -s -X POST "http://127.0.0.1:8000/api/assistant/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain why Tallur is classified as Band 4.", "conversation_history": []}' | jq '{answer: .answer, fallback: .is_fallback, tools_called: .tools_called}'
```

### Query F: Test Assistant Active Refusal Gate
```bash
curl -s -X POST "http://127.0.0.1:8000/api/assistant/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the live wind speed of Cyclone Biparjoy?", "conversation_history": []}' | jq '{answer: .answer, fallback: .is_fallback}'
```

---

## 3. Targeted Test Execution Commands

```bash
# A. CI/CD Workflow & Production Deployment Tests (32 tests passing)
.venv/bin/pytest tests/test_ci_cd_workflow.py tests/test_deploy_production.py -v

# B. Model Integrity, Cryptographic Hashes & Tampering Tests (45 tests)
.venv/bin/pytest tests/test_model_integrity_and_safeguards.py tests/test_local_model_registry.py -v

# C. Model Quality Evaluation Gates (14 tests)
.venv/bin/pytest tests/test_model_quality_gates.py -v

# D. 7 Deterministic Assistant Tools & Acceptance Suite (30 tests)
.venv/bin/pytest tests/test_assistant_tools.py tests/test_assistant_acceptance.py -v

# E. End-to-End User Journey Regression & Tallur Invariant (20 tests)
.venv/bin/pytest tests/test_e2e_regression.py tests/test_route_inventory.py -v

# F. Fast Backend Smoke Check (all tests excluding heavy CyclonLM experiments)
.venv/bin/pytest tests/ -m "not slow" --ignore=tests/cyclonelm -q
```

---

## 4. Live Continuous Deployment Simulation

You can safely demonstrate the production deployment engine on your laptop using isolated simulation modes that **do not modify production containers**:

```bash
# 1. Dry Run Inspection (validates image tag and preflight checks without mutations)
python3 scripts/deploy_production.py --image-tag sha-1221e7c --dry-run

# 2. Isolated Simulation Deployment (spins up temporary simulation stack and probes health)
python3 scripts/deploy_production.py --image-tag sha-1221e7c --simulate

# 3. Automated Atomic Rollback Demonstration (simulates probe failure and verifies rollback)
python3 scripts/deploy_production.py --image-tag sha-1221e7c --simulate --simulate-failure
```

---

## 5. Weights & Biases (W&B) Reproduction Commands

```bash
# 1. Run local offline W&B experiment tracking (records to ./wandb/)
python scripts/run_wandb_tracking.py --mode offline

# 2. Validate registered model artifact cache
python scripts/model_registry_cli.py validate
```
