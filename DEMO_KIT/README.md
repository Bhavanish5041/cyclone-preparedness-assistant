# CYCLONE INTEL: Demonstration Kit & Presentation Master Guide

Welcome to the **CYCLONE INTEL Demonstration Kit**. This repository kit contains everything required to deliver an authoritative, presentation-ready demonstration of the **CYCLONE INTEL Decision Support Platform**—covering problem motivation, machine learning methodology, conversational AI orchestration, and automated continuous deployment.

---

## 1. Quick Navigation

| Document | Core Subject | Recommended Audience |
| :--- | :--- | :--- |
| [**01_PROJECT_OVERVIEW.md**](01_PROJECT_OVERVIEW.md) | Problem statement, spatial universe (1,879 units), core objectives, solution architecture, and feature status matrix. | All / Executive / Technical |
| [**02_SYSTEM_ARCHITECTURE.md**](02_SYSTEM_ARCHITECTURE.md) | End-to-end architecture diagrams (Mermaid), data flow pipelines, Docker services, and ingress security. | Architects / Evaluators |
| [**03_ML_PIPELINE.md**](03_ML_PIPELINE.md) | In-depth mathematical formulation of AHP/MCDA, Multinomial Logistic Regression surrogate classifier, and K-Means archetyping. | Data Scientists / ML Engineers |
| [**04_KMEANS_WANDB.md**](04_KMEANS_WANDB.md) | K-Means vulnerability archetyping, Weights & Biases (W&B) experiment tracking, run tables, and artifact links. | ML Evaluators / Judges |
| [**05_AI_ASSISTANT.md**](05_AI_ASSISTANT.md) | Local Qwen 2.5:0.5B model, 7 deterministic tools, refusal gates, and zero-hallucination grounding safeguards. | AI / NLP / Systems Engineers |
| [**06_MLOPS_CICD.md**](06_MLOPS_CICD.md) | 10 CI quality gate stages, GHCR image publishing with immutable tags (`sha-1221e7c`), and self-hosted CD engine. | DevOps / MLOps Engineers |
| [**07_LIVE_DEMO.md**](07_LIVE_DEMO.md) | Timed 10-15 minute presentation script with step-by-step "What to Show", "What to Say", and expected outputs. | Presenters / Demonstrators |
| [**08_COMMANDS.md**](08_COMMANDS.md) | Copy-paste terminal commands for live testing, API querying, container management, and automated cutover/rollback. | Presenters / Operators |
| [**09_METRICS_AND_RESULTS.md**](09_METRICS_AND_RESULTS.md) | Fully verified metrics ledger (accuracy, F1, silhouette, ARI, test counts) and explicit scientific limitations. | Academic / Technical Reviewers |
| [**10_QA_PREPARATION.md**](10_QA_PREPARATION.md) | Defensible answers to 15+ tough technical questions from professors, industry judges, and hiring evaluators. | Presenters |
| [**11_TROUBLESHOOTING.md**](11_TROUBLESHOOTING.md) | Rapid triage and instant recovery steps for common live demo failure scenarios (Ollama down, port conflicts). | Operators / Support |

---

## 2. Core Scientific & Operational Principles

When presenting CYCLONE INTEL, strictly uphold these four non-negotiable principles:

1. **Analytical Precedence of AHP Composite Scores:**  
   The deterministic Analytic Hierarchy Process (AHP/MCDA) preparedness tier (`Low` $< 40$, `Medium` $40 \le \text{score} < 60$, `High` $\ge 60$) is the canonical baseline authority.
2. **Surrogate ML Demarcation:**  
   The Logistic Regression model (94.3% accuracy, 95.4% AHP concordance) is strictly an *empirical surrogate classifier* approximating AHP tiers across 26 features. It does **not** predict storm tracks, wind speeds, or future casualties.
3. **Descriptive Spatial Archetyping:**  
   K-Means ($k=4$, 622 units in Dakshina Kannada and Udupi) discovers *descriptive vulnerability archetypes* to inform intervention planning. It does not predict disaster risk or damage, and is **never** extrapolated to Uttara Kannada (handled via fallback).
4. **Deterministic Tool Grounding:**  
   The local Qwen 2.5 LLM is strictly bound to 7 deterministic tools and a refusal gate. If the LLM daemon is offline, the system seamlessly falls back to deterministic rule synthesis with **zero downtime and zero hallucination**.

---

## 3. Pre-Demo Environment Checklist

Before commencing a live presentation or evaluation, verify the host status:

```bash
# 1. Verify Docker containers are active and healthy
docker compose ps

# 2. Check backend API health and summary
curl -s http://127.0.0.1:8000/health | jq .
curl -s http://127.0.0.1:8000/api/dashboard/overview | jq .priority_band_counts

# 3. Check local Ollama daemon status
curl -s http://127.0.0.1:11434/api/tags | jq .models[].name

# 4. Verify test suite status (32 CI/CD and deployment tests)
.venv/bin/pytest tests/test_ci_cd_workflow.py tests/test_deploy_production.py -v
```

---

## 4. Repository Assets Included

- **`assets/screenshots/`**: High-resolution interface captures of the React Web Platform (Overview, Community Dossier, GIS Spatial Explorer, Scenario Analysis, AI Readiness, Reports).
- **`assets/results/`**: Empirical research visualizations including K-Means PCA projection ($k=4$), cluster balance curves, AHP sensitivity maps, and healthcare geodesic distance choropleths.
- **`assets/architecture/`**: Visual diagrams detailing data ingestion, ML scoring pipelines, and self-hosted CD workflows.
