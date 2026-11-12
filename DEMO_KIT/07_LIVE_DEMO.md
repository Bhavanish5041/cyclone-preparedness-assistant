# 07. Live Demonstration Script (10–15 Minutes)

This step-by-step presentation script is organized into 6 structured phases. It specifies exactly **what to say**, **what to show on screen**, and the **expected outputs**.

---

## Presentation Timeline

| Phase | Time | Topic | Primary Screen / Tool |
| :---: | :---: | :--- | :--- |
| **1** | 0:00 – 2:00 | Problem Statement & System Architecture | Slide / [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) |
| **2** | 2:00 – 5:00 | Decision Engine, AHP vs. ML, & The Tallur Invariant | Web UI: Overview & [03_ML_PIPELINE.md](03_ML_PIPELINE.md) |
| **3** | 5:00 – 8:00 | React Web Platform & Spatial GIS Explorer | Web UI: Community Profiles & GIS Map |
| **4** | 8:00 – 11:00 | Conversational AI Assistant & Guardrails | Web UI: AI Assistant Drawer |
| **5** | 11:00 – 13:00 | MLOps, W&B Experiments & 10-Stage CI/CD | Terminal / W&B Workspace / GitHub Actions |
| **6** | 13:00 – 15:00 | Technical Limitations & Q&A Transition | Summary Slide / [10_QA_PREPARATION.md](10_QA_PREPARATION.md) |

---

### Phase 1: Problem Statement & Architecture (0:00 – 2:00)

**What to Say:**
> *"Good morning. Today I am presenting **CYCLONE INTEL**, a spatial decision-support platform designed for disaster management authorities along Karnataka's 320-kilometer coastline.  
> 
> Coastal Karnataka encompasses 1,879 revenue villages and census towns across Dakshina Kannada, Udupi, and Uttara Kannada. During severe monsoons and cyclones, emergency responders must answer two vital questions: Which communities require urgent intervention, and what specific resources do they need?  
> 
> Traditional disaster response suffers from two critical flaws: First, relying on ungrounded black-box LLMs that hallucinate non-existent emergency shelters; second, relying on fragile cloud APIs that collapse during storm-induced power and telecommunications outages.  
> 
> CYCLONE INTEL solves this with an end-to-end edge-deployable platform: deterministic multi-criteria scoring via the Analytic Hierarchy Process, descriptive vulnerability archetyping via K-Means, boundary auditing via a 94.3% accurate surrogate classifier, and a private, air-gapped conversational assistant running locally on CPU via Ollama."*

**What to Show:**
- Display the high-level architecture diagram from [02_SYSTEM_ARCHITECTURE.md](02_SYSTEM_ARCHITECTURE.md).

---

### Phase 2: The Decision Engine & The Tallur Invariant (2:00 – 5:00)

**What to Say:**
> *"Before opening the dashboard, I want to clarify a key methodological distinction: **Preparedness Tier** versus **Operational Priority Band**.  
> 
> A community's **AHP Preparedness Tier** (Low, Medium, High) measures installed infrastructure capacity. However, our **Operational Priority Band** (Bands 1 to 4) determines immediate action urgency. For example, a village with a Medium preparedness score of 52 that sits 1.5 km from the coastline or has critically low shelter space is immediately promoted to **Band 2 (Specialized)** to protect lives.  
> 
> Across all 1,879 units, our census is strictly reconciled:
> - **Band 1 (Critical):** 155 units (AHP Low tier, score < 40)
> - **Band 2 (Specialized):** 829 units (Medium tier with sector deficit triggers)
> - **Band 3 (Moderate):** 299 units (Medium tier without critical triggers)
> - **Band 4 (Adequate):** 596 units (High tier, score >= 60)
> 
> A central test invariant throughout our regression suite is **Tallur** in Udupi district (PC11 ID: 608691). With an AHP score of 74.34 and high infrastructure access, Tallur is classified into **Band 4 (Adequate)**."*

**What to Show:**
- Show the Reconciled Priority Band table from [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md).

---

### Phase 3: Live Web UI & GIS Walkthrough (5:00 – 8:00)

**What to Show & Click:**
1. Navigate to the Web Platform: `http://localhost:80/` (or via Cloudflare tunnel).
2. **Overview Screen:** Show the four summary cards:
   - Total Units: `1,879`
   - Band 1 Critical: `155`
   - Band 2 Specialized: `829`
   - High Preparedness: `596`
3. **Community Profiles:** Search for `Tallur` in the search bar.
   - Click on **Tallur (Udupi)**.
   - Point out the metrics: AHP Score: **74.34**, Tier: **High**, Priority Band: **BAND_4_ADEQUATE**.
   - Point out the healthcare distance breakdown: Primary Health Sub-centre: **0.00 km** (in-village), PHC: **3.09 km**, Apex Hospital: **25.98 km**.
4. **Inspect an Out-of-Study Unit:** Search for `Bappalgundi` (Uttara Kannada).
   - Point out that its AHP Score is **38.31** (**BAND_1_CRITICAL**).
   - Show the spatial banner: *`Spatial Coverage: Unassigned to K-Means clustering (limited to DK & Udupi). Evaluated strictly via deterministic sector fallback.`* This proves we never extrapolate unvalidated models.

**What to Say:**
> *"Notice the transparency in the UI. For Tallur, the system clearly disambiguates that while a sub-centre is inside the village boundary (0.00 km), the nearest full PHC is 3.09 km away. And for Bappalgundi in Uttara Kannada, the system explicitly marks it as unassigned to K-Means archetyping, preventing false geographic generalizations."*

---

### Phase 4: Conversational AI Assistant & Guardrails (8:00 – 11:00)

**What to Show & Click:**
Click the **AI Assistant** icon in the bottom right corner of the dashboard to open the chat drawer.

#### Demo Turn 1: Grounded Community Explanation
- **Type prompt:** `Explain why Tallur is classified as Band 4.`
- **Expected Output:**
  - Assistant responds in under 1 second.
  - Cites exact AHP score: **74.34**, High tier, and **BAND_4_ADEQUATE**.
  - Highlights high road connectivity and local sub-centre access.
  - Attaches standard statutory advisory planning notice.

#### Demo Turn 2: Active Refusal Gate Test
- **Type prompt:** `What is the current wind speed of Cyclone Biparjoy and will it make landfall in Mangalore tonight?`
- **Expected Output:**
  - **Immediate Refusal**: The assistant refuses the prompt with:  
    *"Refusal Notice: CYCLONE INTEL is a structural preparedness and spatial decision-support platform based on baseline infrastructure indicators. It does NOT forecast real-time meteorological cyclone trajectories, live storm surge inundation, or casualty estimates. For live weather warnings, consult the India Meteorological Department (IMD) at https://mausam.imd.gov.in."*

#### Demo Turn 3: Grounded Multi-Turn Intervention Query
- **Type prompt:** `What interventions are recommended for Edmoge?`
- **Expected Output:**
  - Assistant calls `get_intervention_recommendations`.
  - Recommends `ADV_1_HOSPITAL_ISOLATION` (emergency transit and PHC supplies due to hospital distance > 20 km).

**What to Say:**
> *"Notice what just happened in Turn 2. When prompted for live cyclone tracking, the assistant did not hallucinate a fake weather alert. Our active refusal gate immediately caught the out-of-scope query and directed the user to the IMD. In Turn 1 and 3, all numbers and recommendations came directly from our deterministic tool registry."*

---

### Phase 5: MLOps, W&B Tracking & 10-Stage CI/CD (11:00 – 13:00)

**What to Say:**
> *"Now let's examine the engineering rigor supporting this system. Every model and dataset is version-controlled and tracked in Weights & Biases under entity `bhavanishreddy69-na`. Our 10 clustering runs and 5 prioritization sensitivity experiments are fully synchronized.  
> 
> Furthermore, all code changes pass through a 10-stage automated CI quality gate in GitHub Actions covering 1,107 automated tests. Untrusted Pull Request code is strictly isolated in cloud containers and never executes on our self-hosted runner.  
> 
> Production deployment requires manual maintainer approval in GitHub, an exact immutable image tag like `sha-1221e7c`, and automated health check polling with instant atomic rollback."*

**What to Show:**
- In terminal, execute the tampering detection simulation:
```bash
.venv/bin/pytest tests/test_model_integrity_and_safeguards.py -v -k "tamper"
```
- Or run the preflight verification check:
```bash
.venv/bin/pytest tests/test_ci_cd_workflow.py tests/test_deploy_production.py -v
```
*(Show all 32 tests passing cleanly).*

---

### Phase 6: Limitations & Q&A Transition (13:00 – 15:00)

**What to Say:**
> *"To conclude, CYCLONE INTEL represents an explainable, privacy-preserving, edge-resilient decision-support platform.  
> 
> We are transparent about our boundaries:
> 1. Our baseline demographic and infrastructure indicators reflect the 2011 Census and SHRUG geospatial datasets.
> 2. K-Means clustering is descriptive archetyping, not disaster impact prediction.
> 3. The platform provides decision-support advisories for planners; it does not replace statutory administrative evacuation decrees.
> 
> Thank you, and I look forward to your questions."*
