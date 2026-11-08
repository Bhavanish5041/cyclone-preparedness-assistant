# Intended Use and Model Limitations

## Operational Scope
This model is developed for **decision-support assistance** across the coastal districts of Karnataka (Dakshina Kannada, Udupi, Uttara Kannada).

## Core Limitations
1. **AHP-Tier Surrogate (Not Casualty/Hazard Risk):**
   The model predicts preparedness capacity bands derived from Analytical Hierarchy Process (AHP) scoring. It is **NOT** a calibrated physical cyclone-risk or casualty forecast.

2. **Deterministic AHP Precedence:**
   In any case of conflict between deterministic AHP scoring and Logistic Regression predictions, the deterministic AHP score remains authoritative.

3. **Low-Tier Recall Caveat (73.91%):**
   On held-out testing, Low-tier recall dropped to 73.91% (17 correct, 6 misclassified as Medium). In disaster relief, under-estimating vulnerability is severe. All villages with Low AHP tier or Low predicted probability > 20% should receive priority inspection.

4. **Boundary Case Ambiguity:**
   Villages with top probability margin < 10% or maximum probability < 60% are flagged as boundary cases and require manual review.

5. **Static Geographic Baseline:**
   Trained on 2011 census demographics and infrastructure baseline. Does not reflect infrastructure improvements constructed after baseline collection.
