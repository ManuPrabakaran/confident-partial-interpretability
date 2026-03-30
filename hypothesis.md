# CPI Pre-Experiment Hypotheses

## Novel thesis: interpretability scaling laws

The **central novel conjecture** in this research program is an **interpretability scaling law**: as **effective complexity** of the model (and its representation under the task distribution) grows, **coverage (C)** and **confidence (K)** need not stay high—they are hypothesized to **degrade along predictable curves**, analogous in spirit to capability scaling laws.

Concretely (aligned with `pre-registration_paper.md` §11):

- **C ∼ f₁(dim_eff)** and **K ∼ f₂(dim_eff)**  
- **dim_eff** denotes effective dimensionality of the activation manifold under task distribution **D** (not raw parameter count alone).  
- Candidate shapes (power-law decay, log-linear decay, etc.) are **empirical questions** to be fit after measuring **C** and **K** across controlled scaling ladders.

**Why it matters.** If such relationships hold, one could **forecast** how much interpretability to expect at a given scale and frame **tradeoffs** between capability and CPI-style monitoring. Full methodology and limits are in the pre-registration paper; this file states the hypothesis for the **pre-experiment** program.

**How we test it (high level).** Train or obtain models across a scaling axis (width, depth, data, or estimated **dim_eff**), hold the **CPI measurement protocol** as fixed as possible, sample states, estimate **C** and **K**, then fit and stress-test **f₁**, **f₂**.

---

## Core Hypothesis

Confident Partial Interpretability (CPI) is a measurement and control framework for turning partial mechanistic interpretability into practical runtime safety constraints.

CPI operationalizes interpretability using two metrics:

- Coverage (C): Fraction of causally relevant internal states for which a reliable interpretation exists  
- Confidence (K): Probability that an interpretation correctly predicts behavior under causal intervention  

CPI does not assume that interpretability is globally sufficient to solve alignment. Instead, it claims that sufficiently confident interpretability over a bounded domain is enough to make alignment operationally tractable within that domain.

---

## Primary Predictions

### Prediction 1: Metric Validity

Confidence (K) will correlate with causal intervention success.

- High K → predictions about intervention outcomes are correct  
- Low K → predictions fail  

Success condition:
K reliably predicts intervention outcomes across sampled states.

---

### Prediction 2: Progressive Improvement

Improvements in interpretability methods will increase either:

- Coverage (C), or  
- Confidence (K), or both  

Success signal:
New methods expand the tractable domain or improve prediction reliability.

---

### Prediction 3: Constraint Efficacy

Restricting models to high-confidence regions (CPI-style domain restriction) will measurably constrain behavior.

Success condition:
Domain restriction changes outputs in ways consistent with the interpretability function.

Note:
Capability retention is not assumed.

---

## Experimental Plan

### Model Setup

- Small transformer (1–2 layers initially)  
- Simple, mechanistically analyzable task  

---

### Measurement Pipeline

1. Sample inputs from task distribution D  
2. Sample layer ℓ and token position t  
3. Estimate causal relevance via intervention  
4. Compute K:
   - Predict intervention effect  
   - Compare to observed outcome  
5. Compute C:
   - Fraction of relevant states with K ≥ threshold  

---

## Validation Criteria

### Success if ANY of the following hold:

- K predicts intervention outcomes  
- C or K improves with better methods  
- Domain restriction constrains behavior  
- C/K guide research direction  

---

### Failure if ALL of the following hold:

- High K does not predict outcomes  
- C cannot be expanded  
- Domain restriction has no effect  
- Metrics provide no useful signal  

---

## Notes

- The scaling-law conjecture is **not** assumed when judging Predictions 1–3 above: those gates ask whether **K** and **C** are measurable and useful **before** we commit to a particular **f₁**, **f₂**.  
- No results are claimed in this document.  
- Full pre-registration (methodology, enforcement sketch, scaling section): `pre-registration_paper.md`  
