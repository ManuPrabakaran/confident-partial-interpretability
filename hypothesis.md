# CPI Pre-Experiment Hypotheses

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

- No scaling law is assumed a priori  
- No results are claimed  
- This document is a pre-registration of the experimental program  
