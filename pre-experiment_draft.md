# CPI: Confident Partial Interpretability as an Operational Alignment Framework  
Manu Prabakaran  
Rice University · mp228@rice.edu  

---

## Abstract

We introduce Confident Partial Interpretability (CPI), a measurement framework for quantifying how much of a neural network’s internal behavior is currently interpretable and how reliably those interpretations generalize under causal intervention. CPI defines two core metrics: Coverage (C), the fraction of causally relevant internal states for which a reliable interpretation exists, and Confidence (K), the probability that a given interpretation correctly predicts model behavior under causal intervention.

CPI argues that sufficiently confident interpretability over a bounded domain is enough to make alignment tractable within that domain—converting it from an open philosophical problem into a monitoring and enforcement problem with measurable guarantees. We further hypothesize that C and K degrade along a predictable Interpretability Scaling Law with model complexity, analogous to capability scaling laws. If true, this would allow forecasting of how much interpretability we can expect from models of a given effective dimensionality and how much capability can safely be permitted under a CPI-style whitelist.

This document pre-registers the CPI framework, its measurement methodology, and its central scaling-law hypothesis prior to empirical validation. No empirical results are reported.

---

## 1. Introduction

Modern neural networks achieve high performance while remaining largely opaque in their internal workings. Interpretability research has produced mechanistic explanations for specific circuits—but it lacks a standardized, operational definition of what it means for a model to be “interpretable enough” for safe, controlled deployment. Without such a definition, it is difficult to track progress, compare methods, or forecast how interpretability will scale with model size.

This paper proposes Confident Partial Interpretability (CPI) as a measurement framework that reframes interpretability as a quantifiable, empirically grounded coordinate (C, K) in a two-dimensional state space. CPI does not prescribe a theory of cognition or a fixed ontology of goals or beliefs. Instead, it defines a minimal, testable measurement protocol built on causal interventions and systematic sampling over the model’s internal states. The framework is agnostic to internal architecture and can, in principle, be applied to any differentiable system.

CPI makes three contributions:

1. A measurement framework. The (C, K) coordinate provides a common language for comparing interpretability methods, tracking research progress, and auditing deployed models.  
2. A domain-restriction mechanism. If we can reliably interpret and trust a subset of the model’s state space, we can restrict the model to operate only within that region. This converts interpretability progress into concrete operational safety constraints.  
3. A scaling-law hypothesis. As model complexity increases, both C and K degrade along a predictable curve. If this holds, it enables forecasting of how much interpretability we can expect from a model of a given effective dimensionality.  

The paper is structured as follows: Sections 2–3 define the CPI metrics. Section 4 addresses what operational alignment tractability actually means under CPI, and clarifies the limits of that claim. Sections 5–8 develop the measurement methodology. Sections 9–11 present the enforcement mechanism. Section 12 advances the scaling-law hypothesis. Section 13 pre-registers the empirical program.

---

## 2. What CPI Is

CPI is a measurement framework—not a theory of cognition or alignment. It defines two instruments for quantifying the interpretability state of a neural network at any point in time:

1. Coverage (C): The fraction of a model’s relevant internal states for which a reliable interpretation exists.  
2. Confidence (K): The probability that a given interpretation correctly predicts model behavior under causal intervention.  

CPI treats (C, K) as a coordinate in interpretability space. The current default coordinate for any real deployed model is approximately (C = 0, K = 0). The research goal is to move this coordinate upward and track what happens to safety, capability, and deployment risk as it does.

CPI does not claim that interpretability by itself solves alignment. Rather, it claims that if a sufficiently confident and partial interpretability function exists over some domain, then many aspects of alignment become tractable monitoring and enforcement problems within that domain. The boundaries of that claim are explored carefully in Section 4.

---

## 2.1 State Definition

A state s is a distribution over residual stream activations at layer ℓ and token position t, under an input distribution D, optionally conditioned on an intervention I. States are defined as distributions because individual activation vectors are too input-dependent to serve as stable objects of interpretation. A single vector is treated as a sample from the underlying state distribution, not as the state itself.

A state dimension is relevant if perturbing it produces a statistically significant change in the output distribution beyond a small threshold ε under task T. Dimensions that are dead, noise, or causally inert under task-relevant conditions are excluded from the relevant state space.

---

## 3. Why These Two Measurements?

### 3.1 Why K and Not Just Accuracy?

Standard performance metrics measure what a model does; K measures whether we understand why it does it. A model can perform well for the wrong reasons while its internals behave in ways that mismatch our assumptions.

Formally, K is defined via causal intervention: a prediction of the form “Intervening on component X will shift the output distribution by Δ under task T” is judged correct if the observed distribution-shift matches the predicted shift within a tolerance under a specified divergence metric (e.g., KL divergence or total variation distance). Because states are distributions, K measures predictive validity over distributional changes, not point predictions.

Thus:

1. High accuracy + low K: The model is doing something we don’t understand.  
2. High accuracy + high K: We actually know what the model is doing, at least in the regions where we have interpretations.  

---

### 3.2 Why C and Not Just K?

K tells us how reliable our interpretations are in the regions where they exist. C tells us how much of the model those interpretations cover.

A model could have K = 0.95 but C = 0.03, meaning we understand 3% of its behavior extremely well and have essentially no idea about the remaining 97%. C forces honesty about coverage gaps.

Together, C and K form a (coverage, confidence) coordinate that describes where a model sits in interpretability space and how much of that interpretability is empirically grounded.

---

## 4. What CPI Does and Does Not Guarantee

### 4.1 Operational Tractability

If K reaches 1 over some domain D and we restrict operation to that domain, alignment is operationally tractable for that deployment context. Within this region, model behavior is fully predictable under intervention, continuously monitorable, and in principle steerable through inversion of the interpretability function f.

This does not imply that alignment is solved globally. The guarantee holds only within the domain where both coverage and confidence are sufficient. Outside that domain, the model remains unconstrained.

Operational tractability should therefore be understood as a domain-relative claim: alignment becomes a bounded engineering problem inside the interpreted region, while remaining an open problem outside it.

---

### 4.2 Interpretability Is Not Alignment

K measures predictive correctness, not objective desirability. A model can be perfectly interpretable and still pursue undesirable goals. CPI does not claim that interpretability guarantees alignment in a normative sense. It claims that interpretability enables monitoring and control, which are necessary components of alignment.

---

### 4.3 Representation Aliasing

Multiple internal objectives or policies may map to the same interpreted feature. This is representation aliasing. In such cases, the interpretability function f may appear to correctly predict behavior under intervention while still misrepresenting the underlying objective.

Aliasing can be detected when K fails to generalize across distribution shifts or adversarial probes. Resolving aliasing requires increasing representational resolution, not merely improving predictive calibration.

---

### 4.4 Compositional Opacity

Even if individual states are interpretable, their composition may produce emergent behavior that is not predictable from the components. This is a coverage problem: compositional states that affect output behavior must be included in the relevant state space and measured under C.

---

### 4.5 Deception Inside the Safe Region

CPI constrains transitions into unmeasured regions, but it does not eliminate the possibility of deceptive objectives that are already present inside the interpreted domain. In this case, deception is reduced to interpretability ambiguity rather than hidden behavior outside the monitored region.

---

## 5. The Output Space Y

### 5.1 Ideal Y

In the ideal case, Y represents a complete telemetry of the model’s internal cognition at any point in time. This includes everything the model currently attends to, its active goals and subgoals, its reasoning trace, its uncertainty state, and its counterfactual salience.

Such a representation would reduce alignment to a monitoring and enforcement problem. However, this ideal is not currently achievable.

---

### 5.2 Causal Discovery

Instead of predefining Y, CPI constructs it through causal discovery:

1. Fix a task T  
2. Perform systematic interventions across internal states  
3. Identify the minimal subspace whose intervention profile explains output variance  
4. Define this subspace as Y  
5. Label dimensions post-hoc, subject to validation via K  

Under this procedure, Y is not assumed—it is discovered. Labels such as “goal” or “inclination” describe causal roles identified through intervention, not prior conceptual assumptions.

---

### 5.3 Task-Specific Y

Y is task-dependent. Different tasks induce different causal subspaces. This is not a limitation: a model only needs to be interpretable with respect to the tasks it is deployed on.

The minimal working Y currently targets:

- Goal state: dimensions mediating what the model is trying to achieve  
- Inclination state: dimensions mediating implicit tradeoffs or biases  

These are placeholders pending empirical discovery.

---

## 6. How to Compute K

K = (correct intervention predictions) / (total interventions attempted)

A prediction is of the form:
“Intervening on state s will change output behavior by Δ.”

A prediction is correct if the observed change matches the predicted change within tolerance under a chosen divergence metric.

---

### 6.1 K-Measurable Domain

K is only defined over states where interventions can be performed and evaluated. This defines the tractable domain. States outside this domain are excluded from K but still count against coverage.

---

### 6.2 Current Tools

Current interpretability methods (activation patching, ablation, sparse autoencoders, steering) provide a limited K-measurable domain, mostly restricted to small models and clean circuits.

---

## 7. How to Compute C

C = (states with K ≥ τ) / (total relevant states sampled)

---

### 7.1 Relevant States

Relevant states are defined by causal influence: perturbations that produce statistically significant changes in output behavior beyond ε.

---

### 7.2 Two Gaps

C captures two distinct gaps:

- Tractability gap: states where measurement is not possible  
- Confidence gap: states where measurement is possible but K is insufficient  

Both count against coverage.

---

### 7.3 Sampling

States must be sampled from the full relevant state space, not from already-interpreted regions.

In practice:
- Sample inputs from D  
- Sample layer ℓ and token position t uniformly or according to activation salience  
- Estimate relevance via randomized interventions  

---

## 8. Domain Restriction Mechanism

CPI introduces a domain restriction mechanism that limits model operation to a safe subset of state space.

S_safe = { h : K(h) ≥ τ }

---

### 8.1 Whitelisting

The model is restricted to operate only within S_safe. This is a whitelist approach: instead of identifying dangerous states, we identify safe ones and prohibit everything else.

S_safe is not explicitly enumerated but implicitly defined by the set of states for which K exceeds threshold τ.

---

### 8.2 Layer-wise Enforcement

At each layer:

1. Compute candidate activation h  
2. Evaluate whether h ∈ S_safe  
3. If not, intervene (block, project, or rollback)  

This can be implemented as a forward hook on the model’s computation.

---

### 8.3 Policy Thresholds

The choice of τ is a policy decision that depends on acceptable risk tolerance and deployment context.

---

## 9. When Is Alignment Operationally Tractable?

If K → 1 over a domain and the model is restricted to that domain, then:

- Behavior is predictable under intervention  
- Internal state is monitorable  
- Control is possible through steering  

Alignment becomes a bounded engineering problem involving coverage expansion, robustness under distribution shift, and capability tradeoffs.

---

## 10. The Case for CPI

### 10.1 Feasibility

Neural networks are fully observable systems. If a mapping f from internal states to causal descriptions exists over some domain, then monitoring and control become possible.

---

### 10.2 Remaining Risks

CPI does not eliminate all risks. Remaining failure modes include:

- Representation aliasing  
- Compositional opacity  
- Verifier failure  
- Computational cost of enforcement  
- Deception within the interpreted domain  

---

## 11. Interpretability Scaling Law

We hypothesize:

C ∼ f₁(dim_eff),   K ∼ f₂(dim_eff)

where dim_eff is the effective dimensionality of the activation manifold under task distribution D.

---

### 11.3 Expected Functional Forms

Candidate forms include:

- Power-law decay  
- Log-linear decay  

The exact form is an empirical question.

---

### 11.4 Experimental Design

To test the scaling law:

1. Train models of varying size  
2. Estimate dim_eff  
3. Sample states and compute C and K  
4. Fit functional relationships  

---

## 12. Research Program

The CPI research program proceeds in stages:

1. Measure C and K on small models  
2. Scale measurement across architectures  
3. Expand coverage through improved methods  
4. Evaluate domain restriction effectiveness  

---

## 13. Long-Horizon Conjecture

As interpretability methods improve, K approaches a ceiling on the covered domain. Coverage expansion remains the primary bottleneck.

---

## 14. Pre-Registration

This document pre-registers the CPI framework, measurement methodology, and scaling-law hypothesis prior to empirical validation.

No empirical results are reported.

---

## References

Elhage et al., Transformer Circuits (2021)  
Olah et al., Superposition (2022)  
Causal Scrubbing (2022)  
Sparse Autoencoders (2023)  
Hubinger et al., Risks from Learned Optimization (2019)  
ARC, Eliciting Latent Knowledge (2021)  
Representation Engineering (2023)  
Activation Steering (2023)  

---

Signed: Manu Prabakaran  
Date: March 30, 2026


