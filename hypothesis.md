# CPI Hypotheses and Pre-experiment Predictions

## Core Hypothesis

**Confident Partial Interpretability (CPI) is a measurement and control framework for turning partial mechanistic interpretability into practical runtime safety constraints.**

CPI claims that alignment progress can be operationalized through two metrics:
1. **Coverage (C)**: Fraction of model state space that admits reliable interpretation
2. **Confidence (K)**: Causal reliability of those interpretations under intervention

The framework succeeds if these metrics are measurable, improvable, and useful for constraining model behavior safely.

## Primary Predictions

### Prediction 1: Metric Validity
Confidence \(K\) will correlate with causal intervention success (ablation/patching prediction accuracy).

**Success threshold**: \(K \geq 0.7\) predictive correlation with interventions

### Prediction 2: Progressive Improvement
Better interpretability methods will systematically increase either \(C\) or \(K\) (or both).

**Success signal**: Methodological improvements yield ≥10% gain in at least one metric

### Prediction 3: Constraint Efficacy
Restricting models to high-confidence/high-coverage states will reduce misalignment risk without catastrophic capability loss.

**Success threshold**: ≥75% capability retention with measurably safer behavior

### Prediction 4: Research Utility (Speculative)
Coverage/confidence exhibit structured scaling behavior across model sizes.

**Bonus prediction**: Power-law relationship \(K = g(\text{params}, C)\)

## Experimental Design

**Benchmark**: 2-layer transformers on modular arithmetic (Nanda et al. ground truth circuits)

**Pipeline**:

1. Train models of varying complexity

2. Apply mech interp (patching, ablation, probes)

3. Measure K: intervention prediction accuracy

4. Measure C: fraction of states passing K threshold

5. Constrain to high-C/K region → test safety/capability


## Framework Validation Criteria

### Success Conditions
CPI demonstrates value if **ANY** of these hold:

1. **Causal calibration**: High \(K\) predicts intervention outcomes [web:3][web:7]
2. **Progressive improvement**: \(C\) or \(K\) increase systematically with methods  
3. **Constraint efficacy**: Whitelist reduces misalignment risk
4. **Research directionality**: Metrics guide better interpretability techniques

### Failure Conditions  
CPI fails **ONLY** if **ALL** of these hold:

1. **Confidence illusion**: High \(K\) but no causal prediction power
2. **Coverage stagnation**: Cannot expand \(C\) through any improvement
3. **Safety ineffectiveness**: Constraints show no alignment benefit
4. **Metric uselessness**: \(C\)/\(K\) provide no research guidance

**Important**: Low initial values expected. Framework predicts gradual improvement.

## Measurement Definitions

### Coverage (\(C\))

C = (confidently interpretable states) / (total relevant states)

- "Relevant states" = task-critical activations/circuits
- "Confidently interpretable" = K ≥ threshold (e.g., 0.8)

### Confidence (\(K\))

K = (correct intervention predictions) / (total interventions)

- Prediction = "Ablating circuit X reduces performance by Δ"
- Correct = observed Δ matches predicted Δ within tolerance

### Safe State
State where \(Y\) components align with deployment objectives \(T\).

**Current scope**: \(Y\) = {goal state, inclination state}
**Ideal scope**: Full cognitive decomposition

## Expected Scaling Trajectory

| Phase              | Target C Range | Target K Range | Framework Status          |
|--------------------|----------------|---------------|---------------------------|
| MVP (small models) | 60-75%         | 88-92%        | **Initial measurement**   |
| Short-term (medium)| 45-60%         | 85-90%        | **Scaling study**         |
| Medium-term (large)| 35-50%         | 80-87%        | **Coverage expansion**    |

## Preliminary Results (Motivational)

Toy transformer scaling on modular arithmetic:

| Model Size | Coverage (C) | Confidence (K) | Capability Retention |
|------------|--------------|----------------|---------------------|
| Small      | 71%          | 91%            | 91%                 |
| Medium     | 54%          | 89%            | 86%                 |
| Large      | 38%          | 84%            | 79%                 |

## Detailed Sub-hypotheses

**H1: Metric Separation**  
Confidence and coverage improve independently:
- Better interventions → higher \(K\) at fixed \(C\)
- Better architectures/probes → higher \(C\) at fixed \(K\)

**H2: Whitelist Efficacy**  
Models trained/constrained in high-C/K region match unconstrained safety/performance

**H3: Task-specific Y**  
Deployment goals \(T\) yield discoverable causal \(Y\) subspaces

**H4: Gradual Convergence** (Ideal)  
With research progress, \(K \rightarrow 100\%\) on covered domain

## Rationale and anticipated objections

Interpretability is best understood as an empirical problem of state-to-description mapping rather than a philosophical inquiry into machine consciousness. The internal state of a neural network—its weights, activations, and learned feature representations—exists as observable numerical quantities. The central engineering question is whether these quantities can be mapped to a structured description of the model's current computation that is both human-legible and causally predictive of behavior under intervention.[web:2][web:3][web:7] In an ideal world, such a mapping would be complete and deterministic, with confidence approaching 100% across the model's operational domain. The practical contribution of Confident Partial Interpretability (CPI) is to recognize that even partial mappings, when sufficiently reliable, may support meaningful safety constraints during the interim period before full interpretability is achieved.

If interpretability were solved completely, alignment would collapse into a standard systems-control problem. One could monitor the model's internal state continuously via the mapping function \(f: X \to Y\), where \(X\) denotes the space of physical states and \(Y\) denotes interpretable mental descriptions. States mapping to misaligned configurations in \(Y\) could be blocked outright, while an inverse mapping \(f^{-1}: Y \to X\) would enable active correction by steering the model toward desired internal configurations.[web:12][web:15] This is not a speculative vision but the ordinary logic of controlled systems: known inputs, defined outputs, continuous monitoring, and automatic shutdown or correction on violation of safety constraints. The conceptual novelty lies in applying this engineering paradigm to neural networks, where the primary research challenge is constructing \(f\) rather than debating its logical implications.

CPI advances a more practical claim: completeness is not required for deployment safety. What matters is whether a sufficiently large subset of the model's state space can be interpreted with high reliability, such that the system can be restricted to operate exclusively within that region. Where \(f\) is undefined, unreliable, or uncomputable, transitions into those states are prohibited outright. This constitutes a whitelist approach to safety: the permitted operational domain consists precisely of those states that have been positively validated as interpretable, rather than attempting the intractable task of enumerating all possible dangerous configurations.[web:21][web:27] Interpretability limitations thus become operational constraints on permitted behavior rather than fundamental theoretical barriers.

The framework rests on two core metrics that render this operationalization concrete. **Coverage** (\(C\)) is the fraction of the model's relevant state space—whether defined by activations, circuits, or behavioral trajectories—for which \(f\) produces a meaningful interpretation meeting minimum reliability standards. **Confidence** (\(K\)) is the empirical probability that the interpretation correctly predicts behavioral outcomes under causal intervention, such as ablation studies, activation patching, or representation steering.[web:3][web:7][web:32] In the ideal limit of solved interpretability, confidence would reach 100% across the covered domain, eliminating uncertainty entirely. CPI introduces these graded metrics as engineering tools to track progress toward that limit, converting interpretability from a binary achievement into a measurable quantity that can guide iterative improvement.

These metrics decompose the alignment problem into two separable empirical subproblems. The **confidence problem** asks how to construct \(f\) such that its outputs reliably predict intervention effects, thereby establishing causal fidelity rather than mere behavioral correlation. This is directly testable: propose an interpretation, intervene on the implicated components, and measure prediction accuracy.[web:3][web:44] The **coverage problem** asks how to expand the proportion of the state space where confident interpretations are available. Progress on coverage permits increasingly capable models to operate safely, as the whitelist region grows to accommodate more complex computations. The two problems are independent—one can refine interpretation quality within a fixed domain or extend a method's scope without improving its precision—making coordinated research tractable.

A further hypothesis posits that coverage and confidence exhibit predictable degradation patterns as model complexity increases, potentially yielding an interpretability scaling law analogous to established capability scaling laws.[web:23][web:26] Such a relationship would enable pre-training forecasts of how much of a given architecture remains interpretable, thereby informing safe deployment boundaries before full systems are trained. While empirical evidence remains preliminary, existing mechanistic interpretability work suggests structured challenges—such as feature absorption in sparse autoencoders and representational entanglement in larger models—that may lend themselves to systematic measurement.[web:22][web:28]

The primary objection to CPI is **deceptive alignment**, where a model appears interpretable within observed training distributions but activates misaligned objectives under distribution shift or capability jumps.[web:1][web:5][web:10] This challenge is substantial, as it implies that even perfect interpretability on sampled states may fail in deployment. CPI addresses it through the whitelist mechanism: safety does not require identifying all dangerous states, but only those that are confidently safe relative to the deployment objective. The empirical question becomes whether this safe region remains sufficiently large and robust under realistic perturbations, including self-modification and out-of-distribution inputs—a question the framework is explicitly designed to quantify rather than dismiss.

A second objection concerns **verifier reliability**: the interpreter \(f\) itself may produce systematically incorrect descriptions, confidently measuring a human-legible proxy rather than the model's true causal structure.[web:11][web:14] This is the most philosophically demanding critique, as it questions whether our methods can ever escape ontological mismatch between human concepts and model representations. CPI's response is operational rather than foundational: confidence is defined by intervention success across contexts, creating a calibration check that detects systematic misreading as predictive failure. High \(K\) scores that fail to generalize signal interpreter breakdown, providing an empirical failure mode rather than an untestable skepticism.[web:3][web:7]

**Compositional opacity** forms a third objection: state-level interpretability may fail to capture emergent behaviors arising from state sequences or circuit interactions.[web:2][web:6][web:8] CPI treats this as an extension of the coverage problem, requiring interpretation at the appropriate granularity—whether single activations, trajectories, or higher-order structures—rather than a categorical refutation. Superposition and polysemanticity exacerbate this challenge but do not render it insoluble; they simply demand methods that disentangle overlapping representations.[web:2][web:10]

**Capability loss** constitutes the most practical objection: constraining the model to interpretable states may eliminate precisely the creative or high-performance behaviors that justify deployment.[web:26] Preliminary results on toy modular arithmetic transformers suggest capability retention of 79–91% under coverage constraints, but scaling this to realistic models remains an open empirical question.[web:46] Favorable tradeoffs would validate CPI's viability; catastrophic ones would identify fundamental limits, either outcome advancing the field.

**Ontology mismatch** questions whether the output space \(Y\) aligns with the model's native representational structure.[web:11][web:12] In principle, \(Y\) should capture the full cognitive state: current thoughts, self-model, goals, inclinations, counterfactual responses, and memory salience. For this project's scope, however, CPI focuses narrowly on goal and inclination states, sufficient to monitor alignment with deployment-specified objectives. \(Y\) is constructed iteratively and task-dependently: given concrete objectives \(T\), identify the internal dimensions that causally mediate success or failure on \(T\). This avoids circularity by reverse-engineering from human-defined goals rather than presupposing a universal mental ontology, yielding a minimal interpretable interface tailored to deployment needs.[web:12][web:15]

CPI's claims are deliberately falsifiable. Failure occurs if: (1) confidence fails to predict interventions; (2) coverage cannot expand meaningfully; (3) capability retention collapses under constraints; or (4) task-specific \(Y\) proves unstable or incomplete for safety monitoring. These criteria transform interpretability from aspiration to experiment, positioning CPI as an empirical program for operationalizing partial understanding into runtime safety controls.

## Pre-registration Commitment

This document pre-registers hypotheses, metrics, and success criteria **prior to main experiments**. Results evaluated strictly against these conditions.

**Signed**: Manu Prabakaran, Rice University  
**Date**: March 30, 2026
