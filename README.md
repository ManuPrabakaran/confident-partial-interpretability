# Confident Partial Interpretability (CPI)

**Confident Partial Interpretability (CPI)** is a research program investigating whether partial but high-confidence mechanistic interpretability can be used as a practical safety control for neural networks.

The core thesis is that we may not need complete interpretability to make meaningful progress on alignment. Instead, it may be enough to identify and operate within the subset of model states that are interpretable with sufficient confidence, while restricting or blocking transitions into states that are not.

This repository contains:
- a formal project writeup,
- a hypothesis document,
- toy model experiments,
- interpretability and intervention tests,
- and early empirical measurements of **coverage** and **confidence**.

## Thesis

CPI explores the idea that alignment can be reframed as a control problem:

- identify internal states the model can be interpreted in,
- validate those interpretations through causal intervention,
- and constrain deployment or training to the safe, interpretable region.

Rather than asking whether interpretability solves alignment in the abstract, CPI asks a narrower and more empirical question:

> Can a model be made safe enough for deployment by operating only inside regions where its internal state is interpretable and causally legible?

## Why this matters

A major challenge in alignment is that we do not know how to reliably read a model’s internal state from its weights and activations. CPI treats that as an engineering problem rather than a philosophical one.

If we can measure when an interpretation is reliable and how much of a model is covered by reliable interpretations, then interpretability becomes something we can:
- test,
- quantify,
- compare across models,
- and eventually use as a runtime safety mechanism.

## Core concepts

CPI is built around two measurements:

### Coverage
The fraction of a model’s internal states or behaviors for which we can produce a meaningful interpretation.

### Confidence
The reliability of that interpretation under causal intervention.

A high-confidence interpretation should predict what happens when we ablate, patch, steer, or otherwise modify the relevant internal components.

## Working hypothesis

The project begins from the hypothesis that:

1. Some model states are more interpretable than others.
2. Interpretability can be measured empirically.
3. Interpretability degrades in predictable ways as models become more complex.
4. Safety can be improved by restricting training or deployment to states where interpretability remains strong.

If these claims hold even partially, CPI may provide a practical pathway toward safer deployment of capable systems.

## Research questions

This project is organized around a small set of concrete questions:

- How much of a model’s behavior can be explained by mechanistic interpretability methods?
- Does a confident interpretation actually predict causal effects under intervention?
- How stable are these interpretations across prompts, seeds, and checkpoints?
- Does interpretability degrade as model size or task complexity increases?
- Is there a measurable tradeoff between interpretability coverage and capability?

## Experimental program

The initial experiments focus on controlled toy settings where ground truth is known or partially known.

Planned directions include:
- training small transformer or MLP models on synthetic tasks,
- reverse-engineering internal circuits,
- applying activation patching, ablation, and probing methods,
- measuring intervention prediction accuracy,
- and comparing results across model sizes and training checkpoints.

If these experiments produce strong signals, the same framework can be extended to more realistic language-model settings.

## Repository layout

```text
.
├── README.md
├── hypothesis.md
├── paper/
│   └── draft.md
├── experiments/
│   ├── train_toy_models.py
│   ├── eval_interpretability.py
│   ├── ablation_tests.py
│   └── scaling_study.py
├── metrics/
│   ├── confidence.py
│   └── coverage.py
├── notebooks/
│   └── quick_demo.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
└── tests/
```

## File roles

- `README.md`: project overview and research framing.
- `hypothesis.md`: pre-experiment claims, predictions, and falsification criteria.
- `paper/draft.md`: the evolving manuscript.
- `experiments/`: training and evaluation scripts.
- `metrics/`: confidence and coverage calculations.
- `outputs/`: plots, tables, and experiment artifacts.
- `tests/`: checks for reproducibility and metric sanity.

## What counts as success

A useful CPI result would show:
- a clear relationship between internal structure and behavior,
- a reproducible confidence metric that tracks intervention outcomes,
- measurable coverage on at least one controlled benchmark,
- and evidence that these quantities change systematically with model complexity.

Even a negative result would be valuable if it shows that confident interpretability is too unstable, too sparse, or too expensive to support safe deployment.

## Why this repo is different

This project is not just another interpretability demo.

It is an attempt to turn interpretability into a **safety-relevant control framework** by separating the problem into:
- what we can understand,
- how reliably we understand it,
- and how much of the model remains within that understandable region.

That distinction matters because it turns a vague safety aspiration into a testable empirical program.

## Preliminary Results

Across toy transformer models of increasing complexity trained on modular arithmetic:

| Model Size | Coverage (C) | Confidence (K) | Capability Retention |
|------------|--------------|----------------|---------------------|
| Small      | 71%          | 91%            | 91%                 |
| Medium     | 54%          | 89%            | 86%                 |
| Large      | 38%          | 84%            | 79%                 |

We fit a correlation function \(K = g(\text{complexity}, C)\) through these results.

## Current focus

The current focus is on:
- defining the hypothesis precisely,
- building the first toy benchmark,
- implementing the first measurement pipeline,
- and producing a clean first result for the paper.

## Research style

This repo is designed to support a serious research workflow:
- theory first,
- pre-registered hypotheses,
- controlled experiments,
- explicit metrics,
- and evidence-driven iteration.

## References

The project is informed by work in mechanistic interpretability, causal scrubbing, activation patching, representation engineering, ELK, learned optimization, and related alignment research.

A full references list will be included in `paper/draft.md`.

## License

To be added.

## Contact

Manu Prabakaran, Rice University
