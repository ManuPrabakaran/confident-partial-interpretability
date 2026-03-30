"""
Causal interventions (ablation, patching, steering) on model internals.

Concrete ablations for CPI metrics live in:
- ``experiments/measure_cpi_toy.py`` — residual zero-ablation + gradient Δ on TinyGPT.
- ``experiments/hf_cpi_probe.py`` — same pattern on Hugging Face GPT-2–style blocks.

See pre-registration_paper.md §5–6 and hypothesis.md (measurement pipeline).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class ModelAdapter(Protocol):
    """Minimal interface for a model that supports hidden-state reads/writes."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...


def run_intervention_batch(
    model: ModelAdapter,
    *,
    layer: int,
    intervention_fn: Callable[..., Any],
) -> None:
    raise NotImplementedError("Use measure_cpi_toy.py / hf_cpi_probe.py for batched CPI runs.")
