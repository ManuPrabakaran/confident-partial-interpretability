"""
Causal interventions (ablation, patching, steering) on model internals.

This module will wrap forward hooks and batch evaluation. See
pre-registration_paper.md §5–6 and hypothesis.md (measurement pipeline).

Not implemented yet.
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
    raise NotImplementedError("Hook-based interventions — implement after toy model lands.")
