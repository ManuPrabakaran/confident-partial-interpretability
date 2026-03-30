"""Confidence K: fraction of intervention predictions that match observed effects.

Aligned with pre-registration_paper.md §6. Predictions and observations can be
scalars (e.g. Δ logit or accuracy drop) or vectors; comparison uses absolute error
and a per-dimension tolerance.
"""

from __future__ import annotations

import numpy as np


def compute_k(
    predicted_delta: np.ndarray,
    observed_delta: np.ndarray,
    *,
    atol: float = 0.1,
) -> float:
    """Return K in [0, 1] for a batch of intervention predictions.

    Each row is one intervention trial. A trial counts as correct if
    ``max_i |pred_i - obs_i| <= atol`` (for 1D, i runs over a single element).

    Parameters
    ----------
    predicted_delta :
        Shape (n_trials, d) or (n_trials,) — predicted change under intervention.
    observed_delta :
        Same shape as ``predicted_delta``.
    atol :
        Maximum elementwise absolute error for a trial to count as correct.
    """
    pred = np.atleast_2d(np.asarray(predicted_delta, dtype=np.float64))
    obs = np.atleast_2d(np.asarray(observed_delta, dtype=np.float64))
    if pred.shape != obs.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {obs.shape}")

    err = np.max(np.abs(pred - obs), axis=1)
    correct = err <= atol
    n = correct.size
    if n == 0:
        return float("nan")
    return float(np.mean(correct))
