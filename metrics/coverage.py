"""Coverage C: fraction of relevant states with per-state K at least τ.

Aligned with pre-registration_paper.md §7: C = (states with K ≥ τ) / (total relevant states).
"""

from __future__ import annotations


def compute_c(per_state_k: list[float] | list[int], *, tau: float) -> float:
    """Return coverage C in [0, 1].

    Parameters
    ----------
    per_state_k :
        Estimated confidence for each relevant state (each in [0, 1], or 0/1).
    tau :
        Policy threshold τ (same units as K).
    """
    if not per_state_k:
        return float("nan")
    ok = sum(1 for k in per_state_k if k >= tau)
    return ok / len(per_state_k)
