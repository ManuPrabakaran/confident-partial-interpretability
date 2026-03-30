"""CPI metrics: confidence (K) and coverage (C). See pre-registration_paper.md §6–7."""

from metrics.confidence import compute_k
from metrics.coverage import compute_c

__all__ = ["compute_k", "compute_c"]
