import math

import numpy as np
import pytest

from metrics.confidence import compute_k
from metrics.coverage import compute_c


def test_compute_k_all_correct():
    pred = np.array([[1.0], [2.0], [-0.5]])
    obs = pred.copy()
    assert compute_k(pred, obs, atol=0.01) == 1.0


def test_compute_k_half_wrong():
    # One dimension per trial (column vectors), so three interventions.
    pred = np.array([[1.0], [2.0], [3.0]])
    obs = np.array([[1.0], [9.0], [3.0]])
    assert compute_k(pred, obs, atol=0.1) == pytest.approx(2 / 3)


def test_compute_c():
    assert compute_c([0.9, 0.8, 0.5], tau=0.75) == pytest.approx(2 / 3)


def test_compute_c_empty_is_nan():
    assert math.isnan(compute_c([], tau=0.5))
