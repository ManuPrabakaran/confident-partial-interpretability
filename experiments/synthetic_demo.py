#!/usr/bin/env python3
"""Synthetic end-to-end walkthrough for C and K (NOT real model data).

Use this to verify the metric pipeline and to demo structure before PyTorch
experiments land. Numbers are illustrative only; see README Status section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.confidence import compute_k  # noqa: E402
from metrics.coverage import compute_c  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_synthetic(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    n_states = int(cfg["n_relevant_states"])
    n_int = int(cfg["interventions_per_state"])
    atol = float(cfg["k_atol"])
    tau = float(cfg["coverage_tau"])
    acc = float(cfg["synthetic_interpreter_accuracy"])

    per_state_k: list[float] = []
    all_pred: list[np.ndarray] = []
    all_obs: list[np.ndarray] = []

    for _ in range(n_states):
        # Small-magnitude effects so atol is meaningful for a toy sanity check
        true_effects = rng.normal(0.0, 0.35, size=n_int)
        # Interpreter guesses true effect with Bernoulli noise on correctness
        guesses = np.where(
            rng.random(n_int) < acc,
            true_effects + rng.normal(0.0, 0.04, size=n_int),
            true_effects + rng.normal(0.0, 0.45, size=n_int),
        )
        # One row per intervention trial (not one trial with d columns)
        g = guesses.reshape(-1, 1)
        t = true_effects.reshape(-1, 1)
        ki = compute_k(g, t, atol=atol)
        per_state_k.append(ki)
        all_pred.append(g)
        all_obs.append(t)

    pooled_pred = np.vstack(all_pred)
    pooled_obs = np.vstack(all_obs)
    k_global = compute_k(pooled_pred, pooled_obs, atol=atol)
    c = compute_c(per_state_k, tau=tau)

    return {
        "protocol": {
            "n_relevant_states": n_states,
            "interventions_per_state": n_int,
            "k_atol": atol,
            "coverage_tau": tau,
            "synthetic_interpreter_accuracy": acc,
        },
        "K_global": k_global,
        "C": c,
        "per_state_K_mean": float(np.mean(per_state_k)),
        "per_state_K_std": float(np.std(per_state_k)),
        "disclaimer": "synthetic_demo — not empirical CPI measurement",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML config path",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "synthetic_demo.json",
        help="Write JSON summary here",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    result = run_synthetic(cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
