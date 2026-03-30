#!/usr/bin/env python3
"""
Create ONE CPI summary graph + print FOUR headline statistics.

Graph:
  - bar chart of `per_bucket_K` across buckets
  - horizontal line at `tau` (coverage threshold), when present

Stats printed to stdout:
  1) K_global
  2) C (coverage)
  3) std(per_bucket_K)  (stability across buckets)
  4) relevance_pass_rate = n_relevant_trials / (n_relevant_trials + n_irrelevant_skipped)

Expected input JSON schema:
  - produced by `experiments/hf_cpi_probe.py` (preferred) or toy script with compatible keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Install matplotlib + numpy plots deps: pip install -e '.[plots]'", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path, help="CPI JSON artifact")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    per_bucket_k = data.get("per_bucket_K", None)
    if not per_bucket_k:
        raise SystemExit(
            "JSON missing non-empty `per_bucket_K`. "
            "Run `hf_cpi_probe.py` (or update the script schema) to get per-bucket K."
        )

    k_global = float(data.get("K_global", float("nan")))
    c_cov = data.get("C", float("nan"))
    c_cov = float(c_cov) if c_cov == c_cov else float("nan")  # NaN-safe cast

    import numpy as _np

    per_bucket_k_arr = _np.array(per_bucket_k, dtype=_np.float64)
    per_bucket_k_std = float(_np.std(per_bucket_k_arr))

    n_rel = int(data.get("n_relevant_trials", 0))
    n_irrel = int(data.get("n_irrelevant_skipped", 0))
    denom = n_rel + n_irrel
    relevance_pass_rate = float(n_rel / denom) if denom > 0 else float("nan")

    # Optional tau line
    tau = None
    protocol = data.get("protocol", {}) or {}
    if isinstance(protocol, dict) and "tau" in protocol:
        try:
            tau = float(protocol["tau"])
        except Exception:
            tau = None

    # One graph
    labels = list(range(1, len(per_bucket_k_arr) + 1))
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.bar(labels, per_bucket_k_arr, color="#2a6f97", alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("bucket index")
    ax.set_ylabel("per-bucket K")
    ax.set_title(args.json_path.stem)

    if tau is not None and tau == tau:  # not NaN
        ax.axhline(tau, color="#6bb392", linestyle="--", linewidth=1.5, label=f"tau={tau:.3g}")
        ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out = args.out or args.json_path.with_suffix(".png")
    fig.savefig(out, dpi=150)

    # Print FOUR headline stats
    print("Headline stats (CPI):")
    print(f"1) K_global = {k_global:.6g}")
    print(f"2) C = {c_cov:.6g}")
    print(f"3) std(per_bucket_K) = {per_bucket_k_std:.6g}")
    print(f"4) relevance_pass_rate = {relevance_pass_rate:.6g}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

