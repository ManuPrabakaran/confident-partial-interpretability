#!/usr/bin/env python3
"""Bar chart of K (and C if present) from a CPI JSON artifact. Optional [plots] extra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install '.[plots]'", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    values = []
    if "K_global" in data and data["K_global"] is not None:
        labels.append("K_global")
        values.append(float(data["K_global"]))
    if "C" in data and data["C"] is not None and str(data["C"]) != "nan":
        labels.append("C")
        values.append(float(data["C"]))

    if not labels:
        raise SystemExit("No K_global or C in JSON")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["#2a6f97", "#6bb392"][: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.set_title(args.json_path.stem)
    fig.tight_layout()
    out = args.out or args.json_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
