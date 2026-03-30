#!/usr/bin/env python3
"""
Sweep Nanda-style modular TinyGPT sizes: train → measure CPI (strict protocol) → JSON + plot.

Trains 2-layer, 4-head models with d_mlp = 2 * d_model across a grid of d_model values,
then runs `measure_cpi_toy.py` with fixed relevance / ablation / atol / tau so points are
comparable on one curve: K_global, C, and accuracy vs log10(n_params).

Writes three standalone PNGs plus a 1×3 triptych. More points: use the default
`--d-models` grid or pass a custom list (each value divisible by 4).

Example:
  python experiments/modular_scaling_sweep.py --steps 4500 --seed 0
  python experiments/modular_scaling_sweep.py --plot-only   # redraw from JSON only
  python experiments/modular_scaling_sweep.py --plot-only --c-full-scale   # C on [0,1]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.measure_cpi_toy import load_model, task_from_ckpt

# d_model must be divisible by n_head=4; d_mlp = 2 * d_model (standard width scaling).
# Dense grid: 16…128 step 4 → 29 points (smoother curves; full sweep is ~29× train+measure).
DEFAULT_D_MODELS = tuple(range(16, 129, 4))


@torch.no_grad()
def eval_modular_accuracy(
    ckpt: Path,
    device: torch.device,
    *,
    n_batches: int,
    batch_size: int,
    seed: int,
) -> float:
    """Fraction correct on modular task: predict (a+b) mod p at '=' position (see tasks_toy.ModularConfig)."""
    model, data = load_model(ckpt, device)
    task = task_from_ckpt(data, device)
    torch.manual_seed(seed)
    correct = 0
    total = 0
    for _ in range(n_batches):
        inp, tgt, pos = task.batch(batch_size, device)
        logits = model(inp)
        b_idx = torch.arange(inp.size(0), device=device)
        pred = logits[b_idx, pos, :].argmax(dim=-1)
        correct += int((pred == tgt).sum().item())
        total += int(inp.size(0))
    return correct / max(total, 1)


def _k_ylim_top(ks: list[float], k_ylim_max: float) -> float:
    k_max_data = max(ks) if ks else 0.0
    k_top = float(k_ylim_max)
    if k_max_data > k_top + 1e-9:
        k_top = min(1.0, k_max_data * 1.08)
        print(
            f"Note: max K_global={k_max_data:.3f} exceeds --k-ylim-max; K panel y-axis extended to {k_top:.3f}",
            file=sys.stderr,
        )
    return k_top


def _c_ylim_bounds(cs: list[float], *, full_scale: bool) -> tuple[float, float]:
    """Y limits for C so low values are not squashed at the bottom of [0, 1]."""
    if full_scale:
        return (-0.05, 1.05)
    c_max = max(cs) if cs else 0.0
    if c_max < 1e-9:
        return (-0.01, 0.1)
    pad = max(0.02, c_max * 0.14)
    top = min(1.0, c_max + pad)
    bottom = -0.05 * top
    return (bottom, top)


def _plot_modular_scaling(
    rows: list[dict], out_dir: Path, k_ylim_max: float, *, c_full_scale: bool
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot (pip install -e '.[plots]')", file=sys.stderr)
        return

    rows = sorted(rows, key=lambda r: float(r["log10_n_params"]))
    xs = [float(r["log10_n_params"]) for r in rows]
    ks = [float(r["K_global"]) for r in rows]
    cs = [float(r["C"]) if r.get("C") == r.get("C") else 0.0 for r in rows]
    acc_pct = [100.0 * float(r["accuracy"]) for r in rows]
    labels = [f"dm{r['d_model']}" for r in rows]
    annotate = len(rows) <= 12
    k_top = _k_ylim_top(ks, k_ylim_max)
    c_lo, c_hi = _c_ylim_bounds(cs, full_scale=c_full_scale)
    x_label = r"$\log_{10}$(# parameters)"
    subtitle = "Modular TinyGPT (fixed strict CPI protocol)"

    def _annotate_k(ax, xs_: list[float], ks_: list[float], labs: list[str]) -> None:
        if not annotate:
            return
        for x, k, lab in zip(xs_, ks_, labs):
            ax.annotate(lab, (x, k), textcoords="offset points", xytext=(3, 3), fontsize=6)

    out_dir.mkdir(parents=True, exist_ok=True)
    readme_fig_dir = ROOT / "docs" / "figures"
    readme_fig_dir.mkdir(parents=True, exist_ok=True)

    def _save_twin(fig, path_out: Path, path_readme: Path) -> None:
        fig.savefig(path_out, dpi=150)
        fig.savefig(path_readme, dpi=150)

    # --- Panel 1: K_global ---
    ms = 4 if len(rows) > 18 else 5
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.6))
    ax1.plot(xs, ks, "o-", color="#2a6f97", linewidth=1.4, markersize=ms, label=r"$K_{\mathrm{global}}$")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"$K_{\mathrm{global}}$")
    ax1.set_ylim(0.0, k_top)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(subtitle + "\n" + r"$K_{\mathrm{global}}$ vs model size")
    _annotate_k(ax1, xs, ks, labels)
    fig1.tight_layout()
    p1 = out_dir / "modular_k_vs_params.png"
    _save_twin(fig1, p1, readme_fig_dir / "modular_k_vs_params.png")
    plt.close(fig1)
    print(f"Wrote {p1}")

    # --- Panel 2: C ---
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.6))
    ax2.plot(xs, cs, "s--", color="#6bb392", linewidth=1.3, markersize=ms, label=r"$C$")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(r"$C$ (coverage)")
    ax2.set_ylim(c_lo, c_hi)
    ax2.grid(True, alpha=0.25)
    ax2.set_title(f"{subtitle}\n$C$ vs model size")
    if annotate:
        for x, c, lab in zip(xs, cs, labels):
            ax2.annotate(lab, (x, c), textcoords="offset points", xytext=(3, 3), fontsize=6)
    fig2.tight_layout()
    p2 = out_dir / "modular_c_vs_params.png"
    _save_twin(fig2, p2, readme_fig_dir / "modular_c_vs_params.png")
    plt.close(fig2)
    print(f"Wrote {p2}")

    # --- Panel 3: Accuracy ---
    fig3, ax3 = plt.subplots(figsize=(5.2, 3.6))
    ax3.plot(xs, acc_pct, "^-", color="#c97c5d", linewidth=1.3, markersize=ms, label="Accuracy")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(-5.0, 105.0)
    ax3.grid(True, alpha=0.25)
    ax3.set_title(f"{subtitle}\nTask accuracy vs model size")
    if annotate:
        for x, a, lab in zip(xs, acc_pct, labels):
            ax3.annotate(lab, (x, a), textcoords="offset points", xytext=(3, 3), fontsize=6)
    fig3.tight_layout()
    p3 = out_dir / "modular_accuracy_vs_params.png"
    _save_twin(fig3, p3, readme_fig_dir / "modular_accuracy_vs_params.png")
    plt.close(fig3)
    print(f"Wrote {p3}")

    # --- Triptych (optional overview) ---
    tms = 3 if len(rows) > 18 else 4
    fig_t, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), sharex=True)
    axes[0].plot(xs, ks, "o-", color="#2a6f97", linewidth=1.2, markersize=tms)
    axes[0].set_ylabel(r"$K_{\mathrm{global}}$")
    axes[0].set_ylim(0.0, k_top)
    axes[0].set_title(r"$K_{\mathrm{global}}$")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(xs, cs, "s--", color="#6bb392", linewidth=1.2, markersize=tms)
    axes[1].set_ylabel(r"$C$")
    axes[1].set_ylim(c_lo, c_hi)
    axes[1].set_title(r"$C$")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(xs, acc_pct, "^-", color="#c97c5d", linewidth=1.2, markersize=tms)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(-5.0, 105.0)
    axes[2].set_title("Accuracy")
    axes[2].grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xlabel(x_label)
    fig_t.suptitle(subtitle, fontsize=10, y=1.02)
    fig_t.tight_layout()
    pt = out_dir / "modular_scaling_triptych.png"
    fig_t.savefig(pt, dpi=150, bbox_inches="tight")
    pt_readme = readme_fig_dir / "modular_scaling_triptych.png"
    fig_t.savefig(pt_readme, dpi=150, bbox_inches="tight")
    plt.close(fig_t)
    print(f"Wrote {pt}")
    print(f"Wrote {pt_readme}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4500, help="Training steps per checkpoint")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=list(DEFAULT_D_MODELS),
        help="d_model values (each must be divisible by 4 for n_head=4)",
    )
    ap.add_argument("--skip-train", action="store_true", help="Only measure if checkpoint exists")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dims-per-bucket", type=int, default=5)
    ap.add_argument("--max-probes-per-bucket", type=int, default=300)
    ap.add_argument("--relevance-epsilon", type=float, default=0.02)
    ap.add_argument("--ablation-coeff", type=float, default=-5.0)
    ap.add_argument("--atol", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "modular_scaling")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Load modular_scaling_summary.json, recompute accuracy, redraw figure (no train/measure)",
    )
    ap.add_argument("--acc-batches", type=int, default=32, help="Batches for modular accuracy eval")
    ap.add_argument("--acc-batch-size", type=int, default=256)
    ap.add_argument(
        "--k-ylim-max",
        type=float,
        default=0.4,
        help="Upper limit for left y-axis (K_global). Data above this clips unless auto-expand triggers",
    )
    ap.add_argument(
        "--c-full-scale",
        action="store_true",
        help="Use full [0, 1] y-axis on C plots; default zooms to data so low C is visible",
    )
    args = ap.parse_args()

    py = sys.executable
    ckpt_dir = args.out_dir / "checkpoints"
    json_dir = args.out_dir / "cpi"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.plot_only:
        summary_path = args.out_dir / "modular_scaling_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"Missing {summary_path}; run without --plot-only first")
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        rows = summary["points"]
        for r in rows:
            ckpt = Path(r["checkpoint"])
            r["accuracy"] = eval_modular_accuracy(
                ckpt,
                device,
                n_batches=args.acc_batches,
                batch_size=args.acc_batch_size,
                seed=args.seed,
            )
        summary["points"] = rows
        summary["accuracy_eval"] = {
            "n_batches": args.acc_batches,
            "batch_size": args.acc_batch_size,
            "seed": args.seed,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated accuracy in {summary_path}")
        if not args.no_plot:
            _plot_modular_scaling(rows, args.out_dir, args.k_ylim_max, c_full_scale=args.c_full_scale)
        return

    rows: list[dict] = []

    for d_model in args.d_models:
        if d_model % 4 != 0:
            raise SystemExit(f"d_model={d_model} not divisible by n_head=4")
        d_mlp = 2 * d_model
        tag = f"dm{d_model}"
        ckpt = ckpt_dir / f"modular_{tag}.pt"
        jpath = json_dir / f"cpi_modular_{tag}.json"

        if not args.skip_train or not ckpt.exists():
            subprocess.run(
                [
                    py,
                    str(ROOT / "experiments" / "train_toy.py"),
                    "--task",
                    "modular",
                    "--steps",
                    str(args.steps),
                    "--batch",
                    str(args.batch),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--d-model",
                    str(d_model),
                    "--d-mlp",
                    str(d_mlp),
                    "--out",
                    str(ckpt),
                ],
                cwd=str(ROOT),
                check=True,
            )

        subprocess.run(
            [
                py,
                str(ROOT / "experiments" / "measure_cpi_toy.py"),
                "--ckpt",
                str(ckpt),
                "--trials",
                str(args.trials),
                "--dims-per-bucket",
                str(args.dims_per_bucket),
                "--max-probes-per-bucket",
                str(args.max_probes_per_bucket),
                "--relevance-epsilon",
                str(args.relevance_epsilon),
                "--ablation-coeff",
                str(args.ablation_coeff),
                "--atol",
                str(args.atol),
                "--tau",
                str(args.tau),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--out",
                str(jpath),
            ],
            cwd=str(ROOT),
            check=True,
        )

        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        proto = data.get("protocol", {}) or {}
        n_params = int(proto.get("n_params", 0))
        acc = eval_modular_accuracy(
            ckpt,
            device,
            n_batches=args.acc_batches,
            batch_size=args.acc_batch_size,
            seed=args.seed,
        )
        rows.append(
            {
                "d_model": d_model,
                "d_mlp": d_mlp,
                "n_layer": 2,
                "n_head": 4,
                "n_params": n_params,
                "log10_n_params": math.log10(n_params) if n_params > 0 else float("nan"),
                "K_global": float(data.get("K_global", float("nan"))),
                "C": float(data.get("C", float("nan"))) if data.get("C") == data.get("C") else float("nan"),
                "accuracy": acc,
                "checkpoint": str(ckpt),
                "cpi_json": str(jpath),
            }
        )

    rows.sort(key=lambda r: int(r["n_params"]))

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "modular",
        "citation": "Nanda et al. (2023) modular addition / grokking-style TinyGPT sweep",
        "train_steps": args.steps,
        "measure": {
            "trials": args.trials,
            "dims_per_bucket": args.dims_per_bucket,
            "max_probes_per_bucket": args.max_probes_per_bucket,
            "relevance_epsilon": args.relevance_epsilon,
            "ablation_coeff": args.ablation_coeff,
            "atol": args.atol,
            "tau": args.tau,
            "seed": args.seed,
        },
        "accuracy_eval": {
            "n_batches": args.acc_batches,
            "batch_size": args.acc_batch_size,
            "seed": args.seed,
        },
        "points": rows,
    }

    summary_path = args.out_dir / "modular_scaling_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    if not args.no_plot:
        _plot_modular_scaling(rows, args.out_dir, args.k_ylim_max, c_full_scale=args.c_full_scale)


if __name__ == "__main__":
    main()
