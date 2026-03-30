#!/usr/bin/env python3
"""
Estimate CPI-style K (and coarse C) on a trained TinyGPT using residual ablations.

Interpretation for K: first-order attribution predicts the change in the target
logit when residual dimension (ℓ, t, d) is zeroed; we compare to the actual Δ
from a forward ablation. This is a concrete, testable prediction under
intervention (pre-registration_paper.md §6), in the spirit of circuit-level
hypotheses in Nanda et al. (2023) / Transformer Circuits.

Does not claim SOTA mech interp — only a measurement scaffold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from metrics.confidence import compute_k
from metrics.coverage import compute_c
from experiments.tasks_toy import InductionConfig, ModularConfig
from experiments.tiny_gpt import TinyGPT, TinyGPTConfig


def load_model(ckpt: Path, device: torch.device) -> tuple[TinyGPT, dict]:
    try:
        data = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(ckpt, map_location=device)
    cfg = TinyGPTConfig(**data["tiny_config"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, data


def task_from_ckpt(data: dict, device: torch.device) -> ModularConfig | InductionConfig:
    t = data["task"]
    if t == "modular":
        return ModularConfig(prime_p=data["task_config"]["prime_p"])
    return InductionConfig(**data["task_config"])


def scalar_logit(logits: torch.Tensor, b: int, pos: int, target: int) -> torch.Tensor:
    return logits[b, pos, target]


def measure_trial(
    model: TinyGPT,
    inp: torch.Tensor,
    tgt: torch.Tensor,
    pos: torch.Tensor,
    layer: int,
    tpos: int,
    dim: int,
    *,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (predicted_delta, observed_delta) on target logit for batch item 0."""
    b = 0
    tgt0 = int(tgt[b].item())
    p0 = int(pos[b].item())

    model.zero_grad(set_to_none=True)
    logits, resids = model(inp, return_residuals=True)
    for r in resids:
        r.retain_grad()
    logit_star = scalar_logit(logits, b, p0, tgt0)
    logit_star.backward(retain_graph=False)

    h = resids[layer].detach()[b, tpos, dim]
    g = resids[layer].grad[b, tpos, dim] if resids[layer].grad is not None else torch.zeros_like(h)
    pred = float((-h * g).item())

    clean = float(scalar_logit(logits, b, p0, tgt0).detach().item())

    def ablate_hook(_m, _inp, out: torch.Tensor) -> torch.Tensor:
        o = out.clone()
        o[b, tpos, dim] = 0
        return o

    hnd = model.blocks[layer].register_forward_hook(ablate_hook)
    try:
        with torch.no_grad():
            logits2 = model(inp)
    finally:
        hnd.remove()

    ablated = float(scalar_logit(logits2, b, p0, tgt0).item())
    obs = ablated - clean
    return pred, obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--dims-per-bucket", type=int, default=5, help="Relevant dims per (layer,pos) bucket.")
    ap.add_argument(
        "--max-probes-per-bucket",
        type=int,
        default=300,
        help="Max random dim draws per bucket (relevance filter may skip many).",
    )
    ap.add_argument(
        "--relevance-epsilon",
        type=float,
        default=0.0,
        help="|observed Δlogit| ≥ ε to count as causally relevant (0 = off).",
    )
    ap.add_argument("--atol", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, meta = load_model(args.ckpt, device)
    tc = task_from_ckpt(meta, device)
    n_layer = len(model.blocks)
    d_model = model.cfg.d_model

    preds: list[float] = []
    obses: list[float] = []
    per_bucket_k: list[float] = []
    n_irrelevant = 0
    n_incomplete_buckets = 0

    for _t in range(args.trials):
        inp, tgt, pos = tc.batch(1, device)
        layer = int(torch.randint(0, n_layer, (1,), device=device).item())
        T = inp.size(1)
        tpos = int(torch.randint(0, T, (1,), device=device).item())
        bucket_pred: list[float] = []
        bucket_obs: list[float] = []
        probes = 0
        while len(bucket_pred) < args.dims_per_bucket and probes < args.max_probes_per_bucket:
            probes += 1
            dim = int(torch.randint(0, d_model, (1,), device=device).item())
            pr, ob = measure_trial(model, inp, tgt, pos, layer, tpos, dim, device=device)
            if args.relevance_epsilon > 0 and abs(ob) < args.relevance_epsilon:
                n_irrelevant += 1
                continue
            bucket_pred.append(pr)
            bucket_obs.append(ob)
        if len(bucket_pred) < args.dims_per_bucket:
            n_incomplete_buckets += 1
            continue
        preds.extend(bucket_pred)
        obses.extend(bucket_obs)
        bk = compute_k(
            np.array(bucket_pred, dtype=np.float64).reshape(-1, 1),
            np.array(bucket_obs, dtype=np.float64).reshape(-1, 1),
            atol=args.atol,
        )
        if not np.isnan(bk):
            per_bucket_k.append(float(bk))

    k_global = compute_k(
        np.array(preds, dtype=np.float64).reshape(-1, 1),
        np.array(obses, dtype=np.float64).reshape(-1, 1),
        atol=args.atol,
    )
    c_cov = compute_c(per_bucket_k, tau=args.tau) if per_bucket_k else float("nan")

    protocol = {
        "script": "measure_cpi_toy.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.ckpt),
        "task": meta.get("task"),
        "trials": args.trials,
        "dims_per_bucket": args.dims_per_bucket,
        "max_probes_per_bucket": args.max_probes_per_bucket,
        "relevance_epsilon": args.relevance_epsilon,
        "atol": args.atol,
        "tau": args.tau,
        "seed": args.seed,
        "device": str(device),
        "sampling": "(layer, token_pos) uniform; x from task.batch(); dim uniform until bucket full",
        "target_logit": "correct-class logit at task prediction position",
    }

    out = {
        "K_global": k_global,
        "C": c_cov,
        "n_relevant_trials": len(preds),
        "n_irrelevant_skipped": n_irrelevant,
        "n_complete_buckets": len(per_bucket_k),
        "n_incomplete_buckets": n_incomplete_buckets,
        "per_bucket_K": per_bucket_k,
        "protocol": protocol,
        "checkpoint": str(args.ckpt),
        "task": meta.get("task"),
        "citation_context": meta.get("citation"),
        "references": [
            "Nanda, Chan, Lieberum, Smith, Steinhardt (2023). Progress measures for grokking via mechanistic interpretability. arXiv:2301.05217",
            "Olsson et al. (2022). In-context learning and induction heads. arXiv:2209.11895",
            "Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread.",
            "pre-registration_paper.md §2.1 relevance, §6–7",
        ],
        "disclaimer": "Toy model + first-order attribution vs ablation; compare runs via identical protocol blocks.",
    }

    print(json.dumps(out, indent=2))
    out_path = args.out or (ROOT / "outputs" / f"cpi_toy_{meta.get('task', 'unk')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
