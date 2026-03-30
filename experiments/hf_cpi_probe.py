#!/usr/bin/env python3
"""
CPI-style K and C on a Hugging Face causal LM (GPT-2–class stacks).

- **Relevance** (pre-registration §2.1, §7.1): only count dimensions where
  |Δ logit| ≥ ε under zero-ablation (otherwise the axis is treated as
  causally inert for this scalar task).
- **C**: per-bucket K on ``dims_per_bucket`` *relevant* dimensions at one
  (layer, position, prompt) draw, then fraction of buckets with K ≥ τ.

Why not Ollama: no residual access via the default API. Use HF weights + hooks.

Example:
  pip install '.[experiments]'
  python experiments/hf_cpi_probe.py --model gpt2 --n-buckets 15 --dims-per-bucket 5 \\
    --relevance-epsilon 0.02 --atol 0.5 --tau 0.6
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics.confidence import compute_k
from metrics.coverage import compute_c


def _get_blocks(model: torch.nn.Module):
    inner = getattr(model, "transformer", None) or getattr(model, "model", None)
    blocks = getattr(inner, "h", None) or getattr(inner, "layers", None)
    if blocks is None:
        raise SystemExit(
            "Expected GPT-2-like stack (transformer.h). "
            "For other architectures, extend hook paths or use nnsight."
        )
    return blocks


def measure_dim_ablation(
    model: torch.nn.Module,
    blocks: torch.nn.ModuleList,
    layer: int,
    inputs: dict,
    pos: int,
    next_id: int,
    dim: int,
    ablation_coeff: float,
    device_t: torch.device,
) -> tuple[float, float]:
    """First-order pred Δ and observed Δ on logit[0, pos, next_id]."""
    saved: dict[str, torch.Tensor] = {}

    def capture_hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t.retain_grad()
        saved["h"] = t
        return out

    h = blocks[layer].register_forward_hook(capture_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(**inputs)
        logits = out.logits
        logit_star = logits[0, pos, next_id]
        logit_star.backward()
    finally:
        h.remove()

    H = saved.get("h")
    if H is None or H.grad is None:
        return float("nan"), float("nan")

    hvd = H.detach()[0, pos, dim]
    gvd = H.grad[0, pos, dim]
    # Intervention: set h_dim -> ablation_coeff * h_dim.
    # Activation-space delta is (h' - h) = (ablation_coeff - 1) * h.
    delta_h = (float(ablation_coeff) - 1.0) * hvd
    pred = float((delta_h * gvd).item())
    clean = float(logits[0, pos, next_id].detach().item())

    def ablate_hook(_m, _inp, out):
        if isinstance(out, tuple):
            o0 = out[0].clone()
            o0[0, pos, dim] = float(ablation_coeff) * o0[0, pos, dim]
            return (o0,) + tuple(out[1:])
        o = out.clone()
        o[0, pos, dim] = float(ablation_coeff) * o[0, pos, dim]
        return o

    h2 = blocks[layer].register_forward_hook(ablate_hook)
    try:
        with torch.no_grad():
            out2 = model(**inputs)
            ablated = float(out2.logits[0, pos, next_id].item())
    finally:
        h2.remove()

    obs = ablated - clean
    return pred, obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Single prompt if no --prompts-file (default input distribution is this one string).",
    )
    ap.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Newline-separated prompts; each bucket samples one line (x ~ empirical D).",
    )
    ap.add_argument("--layer", type=int, default=None, help="Block index; default = n_layer//2.")
    ap.add_argument(
        "--position",
        type=int,
        default=None,
        help="Token index for intervention; default = last token (T-1).",
    )
    ap.add_argument("--n-buckets", type=int, default=20, help="Number of (prompt,layer,pos) buckets for C.")
    ap.add_argument("--dims-per-bucket", type=int, default=5, help="Relevant dims per bucket before local K.")
    ap.add_argument(
        "--max-probes-per-bucket",
        type=int,
        default=400,
        help="Max random dim probes per bucket (stop early when bucket full).",
    )
    ap.add_argument(
        "--relevance-epsilon",
        type=float,
        default=0.0,
        help="|observed Δlogit| must be ≥ this to count as causally relevant (0 = disable filter).",
    )
    ap.add_argument(
        "--ablation-coeff",
        type=float,
        default=0.0,
        help="Set residual dimension to coeff * activation (0.0 = zero-ablation; -1.0 = flip sign).",
    )
    ap.add_argument("--atol", type=float, default=1.0, help="Tolerance for K (pred vs obs).")
    ap.add_argument("--tau", type=float, default=0.5, help="Coverage threshold τ on per-bucket K.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cpu_rng = torch.Generator().manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    if args.prompts_file is not None:
        text = args.prompts_file.read_text(encoding="utf-8")
        prompts = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not prompts:
            raise SystemExit("--prompts-file is empty")
    else:
        prompts = [args.prompt]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device_t)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    blocks = _get_blocks(model)
    n_layer = len(blocks)
    d_model = int(model.config.hidden_size)

    all_preds: list[float] = []
    all_obs: list[float] = []
    per_bucket_k: list[float] = []
    n_irrelevant = 0
    n_incomplete_buckets = 0

    for _ in range(args.n_buckets):
        pi = int(torch.randint(0, len(prompts), (1,), generator=cpu_rng).item())
        prompt = prompts[pi]
        inputs = tok(prompt, return_tensors="pt").to(device_t)
        input_ids = inputs["input_ids"]
        _, T = input_ids.shape
        pos = args.position if args.position is not None else T - 1
        pos = max(0, min(pos, T - 1))

        if args.layer is not None:
            layer = args.layer
        elif n_layer > 1:
            layer = int(torch.randint(0, n_layer, (1,), generator=cpu_rng).item())
        else:
            layer = 0

        with torch.no_grad():
            base_out = model(**inputs)
            next_id = int(base_out.logits[0, pos, :].argmax().item())

        b_preds: list[float] = []
        b_obs: list[float] = []
        probes = 0
        while len(b_preds) < args.dims_per_bucket and probes < args.max_probes_per_bucket:
            probes += 1
            dim = int(torch.randint(0, d_model, (1,), device=device_t).item())
            pred, obs = measure_dim_ablation(
                model,
                blocks,
                layer,
                inputs,
                pos,
                next_id,
                dim,
                ablation_coeff=args.ablation_coeff,
                device_t=device_t,
            )
            if pred != pred or obs != obs:
                continue
            if args.relevance_epsilon > 0 and abs(obs) < args.relevance_epsilon:
                n_irrelevant += 1
                continue
            b_preds.append(pred)
            b_obs.append(obs)

        if len(b_preds) < args.dims_per_bucket:
            n_incomplete_buckets += 1
            continue

        all_preds.extend(b_preds)
        all_obs.extend(b_obs)

        bk = compute_k(
            np.array(b_preds, dtype=np.float64).reshape(-1, 1),
            np.array(b_obs, dtype=np.float64).reshape(-1, 1),
            atol=args.atol,
        )
        if not np.isnan(bk):
            per_bucket_k.append(float(bk))

    k_global = (
        compute_k(
            np.array(all_preds, dtype=np.float64).reshape(-1, 1),
            np.array(all_obs, dtype=np.float64).reshape(-1, 1),
            atol=args.atol,
        )
        if all_preds
        else float("nan")
    )
    c_cov = compute_c(per_bucket_k, tau=args.tau) if per_bucket_k else float("nan")

    protocol = {
        "script": "hf_cpi_probe.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "prompts": prompts,
        "prompts_source": str(args.prompts_file) if args.prompts_file else "single --prompt",
        "layer_policy": (
            f"fixed_layer_{args.layer}" if args.layer is not None else "uniform_random_per_bucket"
        ),
        "layer_fixed": args.layer,
        "position_policy": "fixed --position" if args.position is not None else "last_token_T_minus_1",
        "position_fixed": args.position,
        "target_logit": "argmax_next_token_at_position",
        "n_buckets_requested": args.n_buckets,
        "dims_per_bucket": args.dims_per_bucket,
        "max_probes_per_bucket": args.max_probes_per_bucket,
        "relevance_epsilon": args.relevance_epsilon,
        "ablation_coeff": args.ablation_coeff,
        "atol": args.atol,
        "tau": args.tau,
        "seed": args.seed,
        "device": str(device_t),
        "d_model": d_model,
        "n_layer": n_layer,
    }

    report = {
        "K_global": k_global,
        "C": c_cov,
        "n_relevant_trials": len(all_preds),
        "n_irrelevant_skipped": n_irrelevant,
        "n_complete_buckets": len(per_bucket_k),
        "n_incomplete_buckets": n_incomplete_buckets,
        "per_bucket_K": per_bucket_k,
        "protocol": protocol,
        "note": "HF weights + hooks; not Ollama. K/C depend on protocol — compare runs with identical JSON protocol.",
        "references": [
            "Nanda et al. (2023) arXiv:2301.05217",
            "pre-registration_paper.md §2.1 relevance, §6–7 K and C",
        ],
    }

    print(json.dumps(report, indent=2))
    outp = args.out or (ROOT / "outputs" / f"cpi_hf_{args.model.replace('/', '_')}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
