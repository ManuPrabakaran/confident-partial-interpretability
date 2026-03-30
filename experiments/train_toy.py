#!/usr/bin/env python3
"""
Train TinyGPT on modular addition (Nanda et al., 2023) or induction (Olsson et al., 2022).

Example:
  python experiments/train_toy.py --task modular --steps 4000 --out outputs/checkpoints/modular.pt
  python experiments/train_toy.py --task induction --steps 8000 --out outputs/checkpoints/induction.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from experiments.tasks_toy import InductionConfig, ModularConfig
from experiments.tiny_gpt import TinyGPT, TinyGPTConfig


def loss_batch(
    logits: torch.Tensor, inp: torch.Tensor, tgt: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """CE at per-row positions pos (variable for induction with padding)."""
    B = inp.size(0)
    losses = []
    for i in range(B):
        pi = int(pos[i].item())
        losses.append(F.cross_entropy(logits[i, pi], tgt[i]))
    return torch.stack(losses).mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("modular", "induction"), required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-mlp", type=int, default=256)
    p.add_argument("--prime-p", type=int, default=97, help="Modulus p (Nanda-style; often prime, e.g. 97 or 113).")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "checkpoints" / "toy.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.task == "modular":
        tc = ModularConfig(prime_p=args.prime_p)
        vocab = tc.vocab_size
        max_len = 8
        cite = "Nanda et al. (2023), modular addition / grokking setting"
    else:
        tc = InductionConfig()
        vocab = tc.vocab_size
        max_len = 32
        cite = "Olsson et al. (2022), induction / repeated pattern"

    if args.d_model % args.n_head != 0:
        raise SystemExit(f"--d-model ({args.d_model}) must be divisible by --n-head ({args.n_head})")

    cfg = TinyGPTConfig(
        vocab_size=vocab,
        max_seq_len=max_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
    )
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        inp, tgt, pos = tc.batch(args.batch, device)
        logits = model(inp)
        loss = loss_batch(logits, inp, tgt, pos)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "tiny_config": cfg.__dict__,
            "task": args.task,
            "task_config": tc.__dict__ if args.task == "induction" else {"prime_p": tc.prime_p},
            "citation": cite,
            "steps": args.steps,
            "seed": args.seed,
        },
        args.out,
    )
    print(f"Saved {args.out} ({cite})")


if __name__ == "__main__":
    main()
