"""
Toy tasks for CPI experiments.

- Modular addition follows the setting used in mechanistic work on **grokking**
  (Nanda et al., 2023): learn (a + b) mod p from symbolic tokens. See also
  Transformer Circuits (Elhage et al., 2021) for architectural context.

- Induction follows the **induction head** / in-context copy setup analyzed in
  Olsson et al. (2022) and related work; we use a minimal repeated-subsequence
  prediction task suitable for tiny LMs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModularConfig:
    """Nanda et al. (2023)-style modular addition: predict (a+b) mod p."""

    prime_p: int = 97

    @property
    def eq_token(self) -> int:
        return self.prime_p

    @property
    def vocab_size(self) -> int:
        return self.prime_p + 1  # 0..p-1 digits + '='

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """input_ids [B,3], targets c [B], predict_position=2 (after '=')."""
        ab = torch.randint(0, self.prime_p, (batch_size, 2), device=device)
        c = (ab[:, 0] + ab[:, 1]) % self.prime_p
        eq = torch.full((batch_size,), self.eq_token, device=device, dtype=torch.long)
        inp = torch.stack([ab[:, 0], ab[:, 1], eq], dim=1)
        pos = torch.full((batch_size,), 2, device=device, dtype=torch.long)
        return inp, c, pos


@dataclass
class InductionConfig:
    """Minimal induction: [A,B, noise..., A] -> predict B (Olsson-style pattern)."""

    vocab_symbols: int = 16
    prefix_len: int = 6

    @property
    def vocab_size(self) -> int:
        return self.vocab_symbols

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = self.vocab_symbols
        rows: list[torch.Tensor] = []
        targets: list[int] = []
        pred_pos: list[int] = []
        for _ in range(batch_size):
            a = int(torch.randint(0, K, (1,), device=device).item())
            b = int(torch.randint(0, K, (1,), device=device).item())
            while b == a:
                b = int(torch.randint(0, K, (1,), device=device).item())
            n_mid = max(0, self.prefix_len - 3)
            mid = torch.randint(0, K, (n_mid,), device=device).tolist()
            seq = [a, b] + mid + [a]
            rows.append(torch.tensor(seq, dtype=torch.long, device=device))
            targets.append(b)
            pred_pos.append(len(seq) - 1)
        max_t = max(len(r) for r in rows)
        pad = 0
        inp = torch.full((batch_size, max_t), pad, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            inp[i, : len(r)] = r
        tgt = torch.tensor(targets, dtype=torch.long, device=device)
        pos = torch.tensor(pred_pos, dtype=torch.long, device=device)
        return inp, tgt, pos
