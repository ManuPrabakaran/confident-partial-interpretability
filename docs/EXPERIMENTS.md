# CPI experiments: modular addition, induction, and larger models

## Citations (read these in parallel with the code)

1. **Nanda, Chan, Lieberum, Smith, Steinhardt (2023).** *Progress measures for grokking via mechanistic interpretability.* [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) — modular addition, grokking, and circuit-level analysis. Our `experiments/train_toy.py --task modular` implements the **symbolic (a+b) mod p** prediction setting used in that line of work (not a full reproduction of their training budget).

2. **Olsson et al. (2022).** *In-context learning and induction heads.* [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) — induction heads and repeated subsequence structure. Our `experiments/train_toy.py --task induction` is a **minimal** [A,B,…,A]→B task in the same spirit, on a tiny LM.

3. **Elhage et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread — architectural vocabulary (residual stream, attention heads) assumed in the pre-registration paper.

## What runs in this repo

| Script | Purpose |
|--------|---------|
| `experiments/train_toy.py` | Train **TinyGPT** on modular or induction. |
| `experiments/measure_cpi_toy.py` | **K** and **C**: relevance filter ε, bucketed local **K**, frozen **`protocol`** JSON (pre-registration §2.1, §6–7). |
| `experiments/hf_cpi_probe.py` | Same on **HF** GPT-2–class LMs: **C**, optional **`--prompts-file`** for x∼D, **`protocol`** block. |
| `experiments/synthetic_demo.py` | Non-model sanity check for metric code only. |

### Protocol and relevance

- **`--relevance-epsilon`**: only count dimensions where **|observed Δlogit| ≥ ε** under zero-ablation (causally inert axes skipped). `0` disables the filter.
- **`--dims-per-bucket` / `--n-buckets` (HF) or `--trials` (toy)**: build per-bucket **K**, then **C** = fraction of buckets with **K ≥ τ** (`--tau`).
- Every run writes a **`protocol`** object (timestamps, prompts, layer policy, atol, τ, seeds, …) next to **K** / **C** so results are comparable across machines.

Example multi-prompt HF run:

```bash
python experiments/hf_cpi_probe.py --model gpt2 --prompts-file docs/sample_prompts.txt \\
  --n-buckets 12 --dims-per-bucket 5 --relevance-epsilon 0.02 --atol 0.5 --tau 0.6
```

## Ollama vs Hugging Face (for “complex” models)

**Ollama** is ideal for fast local **inference** and demos. It does **not** expose arbitrary residual activations or a stable hook surface for CPI-style interventions through the default HTTP API. Treat outputs from Ollama as **behavioral** only unless you use a stack that exports tensors.

For **inner workings** aligned with CPI (predict Δ under intervention), use:

- **`transformers`** models with **forward hooks** (this repo: `hf_cpi_probe.py`), or  
- **TransformerLens** / **nnsight** on **supported** checkpoints, or  
- Small enough checkpoints that full weights fit on your machine.

To go larger “today”: try `gpt2`, `distilgpt2`, or a small instruction-tuned model on HF that still fits in RAM/VRAM—**not** the same as Ollama’s serving path.

## Scaling up (rough ladder)

1. Stabilize **K** on TinyGPT + modular (Nanda setting) and induction.  
2. Tune **ε**, **atol**, and **τ** on a fixed **`protocol`**; compare JSONs.  
3. Run **hf_cpi_probe** on progressively wider LMs; expect **K** / **C** to move with capacity.  
4. Optional: wire **TransformerLens** for curated interpretability tooling on compatible models.

## Outputs

Checkpoints: `outputs/checkpoints/*.pt`  
Metric JSON: `outputs/cpi_toy_*.json`, `outputs/cpi_hf_*.json`
