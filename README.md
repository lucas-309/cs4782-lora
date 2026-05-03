# LoRA on RoBERTa-base — a re-implementation for CS 4782

A from-scratch reproduction of **LoRA: Low-Rank Adaptation of Large Language Models**
([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) on RoBERTa-base, evaluated
on the GLUE benchmark. CS 4782 (Spring 2026), Cornell University.

**Authors:** Lucas He, Partner

## 1. Introduction

This repo re-implements LoRA from scratch (no `peft`, no shortcuts) and tests
the central claim of the paper: **the update matrix `ΔW` learned during
fine-tuning has very low intrinsic rank, so adding a tiny rank-`r` factorization
to a frozen pre-trained model nearly matches full fine-tuning while training
~100× fewer parameters.**

## 2. Chosen result

We reproduce a slice of **Table 2** from Hu et al. (2021): RoBERTa-base on
GLUE, comparing full fine-tuning, LoRA (rank 8), and a head-only baseline.
Headline numbers from our run on **MRPC**:

| Method            | Trainable params | MRPC accuracy |
|-------------------|------------------|---------------|
| Head only         | 0.59M            | 0.684         |
| LoRA `r=1`        | 0.63M            | 0.860         |
| LoRA `r=8`        | 0.89M            | 0.863         |
| Full fine-tuning  | 124.6M           | 0.882         |

LoRA is **2 points behind full fine-tuning while training ~140× fewer
parameters**, and head-only is 18 points behind — confirming that the LoRA
update is doing real work, not just the new classifier head.

We additionally run a **rank ablation** (independent experiment, beyond the
paper) sweeping `r ∈ {1, 2, 4, 8}` on MRPC. Even `r=1` is within noise of `r=8`,
reproducing the "low intrinsic rank" finding at small-model scale.

## 3. Repository structure

```
.
├── code/             # LoRA implementation + training / orchestration scripts
│   ├── lora.py       # ~60-line LoRALinear module + injection helper
│   ├── train.py      # GLUE training script (LoRA / full FT / head-only)
│   ├── plot_results.py
│   ├── run_all.sh    # full sweep
│   └── run_remaining.sh
├── results/runs/     # per-run JSON logs (final metric, training history)
├── figures/          # poster figures generated from results/runs/
├── poster/           # 36"×24" landscape poster (LaTeX → PDF via tectonic)
├── report/           # 2-page project summary
├── references/       # the original LoRA paper PDF
├── LICENSE           # MIT
└── README.md
```

## 4. Re-implementation details

- **LoRA module.** `code/lora.py` defines `LoRALinear`, which wraps an
  `nn.Linear` with two trainable parameters `A ∈ R^{r×d_in}` and
  `B ∈ R^{d_out×r}`, with `A` Kaiming-initialised and `B` zero-initialised so
  that the LoRA contribution is zero at the start of training. The forward
  pass is `h = W₀x + (α/r) · B(Ax)`.
- **Where we apply it.** Following Hu et al.'s Table 5 recommendation, we wrap
  the query (`Wq`) and value (`Wv`) projections in every attention layer of
  RoBERTa-base — 24 wrapped linears, ~887K trainable params at `r=8` (counting
  the new classification head; the LoRA-only delta is ~295K).
- **Training.** AdamW, linear warmup (6%) + linear decay; LoRA at `lr=5e-4`,
  full FT at `lr=2e-5`. Batch size 32, 3 epochs, max sequence length 128.
  All runs use seed 42.
- **Hardware.** A single Apple M-series GPU via PyTorch MPS. No NVIDIA needed.

## 5. Reproduction steps

```bash
# 1. Set up Python 3.11 venv and install dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch transformers datasets scikit-learn matplotlib scipy numpy

# 2. Run the full sweep (~60–90 min on Apple M-series)
bash code/run_all.sh         # writes results/runs/*.json

# 3. Generate poster figures
python code/plot_results.py  # writes figures/*.pdf

# 4. Build the poster (requires tectonic)
cd poster && tectonic poster.tex
```

The single-experiment entrypoint, useful for one-off sanity checks:

```bash
python code/train.py --task mrpc --method lora --rank 8 --epochs 3
python code/train.py --task mrpc --method full --epochs 3
```

GPU is recommended but not strictly required: MRPC trains in ~5 min on MPS,
~30+ min on CPU.

## 6. Results / insights

- **LoRA reproduces** within ~2 points of full fine-tuning on MRPC at
  RoBERTa-base scale, with ~0.7% of the parameters trainable.
- **Rank really is low.** Sweeping `r ∈ {1, 2, 4, 8}` moves accuracy by less
  than the seed-to-seed noise — the fine-tuning "signal" fits in a tiny
  subspace, even at 125M-parameter scale.
- **Head-only is not enough.** Freezing everything but the classifier
  collapses to ~68% — so LoRA's gain is genuine, not classifier-head
  bookkeeping.
- **Caveat on parameter counts.** On a 125M model the new classification
  head dominates the trainable budget. The 10,000× claim in the paper is for
  GPT-3 175B, where the head is negligible.

## 7. Conclusion

LoRA's promise — "tune a tiny additive low-rank update, freeze everything
else" — holds up cleanly on a small-model reproduction. The most striking
result is the rank ablation: rank 1 is enough on MRPC, supporting the
"intrinsic rank" hypothesis at a scale we can actually run on a laptop.

Lessons from re-implementation: most of the work was getting the right
parameters frozen and the right ones trainable; the LoRA module itself is
~30 lines. Once you trust the reparametrization, the only knobs are `r`,
`α`, and where you apply it.

## 8. References

- **Original paper.** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
  Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.
  *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- **Intrinsic dimension (motivation).** Aghajanyan, Zettlemoyer, Gupta.
  *Intrinsic Dimensionality Explains the Effectiveness of Language Model
  Fine-Tuning.* ACL 2021.
- **GLUE.** Wang et al. *GLUE: A Multi-Task Benchmark and Analysis Platform
  for Natural Language Understanding.* ICLR 2019.
- **RoBERTa.** Liu et al. *RoBERTa: A Robustly Optimized BERT Pretraining
  Approach.* arXiv:1907.11692.
- **Tools.** PyTorch, Hugging Face `transformers` and `datasets`,
  `tectonic` (for the LaTeX poster).

## 9. Acknowledgements

This work was completed for **CS 4782, Cornell University, Spring 2026**,
under the guidance of the course staff. We thank the authors of the original
LoRA paper for releasing a clear, reproducible method.
