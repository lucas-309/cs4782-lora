"""Read the JSON logs in results/runs/ and produce poster figures.

Outputs to figures/:
    rank_ablation.pdf   - accuracy vs r on MRPC
    parameter_efficiency.pdf  - trainable-param bar chart
    training_curve.pdf  - LoRA vs full FT loss curves on MRPC
    xsa_comparison.pdf  - baseline vs LoRA + XSA across ranks on MRPC
    headline_table.tex  - LaTeX-formatted results table
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("results/runs")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cornell-ish red as the accent
ACCENT = "#B31B1B"
INK = "#222222"
MUTED = "#888888"
PDF_METADATA = {
    "Creator": "code/plot_results.py",
    "Producer": "matplotlib",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_runs() -> dict[str, dict]:
    runs = {}
    for path in RUNS_DIR.glob("*.json"):
        with open(path) as f:
            runs[path.stem] = json.load(f)
    return runs


def final_metric(run: dict) -> float:
    """Return the headline metric (accuracy / matthews / pearson)."""
    final = run.get("final", {})
    for k in ("accuracy", "matthews_correlation", "pearson"):
        if k in final:
            return final[k]
    return float("nan")


def plot_rank_ablation(runs: dict) -> None:
    """Accuracy vs rank on MRPC, with full-FT and head-only as reference lines."""
    points = []
    full_acc = None
    head_acc = None
    for name, run in runs.items():
        a = run["args"]
        if a.get("task") != "mrpc":
            continue
        m = final_metric(run)
        if a["method"] == "lora":
            points.append((a["rank"], m, run["trainable"]))
        elif a["method"] == "full":
            full_acc = m
        elif a["method"] == "head":
            head_acc = m
    points.sort()
    if not points:
        print("[skip] rank_ablation: no MRPC LoRA runs")
        return
    ranks = [p[0] for p in points]
    accs = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    if full_acc is not None:
        ax.axhline(full_acc, color=MUTED, linestyle="--", linewidth=1.5,
                   label=f"Full fine-tune ({full_acc:.3f})")
    if head_acc is not None:
        ax.axhline(head_acc, color="#bbbbbb", linestyle=":", linewidth=1.5,
                   label=f"Head only ({head_acc:.3f})")
    ax.plot(ranks, accs, "o-", color=ACCENT, linewidth=2.5, markersize=10,
            label="LoRA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank $r$", fontsize=12)
    ax.set_ylabel("MRPC accuracy", fontsize=12)
    ax.set_title("LoRA accuracy is flat across ranks", fontsize=13, color=INK, pad=10)
    ax.set_ylim(0.65, 0.92)
    ax.grid(axis="y", alpha=0.3)
    for r, a in zip(ranks, accs):
        ax.annotate(f"{a:.3f}", (r, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rank_ablation.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print("[saved] rank_ablation.pdf")


def plot_parameter_efficiency(runs: dict) -> None:
    """Trainable-param comparison: full FT vs LoRA on MRPC."""
    bars = []
    for name, run in runs.items():
        a = run["args"]
        if a.get("task") != "mrpc":
            continue
        method = a["method"]
        if method == "lora":
            label = f"LoRA r={a['rank']}"
        elif method == "full":
            label = "Full FT"
        else:
            label = "Head only"
        bars.append((label, run["trainable"], final_metric(run), method))
    if not bars:
        print("[skip] parameter_efficiency: no MRPC runs")
        return
    bars.sort(key=lambda b: b[1])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels = [b[0] for b in bars]
    params = [b[1] / 1e6 for b in bars]
    accs = [b[2] for b in bars]
    colors = [ACCENT if b[3] == "lora" else MUTED for b in bars]

    xs = np.arange(len(bars))
    ax.bar(xs, params, color=colors, edgecolor=INK, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Trainable params (M)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("LoRA trains 140$\\times$ fewer params", fontsize=13, pad=10)
    for x, p, a in zip(xs, params, accs):
        ax.annotate(f"{a:.3f}", (x, p), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "parameter_efficiency.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print("[saved] parameter_efficiency.pdf")


def plot_training_curves(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    plotted = False
    for name, run in runs.items():
        a = run["args"]
        if a.get("task") != "mrpc":
            continue
        method = a["method"]
        steps = [s["step"] for s in run.get("steps", [])]
        losses = [s["loss"] for s in run.get("steps", [])]
        if not steps:
            continue
        if method == "lora" and a["rank"] == 8:
            ax.plot(steps, losses, color=ACCENT, label="LoRA r=8 (0.7% params)", linewidth=2)
            plotted = True
        elif method == "full":
            ax.plot(steps, losses, color=MUTED, label="Full fine-tune (100% params)", linewidth=2, linestyle="--")
            plotted = True
    if not plotted:
        print("[skip] training_curve: no curves to plot")
        plt.close(fig)
        return
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Training loss", fontsize=12)
    ax.set_title("Same trajectory, fraction of the params", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print("[saved] training_curve.pdf")


def plot_xsa_comparison(runs: dict) -> None:
    """Compare LoRA baseline vs LoRA + XSA across ranks on MRPC."""
    baseline, xsa = {}, {}
    for name, run in runs.items():
        a = run["args"]
        if a.get("task") != "mrpc" or a.get("method") != "lora":
            continue
        is_xsa = a.get("xsa", False) or name.endswith("_xsa")
        (xsa if is_xsa else baseline)[a["rank"]] = final_metric(run)

    ranks = sorted(set(baseline) & set(xsa))
    if not ranks:
        print("[skip] xsa_comparison: missing baseline/xsa pairs")
        return
    base_accs = [baseline[r] for r in ranks]
    xsa_accs = [xsa[r] for r in ranks]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(ranks, base_accs, "o-", color=ACCENT, linewidth=2.5, markersize=10,
            label="LoRA (baseline)")
    ax.plot(ranks, xsa_accs, "s--", color=INK, linewidth=2.0, markersize=9,
            label="LoRA + XSA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank $r$", fontsize=12)
    ax.set_ylabel("MRPC accuracy", fontsize=12)
    ax.set_title("XSA hurts at every rank under LoRA", fontsize=13, color=INK, pad=10)
    ax.set_ylim(0.83, 0.88)
    ax.grid(axis="y", alpha=0.3)
    for r, b in zip(ranks, base_accs):
        ax.annotate(f"{b:.3f}", (r, b), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=ACCENT)
    for r, x in zip(ranks, xsa_accs):
        ax.annotate(f"{x:.3f}", (r, x), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xsa_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[saved] xsa_comparison.pdf")


def write_headline_table(runs: dict) -> None:
    """A small TeX snippet listing the headline numbers."""
    rows = []
    seen = set()
    for name, run in sorted(runs.items()):
        a = run["args"]
        task = a.get("task", "?")
        method = a.get("method", "?")
        key = (task, method, a.get("rank") if method == "lora" else None)
        if key in seen:
            continue
        seen.add(key)
        if method == "lora":
            label = f"LoRA $r{{=}}{a['rank']}$"
        elif method == "full":
            label = "Full fine-tune"
        else:
            label = "Head only"
        m = final_metric(run)
        params = run["trainable"]
        rows.append((task.upper(), label, params, m))

    rows.sort(key=lambda r: (r[0], -r[2]))
    out = ["\\begin{tabular}{llrr}", "\\toprule",
           "Task & Method & Trainable & Score \\\\", "\\midrule"]
    for task, label, params, score in rows:
        out.append(f"{task} & {label} & {params/1e6:.2f}M & {score:.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (FIG_DIR / "headline_table.tex").write_text("\n".join(out))
    print("[saved] headline_table.tex")


def main():
    runs = load_runs()
    print(f"[load] {len(runs)} runs from {RUNS_DIR}")
    plot_rank_ablation(runs)
    plot_parameter_efficiency(runs)
    plot_training_curves(runs)
    plot_xsa_comparison(runs)
    write_headline_table(runs)


if __name__ == "__main__":
    main()
