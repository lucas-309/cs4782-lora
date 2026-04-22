"""Train RoBERTa-base on a GLUE task with LoRA or full fine-tuning.

Usage:
    python train.py --task sst2 --method lora --rank 8 --epochs 3
    python train.py --task mrpc --method full --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from lora import LoRAConfig, count_trainable, inject_lora, mark_only_lora_and_head_trainable

# GLUE task -> (text fields, num labels, metric)
TASK_INFO = {
    "sst2": (("sentence", None), 2, "accuracy"),
    "mrpc": (("sentence1", "sentence2"), 2, "accuracy"),
    "rte": (("sentence1", "sentence2"), 2, "accuracy"),
    "cola": (("sentence", None), 2, "matthews_correlation"),
    "stsb": (("sentence1", "sentence2"), 1, "pearson"),
    "qnli": (("question", "sentence"), 2, "accuracy"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def matthews(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(labels, preds))


def pearson(preds: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import pearsonr

    return float(pearsonr(preds, labels)[0])


def evaluate(model, loader, device, metric: str) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            if metric == "pearson":
                preds = logits.squeeze(-1)
            else:
                preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    if metric == "accuracy":
        return {"accuracy": float((preds == labels).mean())}
    if metric == "matthews_correlation":
        return {"matthews_correlation": matthews(preds, labels)}
    if metric == "pearson":
        return {"pearson": pearson(preds, labels)}
    raise ValueError(metric)


def build_loaders(task: str, tokenizer, batch_size: int, max_len: int):
    fields, num_labels, _ = TASK_INFO[task]
    raw = load_dataset("glue", task)

    def preprocess(ex):
        if fields[1] is None:
            enc = tokenizer(ex[fields[0]], truncation=True, max_length=max_len, padding="max_length")
        else:
            enc = tokenizer(
                ex[fields[0]], ex[fields[1]], truncation=True, max_length=max_len, padding="max_length"
            )
        enc["labels"] = ex["label"]
        return enc

    cols_to_drop = [c for c in raw["train"].column_names if c not in ("label",)]
    tokenized = raw.map(preprocess, batched=True, remove_columns=cols_to_drop)
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    train = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)
    val_split = "validation_matched" if task == "mnli" else "validation"
    val = DataLoader(tokenized[val_split], batch_size=batch_size, shuffle=False)
    return train, val, num_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_INFO.keys()), required=True)
    p.add_argument("--method", choices=["lora", "full", "head"], default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--out", default="results/runs")
    p.add_argument("--max-train-steps", type=int, default=None,
                   help="Cap training steps for fast iteration on a laptop.")
    args = p.parse_args()

    if args.lr is None:
        args.lr = 5e-4 if args.method == "lora" else 2e-5

    set_seed(args.seed)
    device = get_device()
    print(f"[device] {device}")

    fields, num_labels, metric = TASK_INFO[args.task]
    is_regression = metric == "pearson"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        problem_type="regression" if is_regression else None,
    )

    if args.method == "lora":
        wrapped = inject_lora(model, LoRAConfig(r=args.rank, alpha=args.alpha))
        mark_only_lora_and_head_trainable(model)
        print(f"[lora] wrapped {wrapped} linear layers, r={args.rank}, alpha={args.alpha}")
    elif args.method == "head":
        for name, par in model.named_parameters():
            par.requires_grad = "classifier" in name

    trainable, total = count_trainable(model)
    print(f"[params] trainable={trainable:,} total={total:,} ({100*trainable/total:.3f}%)")
    model.to(device)

    train_loader, val_loader, _ = build_loaders(args.task, tokenizer, args.batch_size, args.max_len)
    n_steps_per_epoch = len(train_loader)
    n_total = n_steps_per_epoch * args.epochs
    if args.max_train_steps is not None:
        n_total = min(n_total, args.max_train_steps)
    print(f"[data] train_batches/epoch={n_steps_per_epoch}, val_batches={len(val_loader)}, total_steps={n_total}")

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.06 * n_total), num_training_steps=n_total)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    run_id = f"{args.task}_{args.method}_r{args.rank}_s{args.seed}"
    log_path = Path(args.out) / f"{run_id}.json"
    history = {"args": vars(args), "trainable": trainable, "total": total, "steps": [], "eval": []}

    step = 0
    t0 = time.time()
    done = False
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            if step % 50 == 0:
                history["steps"].append({"step": step, "loss": float(loss.item())})
                print(f"  step {step}/{n_total} loss={loss.item():.4f}")
            step += 1
            if args.max_train_steps and step >= args.max_train_steps:
                done = True
                break
        m = evaluate(model, val_loader, device, metric)
        elapsed = time.time() - t0
        history["eval"].append({"epoch": epoch + 1, "step": step, "elapsed_s": elapsed, **m})
        print(f"[eval epoch {epoch+1}] {m} elapsed={elapsed:.0f}s")
        if done:
            break

    history["final"] = history["eval"][-1] if history["eval"] else {}
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] wrote {log_path}")


if __name__ == "__main__":
    main()
