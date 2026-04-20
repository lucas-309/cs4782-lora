"""From-scratch LoRA implementation for RoBERTa.

Implements Hu et al. 2021, eq. (3): h = W0 x + (alpha/r) B A x, with A ~ N(0, sigma^2),
B = 0 at init. Only A and B are trainable; W0 is frozen.

Wraps the query and value projections of every self-attention layer, matching
the paper's recommended {Wq, Wv} configuration (Table 5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("query", "value")


class LoRALinear(nn.Module):
    """A frozen Linear with a trainable low-rank update added in parallel."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + update * self.scaling


def inject_lora(model: nn.Module, config: LoRAConfig) -> int:
    """Replace target nn.Linear modules with LoRALinear in-place.

    Returns the number of layers wrapped.
    """
    wrapped = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in config.target_modules and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, config.r, config.alpha, config.dropout),
                )
                wrapped += 1
    return wrapped


def mark_only_lora_and_head_trainable(model: nn.Module) -> None:
    """Freeze everything except LoRA params and the classification head."""
    for name, p in model.named_parameters():
        if "lora_" in name or "classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
