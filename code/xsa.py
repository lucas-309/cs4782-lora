"""Exclusive Self-Attention (XSA) patch for HuggingFace BERT/RoBERTa attention.

After computing Y = softmax(QK^T / sqrt(d)) V, XSA removes the component of
each output token along its own value vector V_i, eliminating attention
similarity bias so the layer focuses on cross-token context.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn


def _xsa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # value, attn_output: (B, H, S, D)
    attn_output = torch.matmul(attn_weights, value)

    # XSA: per (batch, head, token), remove projection of output onto self-value.
    v_norm = value / value.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    proj = (attn_output * v_norm).sum(dim=-1, keepdim=True) * v_norm
    attn_output = attn_output - proj

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _xsa_module_forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
    """Drop-in replacement for RobertaSelfAttention.forward using XSA."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)

    query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    attn_output, attn_weights = _xsa_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.dropout.p,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return attn_output, attn_weights


def apply_xsa(model: nn.Module) -> int:
    """Patch every BertSelfAttention / RobertaSelfAttention module to use XSA.

    Returns the number of attention modules patched.
    """
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ in ("BertSelfAttention", "RobertaSelfAttention"):
            module.forward = MethodType(_xsa_module_forward, module)
            patched += 1
    return patched
