"""
Building-block layers: Layer Normalization and the position-wise Feed-Forward network.

Both are implemented manually rather than relying on nn.LayerNorm, so the exact
computation is visible and explained.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization (Ba, Kiros & Hinton, 2016).

    For each token independently, normalize its feature vector to zero mean and
    unit variance, then apply a learnable per-feature scale (`weight`) and shift
    (`bias`). Unlike BatchNorm, the statistics are computed per-example over the
    embedding dimension, not across the batch -- which is what makes it work well
    for sequences of varying length and small batch sizes.

        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    """

    def __init__(self, n_embd: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = x_normalized * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (the "MLP" half of a Transformer block).

    Every token's vector is passed independently through:
        Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd) -> Dropout
    The 4x expansion factor matches the original Transformer and GPT-2; it gives
    the network extra capacity to combine features before projecting back down to
    the residual stream's dimension.
    """

    def __init__(self, n_embd: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.fc_in = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.fc_out = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.gelu(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x
