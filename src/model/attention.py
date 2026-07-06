"""
Multi-head causal self-attention, implemented from first principles
(no nn.MultiheadAttention / F.scaled_dot_product_attention shortcuts).

The core idea of self-attention: for every token, build a query vector (what am
I looking for?), and compare it against every other token's key vector (what do
I contain?) to get attention weights, then take a weighted sum of every token's
value vector (what do I actually offer?) according to those weights. "Multi-head"
just means doing this several times in parallel with smaller vectors, so the
model can attend to different kinds of relationships at once.

"Causal" means a triangular mask prevents each position from attending to future
positions -- required for autoregressive generation, where the model must only
ever use tokens it has already produced.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # One combined linear layer projects each token into its query, key, and
        # value vectors in a single matmul (equivalent to three separate
        # projections, but slightly more efficient).
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular matrix of ones: causal_mask[i, j] == 1 iff j <= i,
        # i.e. position i is allowed to look at position j. Registered as a
        # buffer (not a parameter) so it moves with .to(device) but is never
        # trained or saved-as-a-weight-that-needs-gradients.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.qkv_proj(x)                       # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)                 # each (B, T, C)

        # Reshape C -> (n_head, head_dim) and move the head dimension next to
        # batch, so attention is computed independently and in parallel per head.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: how much should each query
        # attend to each key? Scaling by 1/sqrt(head_dim) keeps the dot products
        # (and thus the softmax gradients) well-behaved regardless of head_dim.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)

        # Block out future positions before the softmax by setting their score to -inf,
        # so they receive ~0 probability mass.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                    # (B, nh, T, hd) weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back into one vector

        y = self.resid_dropout(self.out_proj(y))
        return y
