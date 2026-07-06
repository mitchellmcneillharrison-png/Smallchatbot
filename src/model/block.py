"""
A single Transformer decoder block.

Uses the "pre-norm" arrangement (LayerNorm applied *before* attention/feed-forward,
rather than after) that GPT-2 and most modern Transformers use, since it makes
training deeper stacks more stable.

Each sub-layer is wrapped in a residual ("skip") connection: the sub-layer's
output is *added* to its input rather than replacing it. This gives gradients a
direct path back through the network (mitigating vanishing gradients in deep
stacks) and lets each block learn a small refinement of its input rather than
having to reconstruct it from scratch.
"""

import torch.nn as nn

from .attention import CausalSelfAttention
from .layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffwd = FeedForward(config.n_embd, config.dropout, config.bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection around self-attention
        x = x + self.ffwd(self.ln2(x))   # residual connection around the feed-forward net
        return x
