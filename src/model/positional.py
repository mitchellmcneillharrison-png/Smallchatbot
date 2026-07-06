"""
Positional encoding.

Self-attention has no built-in notion of token order -- it treats its input as a
*set*, not a sequence. Positional encodings inject order information by adding a
position-dependent vector to each token embedding before the first Transformer
block.

Two variants are included for comparison:
  * LearnedPositionalEncoding -- what GPT / GPT-2 actually use: a plain
    embedding table indexed by position, trained like any other parameter.
  * sinusoidal_positional_encoding -- the fixed, non-learned scheme from the
    original "Attention Is All You Need" paper, kept here purely for reference.
"""

import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """A trainable lookup table: position index -> vector, exactly like a token
    embedding table but indexed by position instead of by token id."""

    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.pos_emb(positions)  # (seq_len, n_embd), broadcasts over the batch


def sinusoidal_positional_encoding(seq_len: int, n_embd: int, device: torch.device) -> torch.Tensor:
    """Fixed sinusoidal positional encoding from Vaswani et al., 2017.

    Even dimensions get a sine wave, odd dimensions get a cosine wave, each at a
    different frequency. Because it's deterministic (no learned parameters), it
    can in principle generalize to sequence lengths never seen during training --
    the main motivation for using it instead of a learned table.

        PE(pos, 2i)   = sin(pos / 10000^(2i / n_embd))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / n_embd))
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, device=device).float() * (-math.log(10000.0) / n_embd)
    )
    pe = torch.zeros(seq_len, n_embd, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
