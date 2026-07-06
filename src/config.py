"""
Model configuration.

Keeping every architectural hyperparameter in one small dataclass makes it easy to
save a model's exact shape alongside its weights in a checkpoint, so it can be
reconstructed later without guessing.
"""

from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int          # number of unique tokens the tokenizer produces
    block_size: int = 128    # max context length (number of tokens attended to at once)
    n_layer: int = 4         # number of stacked Transformer decoder blocks
    n_head: int = 4          # number of attention heads per block
    n_embd: int = 128        # embedding / hidden dimension (must be divisible by n_head)
    dropout: float = 0.1     # dropout probability used throughout the network
    bias: bool = True        # whether Linear/LayerNorm layers use a bias term
    pos_encoding: str = "learned"  # "learned" (GPT-style) or "sinusoidal" (original Transformer)

    def as_dict(self):
        return asdict(self)
