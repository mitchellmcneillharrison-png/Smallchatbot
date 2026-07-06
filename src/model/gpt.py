"""
The full GPT-style model: token embeddings + positional encoding -> a stack of
Transformer decoder blocks -> final LayerNorm -> a linear "head" that projects
back into vocabulary-sized logits.

Data flow for a batch of token ids `idx` of shape (B, T):
    (B, T) --token_emb--> (B, T, n_embd) --+ positions--> (B, T, n_embd)
           --N x TransformerBlock-->        (B, T, n_embd)
           --ln_f-->                        (B, T, n_embd)
           --lm_head-->                     (B, T, vocab_size)  <- logits
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .layers import LayerNorm
from .positional import LearnedPositionalEncoding, sinusoidal_positional_encoding


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)

        if config.pos_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(config.block_size, config.n_embd)
        elif config.pos_encoding == "sinusoidal":
            self.pos_encoding = None  # computed on the fly in forward(), no parameters to store
        else:
            raise ValueError(f"Unknown pos_encoding: {config.pos_encoding!r}")

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017 -- used by GPT-2): the input token
        # embedding and the output projection share the same weight matrix.
        # Intuitively, "the vector that represents token X as input" and "the
        # vector used to score token X as a prediction" can reasonably be the
        # same thing, and tying them cuts the parameter count noticeably.
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Extra scaled-down init for the projections that feed directly into a
        # residual connection (GPT-2 paper, section 2.3). Without this, the
        # variance of activations grows with depth as residual contributions
        # accumulate across many layers.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        if self.pos_encoding is not None:
            pos_emb = self.pos_encoding(T, idx.device)  # (T, n_embd)
        else:
            pos_emb = sinusoidal_positional_encoding(T, self.config.n_embd, idx.device)

        x = self.drop(tok_emb + pos_emb)  # broadcast positions over the batch dimension
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and time dimensions so cross_entropy scores every
            # (position -> next token) prediction independently.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` new tokens after the prompt `idx`.

        Each step: run the model on the current sequence, look only at the
        logits for the *last* position (the prediction for "what comes next"),
        turn them into a probability distribution, and sample one token from it.
        That token is appended and fed back in for the next step.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # The model only has `block_size` positional embeddings, so if the
            # running sequence is longer than that, only feed it the most
            # recent `block_size` tokens.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size); temperature controls randomness

            if top_k is not None:
                # Zero out (via -inf) every logit outside the top-k, so sampling
                # is restricted to the k most likely next tokens.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
