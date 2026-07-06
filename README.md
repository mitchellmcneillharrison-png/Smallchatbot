# Tiny GPT (from scratch)

A minimal, fully from-scratch implementation of a GPT-style decoder-only
Transformer language model in PyTorch, built for learning how these models
actually work under the hood.

**No Hugging Face, no pretrained weights, no `nn.MultiheadAttention` /
`nn.TransformerEncoder` shortcuts.** Every core component — tokenizer,
embeddings, positional encoding, multi-head self-attention, feed-forward
layers, layer normalization, residual connections, the training loop,
checkpointing, and text generation — is implemented by hand and commented so
you can read the code as a companion to the theory.

## Architecture

```
tokens (ids) ──► token embedding ──┐
                                    ├─► + ──► dropout ──► [ TransformerBlock × N ] ──► LayerNorm ──► linear head ──► logits
        positions ──► positional encoding ──┘

TransformerBlock:
    x ──► LayerNorm ──► Multi-Head Causal Self-Attention ──► (+x, residual) ──►
      ──► LayerNorm ──► Feed-Forward (Linear → GELU → Linear) ──► (+x, residual) ──► out
```

- **Tokenizer** (`src/tokenizer.py`) — character-level: every unique character
  in the training text becomes one token. Simple, has no "unknown token"
  problem, and keeps the rest of the pipeline easy to follow. (Swap in a
  sub-word BPE tokenizer later without touching any other module — everything
  downstream only depends on `vocab_size`, `encode`, and `decode`.)
- **Embeddings & positional encoding** (`src/model/positional.py`) — token ids
  are looked up in a learned embedding table; a second learned table (GPT-style)
  or a fixed sinusoidal function (original Transformer-style, included for
  comparison) adds position information, since attention has no inherent sense
  of order.
- **Multi-head causal self-attention** (`src/model/attention.py`) — implemented
  manually with explicit query/key/value projections, scaled dot-product
  scores, a triangular causal mask, and softmax — no library attention call.
- **Feed-forward network & LayerNorm** (`src/model/layers.py`) — a hand-written
  `LayerNorm` (mean/variance normalization + learned scale & shift) and the
  position-wise MLP (`Linear → GELU → Linear`) used in every block.
- **Transformer block & residual connections** (`src/model/block.py`) —
  pre-norm attention and feed-forward sub-layers, each wrapped in a residual
  ("skip") connection.
- **Full model** (`src/model/gpt.py`) — stacks the blocks, ties the input
  embedding and output projection weights (a common GPT-2 trick), and exposes
  `generate()` for autoregressive sampling with temperature and top-k.

## Project layout

```
.
├── data/sample.txt        # small original text corpus for a quick demo
├── src/
│   ├── config.py          # GPTConfig dataclass (architecture hyperparameters)
│   ├── tokenizer.py        # CharTokenizer (from scratch)
│   ├── dataset.py          # sliding-window next-token-prediction dataset
│   ├── utils.py            # seeding, device selection, LR schedule
│   └── model/
│       ├── layers.py       # LayerNorm, FeedForward
│       ├── positional.py   # learned + sinusoidal positional encodings
│       ├── attention.py    # CausalSelfAttention
│       ├── block.py        # TransformerBlock (attention + FFN + residuals)
│       └── gpt.py          # full GPT model, forward() and generate()
├── train.py                # training loop with checkpointing
├── generate.py              # load a checkpoint and sample text
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

This trains on `data/sample.txt` (a short original story, just large enough to
demonstrate the pipeline — expect the tiny default model to memorize it
somewhat rather than generalize; point `--data_path` at a larger corpus for
more interesting results) and writes a checkpoint to `checkpoints/ckpt.pt`
every `--eval_interval` steps.

Useful flags:

```bash
python train.py \
  --data_path data/sample.txt \
  --block_size 128 --n_layer 4 --n_head 4 --n_embd 128 \
  --batch_size 32 --max_steps 2000 --lr 3e-4
```

Resume training from a checkpoint:

```bash
python train.py --resume checkpoints/ckpt.pt --max_steps 4000
```

Each checkpoint bundles the model weights, optimizer state, architecture
config, and the tokenizer's vocabulary, so `generate.py` can rebuild the exact
model without needing any of the training flags again.

## Generate text

```bash
python generate.py --checkpoint checkpoints/ckpt.pt --prompt "The lighthouse" \
  --max_new_tokens 300 --temperature 0.8 --top_k 50
```

- `--temperature` — lower (e.g. 0.5) is more conservative/repetitive, higher
  (e.g. 1.2) is more random.
- `--top_k` — restricts sampling to the k most likely next tokens at each
  step; omit or set to a large value to sample from the full distribution.

## Notes on scale

Default hyperparameters (4 layers, 4 heads, 128-dim embeddings, 128-token
context) make this runnable on a laptop CPU in a couple of minutes. It is
intentionally tiny — the point is to see every matrix multiplication that
makes up a GPT, not to produce fluent long-form text. Bump `--n_layer`,
`--n_embd`, `--block_size`, and train on more data to see meaningfully better
generations.
