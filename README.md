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

## Run it in the browser (Vercel demo)

The `web/` directory is a zero-dependency static site that runs the model
**entirely client-side** — no server, no API. `web/gpt.js` is a hand port of
the PyTorch forward pass (`src/model/gpt.py`) to plain JavaScript over
`Float32Array`, using an incremental KV cache so generation is fast. The
trained weights ship as `web/model.json`.

> **What it is (and isn't):** the deployed demo is a ~0.8M-parameter,
> character-level model trained on a single short story. It is a **text
> continuer, not a chatbot** — give it the start of a sentence and it extends
> it in the story's style. It cannot answer questions, hold a conversation, or
> do arithmetic; at this scale those abilities simply aren't there. The point
> is to make the whole GPT architecture small enough to read and run.

Regenerate the weights after training your own model (the shipped `model.json`
was produced with these settings):

```bash
python train.py --out_dir web_ckpt --block_size 128 --n_embd 128 --n_layer 4 --max_steps 2000
python export_web.py --checkpoint web_ckpt/ckpt.pt --out web/model.json
```

Check that the JS port still matches PyTorch (compares logits against a baked-in
reference in `model.json`):

```bash
node web/verify.mjs
```

Preview locally:

```bash
cd web && python -m http.server 8000   # then open http://localhost:8000
```

### Deploying to Vercel

The repo ships a `vercel.json` that serves the `web/` directory as a static
site with no build step. Two ways to deploy:

- **Dashboard:** import the repo at [vercel.com/new](https://vercel.com/new).
  The included `vercel.json` sets the output directory to `web/`, so you can
  accept the defaults and deploy.
- **CLI:** `npm i -g vercel && vercel` from the repo root, then
  `vercel --prod`.

Because everything runs in the browser, the deploy is just static files
(`index.html`, `app.js`, `gpt.js`, `model.json`) — it works on Vercel's free
tier with no environment variables or backend.

## Notes on scale

Default hyperparameters (4 layers, 4 heads, 128-dim embeddings, 128-token
context) make this runnable on a laptop CPU in a couple of minutes. It is
intentionally tiny — the point is to see every matrix multiplication that
makes up a GPT, not to produce fluent long-form text. Bump `--n_layer`,
`--n_embd`, `--block_size`, and train on more data to see meaningfully better
generations.
