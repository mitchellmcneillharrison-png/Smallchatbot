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

The repo has two things you can train with the **same** hand-built model:

1. A **story text generator** (character-level) — the classic "train a tiny GPT
   on some text and watch it babble" demo (`train.py` / `generate.py`).
2. A **sports chatbot** (word-level) that actually answers questions —
   hundreds of athletes, 100+ teams, leagues, championships, and rules, plus a
   little basic knowledge (arithmetic, greetings), built from a ~13k-pair
   dataset (`train_chat.py` / `chat.py`). This is what the browser demo deploys.

   The key trick is **paraphrase robustness**: `build_chat_data.py` generates
   many wordings of every fact ("who is *Cristiano Ronaldo*" / "who's Ronaldo" /
   "tell me about Ronaldo"; "what league do *the Lakers* play in" / "which
   league are the Los Angeles Lakers in"), so the model answers the same
   regardless of phrasing (see `eval_chat.py` and `check_paraphrases.py`).

> **Honest expectations for the chatbot:** it is *tiny* and trained *only* on the
> sports dataset in `build_chat_data.py` — ~350 athletes (soccer, basketball,
> tennis, American football, baseball, boxing, golf, Formula One, hockey,
> cricket, athletics), 100+ teams (full NBA and NFL, plus MLB, NHL, and major
> soccer clubs) with leagues and home venues, playing positions, rules and
> terminology — plus basic arithmetic and greetings. It answers those questions
> and rewordings of them,
> but it has no general knowledge or reasoning — ask anything outside its
> training world and it will confidently make something up. It demonstrates the
> *architecture and training recipe*, not intelligence.

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
├── data/
│   ├── sample.txt         # small original text corpus for the story demo
│   └── chat.jsonl         # generated Q&A pairs for the chatbot (build_chat_data.py)
├── src/
│   ├── config.py          # GPTConfig dataclass (architecture hyperparameters)
│   ├── tokenizer.py        # CharTokenizer (story demo, from scratch)
│   ├── chat_tokenizer.py   # WordTokenizer + special tokens (chatbot)
│   ├── dataset.py          # sliding-window next-token dataset (story)
│   ├── chat_dataset.py     # instruction dataset w/ answer-only loss masking (chatbot)
│   ├── utils.py            # seeding, device selection, LR schedule
│   └── model/
│       ├── layers.py       # LayerNorm, FeedForward
│       ├── positional.py   # learned + sinusoidal positional encodings
│       ├── attention.py    # CausalSelfAttention
│       ├── block.py        # TransformerBlock (attention + FFN + residuals)
│       └── gpt.py          # full GPT model, forward() and generate()
├── train.py                # story training loop
├── generate.py              # sample text from the story model
├── build_chat_data.py       # generate the chatbot's Q&A dataset
├── train_chat.py            # chatbot training loop (instruction tuning)
├── chat.py                  # talk to the trained chatbot (CLI)
├── export_web.py / export_chat_web.py  # export weights to web/model.json
├── web/                     # in-browser demo (static, deploys to Vercel)
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

## Train the chatbot

The chatbot uses the **same model** as the story demo, but with a word-level
tokenizer and instruction-style training data. Generate the data, then train:

```bash
python build_chat_data.py     # writes data/chat.jsonl (Q&A pairs)
python train_chat.py          # instruction-tunes the model (~a few minutes on CPU)
```

`train_chat.py` formats each example as `<user> question <bot> answer <eos>` and
**masks the loss to the answer only**, so the model is rewarded purely for
producing correct answers. It prints sample answers to a few probe questions at
every eval step so you can watch it learn. The checkpoint is saved to
`checkpoints/chat.pt`.

Then talk to it:

```bash
python chat.py                                  # interactive REPL
python chat.py --question "what is 7 plus 8?"   # single question
```

Teach it new things by editing the `FACTS` list in `build_chat_data.py` and
re-running both scripts.

## Run it in the browser (Vercel demo)

The `web/` directory is a zero-dependency static site that runs the **chatbot**
model **entirely client-side** — no server, no API. `web/gpt.js` is a hand port
of the PyTorch forward pass (`src/model/gpt.py`) to plain JavaScript over
`Float32Array`, using an incremental KV cache so generation is fast;
`web/tokenizer.js` is a matching port of the word-level tokenizer. Weights are
quantized to per-row int8 and shipped as a raw binary blob (`web/model.bin`,
~7 MB for a ~7M-parameter model) addressed by a tiny JSON manifest (`web/model.json`
with config, vocabulary, and per-tensor offsets). The browser fetches the blob
as an `ArrayBuffer` with a progress bar and reads each weight as a typed-array
view — no multi-megabyte `JSON.parse`, so the page becomes interactive quickly.

> **What it is (and isn't):** the deployed demo is a ~7M-parameter sports
> chatbot that genuinely answers questions **within its training world**
> (athletes, teams, leagues, championships, and rules from `build_chat_data.py`,
> plus basic arithmetic) and rewordings of them. Ask it something outside that
> world and it will confidently make something up — it has no general knowledge
> or reasoning. It's a demo of the architecture and training recipe, not a real
> assistant.

Regenerate `web/model.json` after (re)training the chatbot:

```bash
python build_chat_data.py
python train_chat.py
python export_chat_web.py --checkpoint checkpoints/chat.pt --out web/model.json
```

Check that the JS port still matches PyTorch (compares logits against a baked-in
reference in `model.json`):

```bash
node web/verify.mjs
```

(To instead deploy the character-level *story* generator, train it and run
`python export_web.py --checkpoint web_ckpt/ckpt.pt --out web/model.json` — but
the shipped `web/` UI is the chatbot.)

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
(`index.html`, `app.js`, `gpt.js`, `tokenizer.js`, `model.json`, `model.bin`) —
it works on Vercel's free tier with no environment variables or backend.

## Notes on scale

These models are intentionally tiny (a few hundred thousand to a couple million
parameters) so the whole thing trains on a laptop CPU in minutes and every
matrix multiply is readable. That small size is also the ceiling: the chatbot
can memorize and generalize across its little dataset, but real open-domain Q&A
and reasoning are emergent properties of models many orders of magnitude larger
trained on far more data. The goal here is to understand the machine, not to
rival a production assistant.
