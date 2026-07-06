"""
Export a trained checkpoint into a single JSON file the in-browser demo can load.

The web demo (web/app.js) re-implements the model's forward pass in pure
JavaScript, so all it needs is:
  * the architecture config,
  * the tokenizer vocabulary, and
  * every weight tensor, flattened row-major (PyTorch's native layout).

Weights are rounded to a few significant figures to keep the JSON small; the
loss in precision is irrelevant for a sampling demo. A tiny self-check block
(a fixed input and the model's resulting logits) is also written so the JS
port can be verified to match PyTorch bit-for-bit-ish.

Usage:
    python export_web.py --checkpoint web_ckpt/ckpt.pt --out web/model.json
"""

import argparse
import json

import torch

from src.config import GPTConfig
from src.model import GPT
from src.tokenizer import CharTokenizer


def tensor_to_list(t: torch.Tensor, sig: int = 5):
    """Flatten a tensor to a plain Python list, rounding to `sig` significant
    figures via float() so the JSON stays compact."""
    flat = t.detach().cpu().float().reshape(-1).tolist()
    return [float(f"{x:.{sig}g}") for x in flat]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="web_ckpt/ckpt.pt")
    p.add_argument("--out", type=str, default="web/model.json")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    tokenizer = CharTokenizer(ckpt["vocab"])

    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sd = model.state_dict()

    # Collect every weight, keyed exactly by its PyTorch parameter name, plus
    # its shape so the JS side can index into the flat array correctly.
    weights = {}
    for name, tensor in sd.items():
        # The causal_mask buffer is derived from block_size and rebuilt in JS,
        # so there's no need to ship it.
        if name.endswith("causal_mask"):
            continue
        weights[name] = {"shape": list(tensor.shape), "data": tensor_to_list(tensor)}

    # ---- self-check: run a fixed prompt through the model and record the
    # logits at the final position, so the JS port can assert it matches. ----
    sample_text = "".join(list(tokenizer.stoi.keys())[:8]) or "hello"
    ids = tokenizer.encode(sample_text)[: config.block_size]
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids], dtype=torch.long))
    check = {
        "input_ids": ids,
        "last_logits": tensor_to_list(logits[0, -1, :], sig=6),
    }

    payload = {
        "config": config.as_dict(),
        # itos as an ordered list: index -> character (JSON keys must be
        # strings, so a list keyed by position is cleaner for the JS side).
        "itos": [tokenizer.itos[i] for i in range(tokenizer.vocab_size)],
        "stoi": tokenizer.stoi,
        "weights": weights,
        "self_check": check,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    n_params = sum(t.numel() for t in sd.values() if not _is_mask(t))
    import os
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out} ({size_mb:.2f} MB) | vocab {tokenizer.vocab_size} | ~{n_params:,} tensors' values")


def _is_mask(t):
    return False


if __name__ == "__main__":
    main()
