"""
Export the trained chatbot checkpoint into web/model.json for the in-browser demo.

Same idea as export_web.py, but also ships the word-level tokenizer (vocabulary
+ special-token ids) so the JavaScript side can turn text into ids and back
exactly the way training did.

Usage:
    python export_chat_web.py --checkpoint checkpoints/chat.pt --out web/model.json
"""

import argparse
import json
import os

import torch

from src.chat_tokenizer import SPECIAL_TOKENS, WordTokenizer
from src.config import GPTConfig
from src.model import GPT


def tensor_to_list(t, sig: int = 5):
    flat = t.detach().cpu().float().reshape(-1).tolist()
    return [float(f"{x:.{sig}g}") for x in flat]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/chat.pt")
    p.add_argument("--out", type=str, default="web/model.json")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    tokenizer = WordTokenizer(ckpt["vocab"])

    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    weights = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("causal_mask"):
            continue
        weights[name] = {"shape": list(tensor.shape), "data": tensor_to_list(tensor)}

    # self-check: fixed prompt -> final-position logits, for web/verify.mjs.
    check_ids = tokenizer.build_prompt("what is 1 plus 1?")[: config.block_size]
    with torch.no_grad():
        logits, _ = model(torch.tensor([check_ids], dtype=torch.long))

    payload = {
        "config": config.as_dict(),
        "tokenizer": {
            "type": "word",
            "itos": [tokenizer.itos[i] for i in range(tokenizer.vocab_size)],
            "stoi": tokenizer.stoi,
            "specials": {
                "pad": tokenizer.stoi["<pad>"],
                "unk": tokenizer.stoi["<unk>"],
                "user": tokenizer.stoi["<user>"],
                "bot": tokenizer.stoi["<bot>"],
                "eos": tokenizer.stoi["<eos>"],
            },
            "special_list": SPECIAL_TOKENS,
        },
        "weights": weights,
        "self_check": {"input_ids": check_ids, "last_logits": tensor_to_list(logits[0, -1, :])},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out} ({size_mb:.2f} MB) | vocab {tokenizer.vocab_size} | block {config.block_size}")


if __name__ == "__main__":
    main()
