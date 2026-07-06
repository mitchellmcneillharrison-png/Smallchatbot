"""
Load a trained checkpoint and generate text from a prompt.

Usage:
    python generate.py --checkpoint checkpoints/ckpt.pt --prompt "Once upon a time" \
        --max_new_tokens 300 --temperature 0.8 --top_k 50
"""

import argparse

import torch

from src.config import GPTConfig
from src.model import GPT
from src.tokenizer import CharTokenizer
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a trained tiny GPT checkpoint.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8, help="higher = more random")
    p.add_argument("--top_k", type=int, default=50, help="restrict sampling to the k most likely tokens")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["vocab"])

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
