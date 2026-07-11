"""
Measure how well the trained chatbot answers, by exact-matching greedy answers
against the dataset.

    python eval_chat.py                       # evaluate on all of data/chat.jsonl
    python eval_chat.py --sample 300          # evaluate on a random subset (faster)

Greedy decoding (top_k=1) is used so the numbers are deterministic. Accuracy is
reported overall and split into arithmetic vs. the hand-written facts/small-talk,
since those behave quite differently.
"""

import argparse
import json
import random

import torch

from src.chat_tokenizer import WordTokenizer
from src.config import GPTConfig
from src.model import GPT
from src.utils import get_device, quantize_model_int8_


@torch.no_grad()
def greedy_answer(model, tokenizer, question, device, max_new_tokens=48):
    ids = torch.tensor([tokenizer.build_prompt(question)], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1,
        eos_token=tokenizer.stoi["<eos>"],
    )
    return tokenizer.decode(out[0, ids.shape[1]:].tolist())


def normalize(s):
    # The dataset answers are mixed-case; the model emits lowercase. Compare
    # case-insensitively and ignore surrounding whitespace.
    return s.strip().lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/chat.pt")
    p.add_argument("--data_path", type=str, default="data/chat.jsonl")
    p.add_argument("--sample", type=int, default=0, help="evaluate a random subset of this size (0 = all)")
    p.add_argument("--show_errors", type=int, default=10)
    p.add_argument("--quantize", choices=["int8"], default=None,
                   help="quantize the model before evaluating, to measure the accuracy cost")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**ckpt["config"])
    tokenizer = WordTokenizer(ckpt["vocab"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if args.quantize == "int8":
        quantize_model_int8_(model)
        print("(evaluating int8-quantized weights)")

    pairs = [(json.loads(l)["q"], json.loads(l)["a"]) for l in open(args.data_path)]
    if args.sample and args.sample < len(pairs):
        random.seed(0)
        pairs = random.sample(pairs, args.sample)

    stats = {"arith": [0, 0], "facts": [0, 0]}  # [correct, total]
    errors = []
    for q, a in pairs:
        cat = "arith" if "equals" in a else "facts"
        got = greedy_answer(model, tokenizer, q, device)
        ok = normalize(got) == normalize(a)
        stats[cat][0] += int(ok)
        stats[cat][1] += 1
        if not ok and len(errors) < args.show_errors:
            errors.append((q, a, got))

    total_c = stats["arith"][0] + stats["facts"][0]
    total_n = stats["arith"][1] + stats["facts"][1]
    print(f"Checkpoint step {ckpt.get('step')} | evaluated {total_n} questions\n")
    for cat in ("arith", "facts"):
        c, n = stats[cat]
        if n:
            print(f"  {cat:6s}: {c}/{n} = {100*c/n:.1f}%")
    print(f"  {'TOTAL':6s}: {total_c}/{total_n} = {100*total_c/total_n:.1f}%")

    if errors:
        print(f"\nSample errors (up to {args.show_errors}):")
        for q, a, got in errors:
            print(f"  Q: {q}\n    expected: {a}\n    got:      {got}")


if __name__ == "__main__":
    main()
