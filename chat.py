"""
Talk to the trained chatbot from the command line.

    python chat.py                                  # interactive REPL
    python chat.py --question "what is 2 plus 2?"   # single question

Remember: this is a tiny closed-domain model. It genuinely answers questions it
was trained on (arithmetic over small numbers, greetings, and the facts in
build_chat_data.py) and close rewordings of them. Ask it something outside that
world and it will confidently make something up -- that's the nature of a model
this small.
"""

import argparse

import torch

from src.chat_tokenizer import WordTokenizer
from src.config import GPTConfig
from src.model import GPT
from src.utils import get_device


def load(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    config = GPTConfig(**ckpt["config"])
    tokenizer = WordTokenizer(ckpt["vocab"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def answer(model, tokenizer, question, device, temperature=0.4, top_k=20, max_new_tokens=48):
    ids = torch.tensor([tokenizer.build_prompt(question)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token=tokenizer.stoi["<eos>"],
    )
    answer_ids = out[0, ids.shape[1]:].tolist()
    return tokenizer.decode(answer_ids)


def main():
    p = argparse.ArgumentParser(description="Chat with the tiny from-scratch chatbot.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/chat.pt")
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    device = get_device()
    model, tokenizer = load(args.checkpoint, device)

    if args.question is not None:
        print(answer(model, tokenizer, args.question, device, args.temperature, args.top_k))
        return

    print("Tiny chatbot ready. Type a question (Ctrl-C or 'quit' to exit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue
        print("Bot:", answer(model, tokenizer, q, device, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
