"""
Validate paraphrase robustness: ask the trained chatbot questions worded
DIFFERENTLY from the training templates and check it still gives the right
answer. This is the property the model is really being trained for -- "who was
the first president of the USA" and "who was the 1st president of the United
States" should land on the same answer.

    python check_paraphrases.py --checkpoint checkpoints/chat.pt

Each probe lists several rewordings and a keyword that must appear in the answer.
"""

import argparse

import torch

from src.chat_tokenizer import WordTokenizer
from src.config import GPTConfig
from src.model import GPT
from src.utils import get_device

# (list of reworded questions, keyword expected in the answer)
PROBES = [
    (["who was the first president of the usa",
      "who was the 1st president of the united states",
      "name the first president of america",
      "tell me the 1st president of the united states of america"], "washington"),
    (["what's the capital city of japan",
      "tell me japan's capital",
      "do you know the capital of japan"], "tokyo"),
    (["how much is 7 plus 8", "add 7 and 8", "what's 7 + 8"], "15"),
    (["what is 6 times 7", "multiply 6 and 7", "6 x 7"], "42"),
    (["which planet is closest to the sun", "what is the nearest planet to the sun"], "mercury"),
    (["what's the tallest mountain on earth", "name the highest mountain"], "everest"),
    (["what is the chemical symbol for gold? just kidding, for oxygen",
      "oxygen's symbol", "what is the symbol for oxygen"], "o"),
    (["how many days does a week have", "number of days in a week"], "seven"),
    (["who created you", "who is your maker", "who built you"], "scratch"),
    (["what currency does france use", "tell me the currency of france"], "euro"),
    (["what language do people speak in brazil", "brazil's language"], "portuguese"),
    (["what is the opposite of hot", "give me the opposite of hot"], "cold"),
]


@torch.no_grad()
def greedy(model, tok, q, device):
    ids = torch.tensor([tok.build_prompt(q)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=48, temperature=1.0, top_k=1, eos_token=tok.stoi["<eos>"])
    return tok.decode(out[0, ids.shape[1]:].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/chat.pt")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = WordTokenizer(ckpt["vocab"])

    total, correct = 0, 0
    for rewordings, keyword in PROBES:
        hits = 0
        for q in rewordings:
            ans = greedy(model, tok, q, device)
            ok = keyword.lower() in ans.lower()
            hits += ok
            total += 1
            correct += ok
            if args.verbose or not ok:
                flag = "ok " if ok else "MISS"
                print(f"  [{flag}] {q!r} -> {ans!r}")
        print(f"[{hits}/{len(rewordings)}] keyword {keyword!r}")
    print(f"\nParaphrase robustness: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
