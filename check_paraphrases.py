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
    # athletes (worded differently than training)
    (["who is cristiano ronaldo", "who's cristiano ronaldo", "tell me about cristiano ronaldo"], "portuguese"),
    (["who is lionel messi", "tell me about messi"], "messi"),
    (["who is lebron james", "who's lebron james"], "basketball"),
    (["who is serena williams", "tell me about serena williams"], "tennis"),
    (["who is muhammad ali", "who was muhammad ali"], "boxer"),
    (["what sport does lewis hamilton play", "what does lewis hamilton play"], "formula one"),
    (["what sport does tiger woods play", "what does tiger woods play"], "golf"),
    # teams
    (["what league do the lakers play in", "which league are the lakers in",
      "what league do the los angeles lakers play in"], "nba"),
    (["what sport do the yankees play", "which sport do the new york yankees play"], "baseball"),
    (["where are manchester united from", "what city are manchester united from",
      "where do manchester united play"], "manchester"),
    (["what league do real madrid play in", "which league are real madrid in"], "la liga"),
    # leagues / rules
    (["what is the super bowl", "what's the super bowl"], "nfl"),
    (["what is the premier league", "what's the premier league"], "england"),
    (["how many players are on a soccer team", "how many players are in soccer"], "eleven"),
    (["how many points is a touchdown", "how much is a touchdown worth"], "six"),
    (["what sport is played at wimbledon", "which sport is wimbledon"], "tennis"),
    # basics retained
    (["how much is 7 plus 8", "add 7 and 8", "what's 7 + 8"], "15"),
    (["who created you", "who built you"], "scratch"),
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
