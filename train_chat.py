"""
Train the from-scratch GPT as a (closed-domain) chatbot on data/chat.jsonl.

This reuses the exact same model as the story demo (src/model/gpt.py) -- only
the tokenizer (word-level) and the training data (question/answer pairs with
answer-only loss masking) differ. That is the whole point: the architecture is
general; what a model "knows" comes entirely from its data.

Usage:
    python build_chat_data.py            # (re)generate data/chat.jsonl first
    python train_chat.py                 # train with sensible defaults
    python train_chat.py --max_steps 6000 --n_embd 192

Because arithmetic pairs vastly outnumber the hand-written facts, the facts are
oversampled during training (--oversample_facts) so the model gives them roughly
equal attention.
"""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader

from src.chat_dataset import ChatDataset
from src.chat_tokenizer import WordTokenizer
from src.config import GPTConfig
from src.model import GPT
from src.utils import get_device, get_lr, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train the tiny GPT as a closed-domain chatbot.")
    p.add_argument("--data_path", type=str, default="data/chat.jsonl")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_name", type=str, default="chat.pt")
    p.add_argument("--resume", type=str, default=None)

    # architecture
    p.add_argument("--block_size", type=int, default=48)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)

    # optimization
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--oversample_facts", type=int, default=8,
                   help="repeat non-arithmetic examples this many times to balance the data")
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["q"], obj["a"]))
    return pairs


def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


# A few held-out probe questions printed during training so we can watch the
# bot learn. Includes rewordings not in the data to gauge generalization.
PROBES = [
    "what is 3 plus 4?",
    "hello",
    "what is the capital of japan?",
    "who are you?",
    "what color is the sky?",
]


@torch.no_grad()
def sample_answer(model, tokenizer, question, device, max_new_tokens=40):
    model.eval()
    ids = torch.tensor([tokenizer.build_prompt(question)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.4, top_k=20,
                         eos_token=tokenizer.stoi["<eos>"])
    answer_ids = out[0, ids.shape[1]:].tolist()
    model.train()
    return tokenizer.decode(answer_ids)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_pairs = load_pairs(args.data_path)

    # Build the vocabulary from the full (un-oversampled) set so ids are stable.
    tokenizer = WordTokenizer.from_pairs(all_pairs)
    print(f"Vocab size: {tokenizer.vocab_size} | pairs: {len(all_pairs)}")

    # Oversample non-arithmetic ("facts"/small-talk/meta) examples for balance.
    is_arith = lambda a: "equals" in a
    train_pairs = []
    for q, a in all_pairs:
        reps = 1 if is_arith(a) else args.oversample_facts
        train_pairs.extend([(q, a)] * reps)
    random.shuffle(train_pairs)

    # Hold out a small validation slice (deduplicated originals).
    random.shuffle(all_pairs)
    n_val = max(20, len(all_pairs) // 20)
    val_pairs, _ = all_pairs[:n_val], all_pairs[n_val:]

    train_ds = ChatDataset(train_pairs, tokenizer, args.block_size)
    val_ds = ChatDataset(val_pairs, tokenizer, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    print(f"Train sequences: {len(train_ds)} (after oversampling) | val: {len(val_ds)}")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        pos_encoding="learned",
    )
    model = GPT(config).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)

    def save_checkpoint(step):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.as_dict(),
                "vocab": tokenizer.stoi,
                "kind": "chat",
                "step": step,
            },
            ckpt_path,
        )

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    train_iter = infinite_batches(train_loader)
    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            vl = val_loss()
            print(f"\nstep {step:5d} | lr {lr:.2e} | train loss {loss.item():.4f} | val loss {vl:.4f}")
            for pq in PROBES:
                print(f"    Q: {pq}\n    A: {sample_answer(model, tokenizer, pq, device)}")
            save_checkpoint(step)

    save_checkpoint(args.max_steps - 1)
    print(f"\nTraining complete. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
