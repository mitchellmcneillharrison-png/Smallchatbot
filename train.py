"""
Training loop for the from-scratch GPT model.

Usage:
    python train.py                                   # train with defaults on data/sample.txt
    python train.py --max_steps 5000 --n_layer 6       # override any hyperparameter
    python train.py --resume checkpoints/ckpt.pt        # continue training from a checkpoint

Everything needed to reconstruct and keep using the model -- weights, optimizer
state, architecture config, and the tokenizer's vocabulary -- is saved together
in a single checkpoint file, so generate.py never needs to know the training
hyperparameters in advance.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.config import GPTConfig
from src.dataset import CharDataset
from src.model import GPT
from src.tokenizer import CharTokenizer
from src.utils import get_device, get_lr, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny GPT-style language model from scratch.")
    p.add_argument("--data_path", type=str, default="data/sample.txt")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")

    # architecture
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos_encoding", type=str, default="learned", choices=["learned", "sinusoidal"])

    # optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # logging / evaluation
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def infinite_batches(loader: DataLoader):
    """DataLoader iterators are exhausted after one epoch; this wraps around
    forever so the training loop can run for an arbitrary number of steps."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def estimate_loss(model, loaders, eval_iters, device):
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        it = infinite_batches(loader)
        for i in range(eval_iters):
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- data & tokenizer -------------------------------------------------
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    train_ds = CharDataset(train_data, args.block_size)
    val_ds = CharDataset(val_data, args.block_size)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True),
    }
    print(f"Vocab size: {tokenizer.vocab_size} | train chars: {len(train_data)} | val chars: {len(val_data)}")

    # ---- model & optimizer -------------------------------------------------
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        pos_encoding=args.pos_encoding,
    )
    model = GPT(config).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1

    def save_checkpoint(step, path):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.as_dict(),
                "vocab": tokenizer.stoi,
                "step": step,
            },
            path,
        )

    # ---- training loop -------------------------------------------------
    train_iter = infinite_batches(loaders["train"])
    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, loaders, args.eval_iters, device)
            print(
                f"step {step:6d} | lr {lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
            )
            ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
            save_checkpoint(step, ckpt_path)

    final_path = os.path.join(args.out_dir, "ckpt.pt")
    save_checkpoint(args.max_steps - 1, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
