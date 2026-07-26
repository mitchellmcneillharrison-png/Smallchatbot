"""
Download a text corpus for real pretraining (as opposed to the tiny hand-built
chatbot data). Streams the chosen dataset and writes up to `--max_mb` megabytes
to `data/pretrain.txt`, which `train.py` then trains on with the character-level
tokenizer.

    python data/get_pretrain_data.py --dataset tinystories --max_mb 200
    python data/get_pretrain_data.py --dataset tinyshakespeare
    python data/get_pretrain_data.py --url https://example.com/corpus.txt --max_mb 100

Datasets:
  * tinystories     -- simple synthetic children's stories, purpose-built so
                       that SMALL models learn to produce coherent English.
                       Ideal for a from-scratch GPT on a single Colab GPU.
  * tinyshakespeare -- ~1 MB of Shakespeare; a quick sanity-check corpus.

Only stdlib is used, so it runs anywhere (it honors HTTP(S)_PROXY env vars).
"""

import argparse
import os
import urllib.request

SOURCES = {
    "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "tinyshakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def main():
    p = argparse.ArgumentParser(description="Download a pretraining text corpus.")
    p.add_argument("--dataset", choices=list(SOURCES), default="tinystories")
    p.add_argument("--url", type=str, default=None, help="download from this URL instead of a named dataset")
    p.add_argument("--out", type=str, default="data/pretrain.txt")
    p.add_argument("--max_mb", type=int, default=200, help="stop after writing this many megabytes (0 = no cap)")
    args = p.parse_args()

    url = args.url or SOURCES[args.dataset]
    cap = args.max_mb * 1024 * 1024 if args.max_mb > 0 else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-gpt/1.0"})
    written = 0
    with urllib.request.urlopen(req) as resp, open(args.out, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            if cap is not None and written + len(chunk) > cap:
                f.write(chunk[: cap - written])
                written = cap
                break
            f.write(chunk)
            written += len(chunk)
            print(f"\r  {written / 1e6:.0f} MB", end="", flush=True)
    print(f"\nWrote {written / 1e6:.1f} MB to {args.out}")

    # Peek at the vocabulary size so you can sanity-check before training.
    with open(args.out, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(1_000_000)
    print(f"Distinct characters in first 1 MB: {len(set(head))}")


if __name__ == "__main__":
    main()
