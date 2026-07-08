"""
Export the trained chatbot checkpoint into web/model.json for the in-browser demo.

Same idea as export_web.py, but also ships the word-level tokenizer (vocabulary
+ special-token ids) so the JavaScript side can turn text into ids and back
exactly the way training did.

Usage:
    python export_chat_web.py --checkpoint checkpoints/chat.pt --out web/model.json
"""

import argparse
import base64
import json
import os

import torch

from src.chat_tokenizer import SPECIAL_TOKENS, WordTokenizer
from src.config import GPTConfig
from src.model import GPT


def tensor_to_list(t, sig: int = 6):
    flat = t.detach().cpu().float().reshape(-1).tolist()
    return [float(f"{x:.{sig}g}") for x in flat]


def tensor_to_f16_b64(t):
    """Quantize a tensor to float16 and base64-encode its raw little-endian
    bytes. Halves the download vs. float32 and shrinks it ~4x vs. JSON numbers,
    with negligible quality loss for a sampling demo. Decoded in web/gpt.js."""
    half = t.detach().cpu().to(torch.float16).contiguous()
    return base64.b64encode(half.numpy().tobytes()).decode("ascii")


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

    # Round every weight to float16-representable values IN PLACE, so the logits
    # the model produces here match what the browser (which loads float16) will
    # compute -- keeping web/verify.mjs's parity check honest.
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(torch.float16).float()

    # Pack all weights into ONE flat float16 binary blob (model.bin) and record
    # each tensor's shape + offset in a small JSON manifest. This is what makes
    # the page load fast: the browser fetches the blob as an ArrayBuffer (no
    # base64 inflation, no giant JSON.parse) and reads each weight as a typed-
    # array view -- see web/gpt.js and web/app.js.
    import numpy as np

    manifest = {}
    chunks = []
    offset = 0  # in float16 elements
    for name, tensor in model.state_dict().items():
        if name.endswith("causal_mask"):
            continue
        flat = tensor.detach().cpu().to(torch.float16).contiguous().numpy().reshape(-1)
        manifest[name] = {"shape": list(tensor.shape), "offset": offset, "n": int(flat.size)}
        chunks.append(flat)
        offset += flat.size

    blob = np.concatenate(chunks).astype("<f2").tobytes()  # little-endian float16
    bin_path = os.path.splitext(args.out)[0] + ".bin"
    with open(bin_path, "wb") as f:
        f.write(blob)

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
        # weights live in the companion .bin file, addressed by this manifest.
        "weights_bin": os.path.basename(bin_path),
        "weights": manifest,
        "self_check": {"input_ids": check_ids, "last_logits": tensor_to_list(logits[0, -1, :])},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    json_mb = os.path.getsize(args.out) / 1e6
    bin_mb = os.path.getsize(bin_path) / 1e6
    print(f"Wrote {args.out} ({json_mb:.2f} MB) + {bin_path} ({bin_mb:.2f} MB) | "
          f"vocab {tokenizer.vocab_size} | block {config.block_size}")


if __name__ == "__main__":
    main()
