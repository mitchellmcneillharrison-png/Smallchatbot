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

    # Quantize every weight to per-row int8 and pack into ONE binary blob
    # (model.bin): first an int8 section (the weights), then a float32 section
    # (the per-row scales). A tiny JSON manifest records each tensor's shape and
    # offsets. This is ~1 byte/param -- half the float16 size -- and the browser
    # reads it as typed-array views (no giant JSON.parse), so the page loads
    # fast. See web/gpt.js (dequantizeInt8), web/app.js, and web/verify.mjs.
    #
    # The model's own weights are replaced in place with the dequantized values
    # first, so the self-check logits match exactly what the browser computes.
    import numpy as np

    from src.utils import quantize_tensor_int8

    manifest = {}
    int8_chunks, scale_chunks = [], []
    offset = 0        # int8 element (== byte) offset
    scale_offset = 0  # float32 scale index
    deq_by_name = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("causal_mask"):
            continue
        q, scales, nrows, deq = quantize_tensor_int8(tensor)
        deq_by_name[name] = deq
        manifest[name] = {"shape": list(tensor.shape), "offset": offset, "n": int(q.size),
                          "srow": scale_offset, "nrows": int(nrows)}
        int8_chunks.append(q.astype(np.int8))
        scale_chunks.append(scales)
        offset += q.size
        scale_offset += nrows

    int8_blob = np.concatenate(int8_chunks).astype(np.int8).tobytes()
    # pad the int8 section to a 4-byte boundary so the float32 scales that follow
    # are aligned for a typed-array view.
    pad = (-len(int8_blob)) % 4
    int8_blob += b"\x00" * pad
    scales_blob = np.concatenate(scale_chunks).astype("<f4").tobytes()
    scales_byte_offset = len(int8_blob)

    bin_path = os.path.splitext(args.out)[0] + ".bin"
    with open(bin_path, "wb") as f:
        f.write(int8_blob)
        f.write(scales_blob)

    # Replace weights with their dequantized values, then compute the self-check.
    with torch.no_grad():
        sd = model.state_dict()
        for name, deq in deq_by_name.items():
            sd[name].copy_(deq.to(sd[name].dtype))

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
        # weights live in the companion .bin file (int8 + float32 scales),
        # addressed by this manifest.
        "quant": "int8",
        "weights_bin": os.path.basename(bin_path),
        "scales_byte_offset": scales_byte_offset,
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
