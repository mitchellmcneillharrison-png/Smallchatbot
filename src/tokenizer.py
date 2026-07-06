"""
A minimal character-level tokenizer, built entirely from scratch (no external
tokenizer libraries). Real GPT models use sub-word Byte-Pair Encoding, but a
character-level vocabulary is trivial to implement correctly, has no
"unknown token" problem, and is perfect for learning how the rest of the
pipeline (embeddings -> model -> logits -> sampled text) fits together.

Swapping this out for a BPE tokenizer later would not require changing any
other module: everything downstream only depends on `vocab_size`, `encode`,
and `decode`.
"""

import json
from typing import Dict, List


class CharTokenizer:
    def __init__(self, stoi: Dict[str, int]):
        # stoi: "string to index" -- maps each character to a unique integer id
        self.stoi = stoi
        # itos: "index to string" -- the inverse mapping, used when decoding
        self.itos = {i: ch for ch, i in stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from every unique character that appears in `text`."""
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        return cls(stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token ids back into a string."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
        return cls(stoi)
