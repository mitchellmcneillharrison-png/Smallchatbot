"""
A word-level tokenizer with special tokens, for the chatbot.

The character-level tokenizer in `tokenizer.py` is great for the story demo, but
for a question-answering bot a word-level vocabulary is a much better fit: the
model works with whole words like "capital" or "plus" as single units, so it has
far less to learn and generalizes to reworded questions more readily.

Tokenization rule (kept deliberately simple so it can be replicated exactly in
JavaScript for the browser demo): lowercase the text, then pull out runs of
letters/digits, contraction suffixes like "'s", and individual punctuation
characters.

Special tokens frame a conversation so the model can tell the user's turn from
its own:
    <user>  ... the user's message ...
    <bot>   ... the model's reply ...
    <eos>   end of the reply (generation stops here)
    <pad>   padding to make batched sequences equal length
    <unk>   any word not seen during training
"""

import json
import re
from typing import Dict, List

# Matches: a word/number run, OR a contraction suffix ("'s", "'re", ...), OR a
# single punctuation character. Must stay identical to the regex in web/tokenizer.js.
_TOKEN_RE = re.compile(r"[a-z0-9]+|'[a-z]+|[^\sa-z0-9]")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<user>", "<bot>", "<eos>"]

# Punctuation that should hug the previous token when we join words back into a
# string (no leading space). Mirrored in the JS detokenizer.
_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "%", "'"}


def tokenize(text: str) -> List[str]:
    """Split raw text into word/punctuation tokens (lowercased)."""
    return _TOKEN_RE.findall(text.lower())


def detokenize(tokens: List[str]) -> str:
    """Join tokens back into a readable string, fixing spacing around
    punctuation and contraction suffixes."""
    out = []
    for i, tok in enumerate(tokens):
        if i > 0 and not (tok in _NO_SPACE_BEFORE or tok.startswith("'")):
            out.append(" ")
        out.append(tok)
    return "".join(out)


class WordTokenizer:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {i: t for t, i in stoi.items()}
        # Cache the special-token ids for convenience.
        for name in SPECIAL_TOKENS:
            setattr(self, name.strip("<>") + "_id", stoi[name])

    @classmethod
    def from_pairs(cls, pairs, min_freq: int = 1) -> "WordTokenizer":
        """Build a vocabulary from (question, answer) pairs. Special tokens get
        the first, fixed ids; ordinary words follow, sorted by frequency then
        alphabetically for determinism."""
        from collections import Counter

        counter = Counter()
        for q, a in pairs:
            counter.update(tokenize(q))
            counter.update(tokenize(a))

        stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        words = sorted(
            [w for w, c in counter.items() if c >= min_freq],
            key=lambda w: (-counter[w], w),
        )
        for w in words:
            if w not in stoi:
                stoi[w] = len(stoi)
        return cls(stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        """Text -> token ids, mapping unknown words to <unk>."""
        unk = self.stoi["<unk>"]
        return [self.stoi.get(tok, unk) for tok in tokenize(text)]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Token ids -> readable text, optionally dropping special tokens."""
        specials = set(SPECIAL_TOKENS)
        toks = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if skip_special and tok in specials:
                continue
            toks.append(tok)
        return detokenize(toks)

    def build_prompt(self, question: str) -> List[int]:
        """Format a user question as the model expects at inference time:
        <user> ...question... <bot>   (the model then generates the answer)."""
        return [self.stoi["<user>"]] + self.encode(question) + [self.stoi["<bot>"]]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))
