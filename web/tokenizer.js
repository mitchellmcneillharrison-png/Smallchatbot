// tokenizer.js -- word-level tokenizer for the browser, a direct port of
// src/chat_tokenizer.py. The tokenization regex and detokenization spacing
// rules MUST match the Python version exactly, or the ids the model sees in the
// browser won't line up with the ones it was trained on.

// Runs of letters/digits, OR a contraction suffix ("'s"), OR a single
// punctuation character. Identical to _TOKEN_RE in chat_tokenizer.py.
const TOKEN_RE = /[a-z0-9]+|'[a-z]+|[^\sa-z0-9]/g;

// Punctuation / suffixes that hug the previous token (no leading space).
const NO_SPACE_BEFORE = new Set([".", ",", "!", "?", ";", ":", ")", "%", "'"]);

export function tokenize(text) {
  return (text.toLowerCase().match(TOKEN_RE)) || [];
}

export function detokenize(tokens) {
  let out = "";
  tokens.forEach((tok, i) => {
    if (i > 0 && !(NO_SPACE_BEFORE.has(tok) || tok.startsWith("'"))) out += " ";
    out += tok;
  });
  return out;
}

export class WordTokenizer {
  // `spec` is the model.json `tokenizer` object: { itos, stoi, specials, special_list }.
  constructor(spec) {
    this.itos = spec.itos;
    this.stoi = spec.stoi;
    this.specials = spec.specials; // { pad, unk, user, bot, eos }
    this.specialSet = new Set(spec.special_list || ["<pad>", "<unk>", "<user>", "<bot>", "<eos>"]);
  }

  encode(text) {
    const unk = this.specials.unk;
    return tokenize(text).map((tok) => {
      const id = this.stoi[tok];
      return id === undefined ? unk : id;
    });
  }

  // Format a user question the way training did: <user> ...q... <bot>
  buildPrompt(question) {
    return [this.specials.user, ...this.encode(question), this.specials.bot];
  }

  decode(ids, skipSpecial = true) {
    const toks = [];
    for (const id of ids) {
      const tok = this.itos[id] ?? "<unk>";
      if (skipSpecial && this.specialSet.has(tok)) continue;
      toks.push(tok);
    }
    return detokenize(toks);
  }
}
