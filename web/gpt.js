// gpt.js -- a pure-JavaScript re-implementation of the tiny GPT forward pass.
//
// This is a hand port of src/model/gpt.py: the SAME maths (embeddings,
// positional encoding, multi-head causal self-attention, LayerNorm,
// feed-forward, residual connections, weight-tied output head) re-expressed in
// plain JS over Float32Array. No ML libraries, no WASM -- just loops, so you
// can read it alongside the Python and see they compute the same thing.
//
// For speed it uses an incremental KV cache: instead of re-running the whole
// context every step (as gpt.py's reference generate() does), each new token
// computes only its own query/key/value and attends over the cached keys and
// values of all previous positions. Because attention is causal, this is
// mathematically identical to a full forward pass -- verified against PyTorch
// in web/verify.mjs.
//
// It has no browser or Node dependencies, so the same file powers both the web
// UI (web/app.js) and the correctness test (web/verify.mjs).

// ---- weight decoding -------------------------------------------------------

// Convert an IEEE-754 half-precision (float16) bit pattern to a JS number.
function halfBitsToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024); // subnormal / zero
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

// Decode a base64 string of little-endian float16 values into a Float32Array.
// `atob` exists in both browsers and Node 22, so this file stays portable.
function decodeFloat16Base64(b64) {
  const bin = atob(b64);
  const count = bin.length >> 1;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const bits = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    out[i] = halfBitsToFloat(bits);
  }
  return out;
}

// ---- small numeric helpers -------------------------------------------------

// erf approximation (Abramowitz & Stegun 7.1.26), max abs error ~1.5e-7.
// Needed because PyTorch's default nn.GELU() is the exact (erf-based) variant.
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return sign * y;
}

// Exact GELU, matching torch.nn.GELU(): 0.5 * x * (1 + erf(x / sqrt(2))).
function gelu(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

// y = x @ W^T + b, where W is stored flat, row-major, shape [outDim, inDim]
// (PyTorch's native nn.Linear layout). b may be null (no bias).
function linear(x, W, b, outDim, inDim) {
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let acc = b ? b[o] : 0;
    const base = o * inDim;
    for (let i = 0; i < inDim; i++) acc += x[i] * W[base + i];
    y[o] = acc;
  }
  return y;
}

// LayerNorm over the last dimension (population variance, unbiased=False),
// matching src/model/layers.py.
function layerNorm(x, weight, bias, n, eps = 1e-5) {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;
  const inv = 1 / Math.sqrt(variance + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = (x[i] - mean) * inv * weight[i];
    if (bias) v += bias[i];
    out[i] = v;
  }
  return out;
}

// Numerically-stable softmax over the first `len` entries of `arr`, in place.
function softmaxInPlace(arr, len) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) if (arr[i] > max) max = arr[i];
  let sum = 0;
  for (let i = 0; i < len; i++) {
    arr[i] = Math.exp(arr[i] - max);
    sum += arr[i];
  }
  for (let i = 0; i < len; i++) arr[i] /= sum;
}

// ---- the model -------------------------------------------------------------

export class TinyGPT {
  constructor(model) {
    this.config = model.config;

    // Convert every weight into a Float32Array once, up front, so the hot
    // generation loop touches only typed arrays. Weights ship either as a plain
    // JSON number[] (`data`, older exports) or as base64-encoded float16
    // (`b64`, compact chat export) -- support both.
    this.w = {};
    for (const [name, tensor] of Object.entries(model.weights)) {
      this.w[name] = tensor.b64 !== undefined
        ? decodeFloat16Base64(tensor.b64)
        : Float32Array.from(tensor.data);
    }

    const c = this.config;
    this.C = c.n_embd;
    this.nHead = c.n_head;
    this.headDim = c.n_embd / c.n_head;
    this.vocab = c.vocab_size;

    // Precompute sinusoidal positional encodings if this model uses them, so
    // the learned-vs-sinusoidal choice is transparent to the step function.
    if (c.pos_encoding === "sinusoidal") {
      this.sinPos = this._buildSinusoidal(c.block_size, c.n_embd);
    }

    this.resetCache();
  }

  resetCache() {
    // Per-layer key/value caches; each is an array (indexed by position) of
    // Float32Array(n_embd) holding that position's concatenated-head k or v.
    this.kCache = Array.from({ length: this.config.n_layer }, () => []);
    this.vCache = Array.from({ length: this.config.n_layer }, () => []);
    this.pos = 0; // next position index to be filled
  }

  _buildSinusoidal(blockSize, nEmbd) {
    // Mirror of sinusoidal_positional_encoding() in src/model/positional.py.
    const pe = [];
    for (let p = 0; p < blockSize; p++) {
      const row = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i += 2) {
        const divTerm = Math.exp((i * -Math.log(10000.0)) / nEmbd);
        row[i] = Math.sin(p * divTerm);
        if (i + 1 < nEmbd) row[i + 1] = Math.cos(p * divTerm);
      }
      pe.push(row);
    }
    return pe;
  }

  _positionEmbedding(pos) {
    const C = this.C;
    if (this.config.pos_encoding === "sinusoidal") return this.sinPos[pos];
    // Learned: row `pos` of the positional embedding table.
    const table = this.w["pos_encoding.pos_emb.weight"];
    return table.subarray(pos * C, pos * C + C);
  }

  // Run ONE token through the whole network at the current cached position and
  // return its vocab-sized logits. Advances the KV cache by one position.
  stepToken(tokenId) {
    const { C, nHead, headDim, vocab } = this;
    const pos = this.pos;
    if (pos >= this.config.block_size)
      throw new Error(`Context length ${this.config.block_size} exceeded`);

    // token embedding + positional embedding (residual stream starts here)
    const tokRow = this.w["token_emb.weight"].subarray(tokenId * C, tokenId * C + C);
    const posRow = this._positionEmbedding(pos);
    let x = new Float32Array(C);
    for (let i = 0; i < C; i++) x[i] = tokRow[i] + posRow[i];

    const w = this.w;
    for (let L = 0; L < this.config.n_layer; L++) {
      const p = `blocks.${L}.`;

      // --- causal self-attention (pre-norm) ---
      const h = layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"], C);
      const qkv = linear(h, w[p + "attn.qkv_proj.weight"], w[p + "attn.qkv_proj.bias"], 3 * C, C);
      const q = qkv.subarray(0, C);
      const k = qkv.slice(C, 2 * C); // slice -> owns its memory, safe to cache
      const v = qkv.slice(2 * C, 3 * C);

      this.kCache[L].push(k);
      this.vCache[L].push(v);
      const T = this.kCache[L].length; // number of positions to attend over (<= pos+1)

      const attnOut = new Float32Array(C);
      const scale = 1 / Math.sqrt(headDim);
      const scores = new Float32Array(T);
      for (let hIdx = 0; hIdx < nHead; hIdx++) {
        const off = hIdx * headDim;
        // scores against every cached key for this head
        for (let j = 0; j < T; j++) {
          const kj = this.kCache[L][j];
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[off + d] * kj[off + d];
          scores[j] = dot * scale;
        }
        softmaxInPlace(scores, T);
        // weighted sum of values
        for (let j = 0; j < T; j++) {
          const vj = this.vCache[L][j];
          const s = scores[j];
          for (let d = 0; d < headDim; d++) attnOut[off + d] += s * vj[off + d];
        }
      }
      const attnProj = linear(attnOut, w[p + "attn.out_proj.weight"], w[p + "attn.out_proj.bias"], C, C);
      for (let i = 0; i < C; i++) x[i] += attnProj[i]; // residual

      // --- feed-forward (pre-norm) ---
      const h2 = layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"], C);
      const ff = linear(h2, w[p + "ffwd.fc_in.weight"], w[p + "ffwd.fc_in.bias"], 4 * C, C);
      for (let i = 0; i < 4 * C; i++) ff[i] = gelu(ff[i]);
      const ffOut = linear(ff, w[p + "ffwd.fc_out.weight"], w[p + "ffwd.fc_out.bias"], C, 4 * C);
      for (let i = 0; i < C; i++) x[i] += ffOut[i]; // residual
    }

    const xf = layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], C);
    // Output head: weight-tied with the token embedding, so lm_head.weight and
    // token_emb.weight are identical; either key works.
    const logits = linear(xf, w["lm_head.weight"], null, vocab, C);

    this.pos += 1;
    return logits;
  }

  // Sample one token id from logits, applying temperature and optional top-k,
  // exactly like GPT.generate() in src/model/gpt.py. `rng` defaults to
  // Math.random but can be injected for deterministic tests.
  sample(logits, temperature = 1.0, topK = 0, rng = Math.random) {
    const n = logits.length;
    const scaled = new Float32Array(n);
    const t = Math.max(temperature, 1e-6);
    for (let i = 0; i < n; i++) scaled[i] = logits[i] / t;

    let allowed = null;
    if (topK && topK < n) {
      // Find the top-k threshold; disallow everything below it.
      const sorted = Array.from(scaled).sort((a, b) => b - a);
      const thresh = sorted[topK - 1];
      allowed = (i) => scaled[i] >= thresh;
    }

    const probs = new Float32Array(n);
    for (let i = 0; i < n; i++) probs[i] = allowed && !allowed(i) ? -Infinity : scaled[i];
    softmaxInPlace(probs, n);

    // inverse-CDF sampling
    let r = rng();
    for (let i = 0; i < n; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return n - 1;
  }

  // Warm the KV cache with the prompt token ids, then autoregressively generate.
  // Stops when `stopToken` (e.g. <eos>) is sampled or the budget runs out.
  // Calls `onToken(id)` for each newly generated token so callers can stream.
  async generateIds(
    promptIds,
    { maxNewTokens = 60, temperature = 0.8, topK = 40, stopToken = null, onToken, rng } = {}
  ) {
    this.resetCache();
    // This simplified KV cache does not slide its window, so the prompt must
    // leave room for at least one generated token within block_size.
    if (promptIds.length > this.config.block_size - 1) {
      promptIds = promptIds.slice(-(this.config.block_size - 1));
    }

    let logits;
    for (const id of promptIds) logits = this.stepToken(id);

    const generated = [];
    const budget = Math.min(maxNewTokens, this.config.block_size - this.pos);
    for (let i = 0; i < budget; i++) {
      const nextId = this.sample(logits, temperature, topK, rng);
      if (nextId === stopToken) break;
      generated.push(nextId);
      if (onToken) {
        onToken(nextId);
        if (i % 2 === 0) await new Promise((r) => setTimeout(r, 0)); // let the UI paint
      }
      logits = this.stepToken(nextId);
    }
    return generated;
  }
}
