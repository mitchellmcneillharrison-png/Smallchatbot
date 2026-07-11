// verify.mjs -- confirms the JavaScript port in gpt.js computes the same thing
// as the PyTorch model. export_web.py bakes a `self_check` block into
// model.json: a fixed input sequence and the logits PyTorch produced for its
// final position. Here we run that same input through the KV-cached JS forward
// and assert the logits match.
//
//   node web/verify.mjs [path/to/model.json]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { TinyGPT, inflateFloat16, dequantizeInt8 } from "./gpt.js";

const path = process.argv[2] || new URL("./model.json", import.meta.url).pathname;
const model = JSON.parse(readFileSync(path, "utf-8"));

// Weights ship in a companion .bin, addressed by the manifest in
// model.weights[name]. Reconstruct each weight to a Float32Array, mirroring the
// browser loader in app.js. Supports int8 (weights + per-row scales) and legacy
// float16; older exports inlined weights in the JSON (passed straight through).
if (model.weights_bin) {
  const raw = readFileSync(join(dirname(path), model.weights_bin));
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength); // aligned copy
  const weights = {};
  if (model.quant === "int8") {
    const int8 = new Int8Array(ab, 0, model.scales_byte_offset);
    const scales = new Float32Array(ab, model.scales_byte_offset);
    for (const [name, w] of Object.entries(model.weights)) {
      weights[name] = dequantizeInt8(
        int8.subarray(w.offset, w.offset + w.n),
        scales.subarray(w.srow, w.srow + w.nrows),
        w.nrows
      );
    }
  } else {
    const u16 = new Uint16Array(ab);
    for (const [name, w] of Object.entries(model.weights)) {
      weights[name] = inflateFloat16(u16.subarray(w.offset, w.offset + w.n));
    }
  }
  model.weights = weights;
}
const gpt = new TinyGPT(model);

// itos lives under model.tokenizer for the chat model (or top-level for the
// older char model); used only for a friendly print of the argmax token.
const itos = model.tokenizer?.itos || model.itos || [];

const { input_ids, last_logits } = model.self_check;

// Feed the exact input through the cached forward; keep the final logits.
gpt.resetCache();
let logits;
for (const id of input_ids) logits = gpt.stepToken(id);

let maxAbsDiff = 0;
for (let i = 0; i < last_logits.length; i++) {
  maxAbsDiff = Math.max(maxAbsDiff, Math.abs(logits[i] - last_logits[i]));
}

// Also check the argmax (greedy next token) agrees -- the property that most
// directly affects generation quality.
const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
const jsArg = argmax(logits);
const pyArg = argmax(last_logits);

console.log(`positions fed:   ${input_ids.length}`);
console.log(`max |Δ logit|:   ${maxAbsDiff.toExponential(3)}`);
console.log(`argmax JS / PT:  ${jsArg} / ${pyArg} (${JSON.stringify(itos[jsArg])})`);

// Small numeric differences are expected (float32 order-of-operations, the erf
// approximation in GELU). A few thousandths is well within tolerance.
const TOL = 2e-2;
if (maxAbsDiff > TOL || jsArg !== pyArg) {
  console.error(`\nFAIL: JS port diverges from PyTorch (tol ${TOL}).`);
  process.exit(1);
}
console.log("\nPASS: JS port matches PyTorch.");
