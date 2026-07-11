// app.js -- browser glue for the chatbot. Loads the exported model + tokenizer,
// renders a chat UI, and streams the model's answer word by word. All model
// maths is in gpt.js; all tokenization is in tokenizer.js.

import { TinyGPT, inflateFloat16, dequantizeInt8 } from "./gpt.js";
import { WordTokenizer } from "./tokenizer.js";

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const input = $("input");
const sendBtn = $("send");

let gpt = null;
let tok = null;
let busy = false;

// Fetch a URL as an ArrayBuffer while reporting download progress, so the user
// sees a moving bar instead of a frozen page during the one-time model load.
async function fetchBufferWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !res.body.getReader) return await res.arrayBuffer(); // fallback
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.length;
  }
  return out.buffer;
}

function setProgress(fraction, label) {
  const fill = $("loadbar-fill");
  if (fill) fill.style.width = `${Math.round(fraction * 100)}%`;
  const hint = $("hint");
  if (hint) hint.textContent = label;
}

async function loadModel() {
  try {
    const meta = await (await fetch("./model.json")).json(); // small: config + tokenizer + manifest
    const c = meta.config;

    // Download the weights blob with a progress bar.
    const buf = await fetchBufferWithProgress("./" + (meta.weights_bin || "model.bin"), (r, t) => {
      const frac = t ? r / t : 0;
      setProgress(frac, t ? `Loading model… ${Math.round(frac * 100)}%` : `Loading model… ${(r / 1e6).toFixed(1)} MB`);
    });

    // Decode the blob into a Float32Array per weight (fast, typed-array views).
    setProgress(1, "Preparing model…");
    const weights = {};
    let nParams = 0;
    if (meta.quant === "int8") {
      // Blob layout: [int8 weights][float32 per-row scales].
      const int8 = new Int8Array(buf, 0, meta.scales_byte_offset);
      const scales = new Float32Array(buf, meta.scales_byte_offset);
      for (const [name, w] of Object.entries(meta.weights)) {
        weights[name] = dequantizeInt8(
          int8.subarray(w.offset, w.offset + w.n),
          scales.subarray(w.srow, w.srow + w.nrows),
          w.nrows
        );
        nParams += w.n;
      }
    } else {
      const u16 = new Uint16Array(buf); // legacy float16 export
      for (const [name, w] of Object.entries(meta.weights)) {
        weights[name] = inflateFloat16(u16.subarray(w.offset, w.offset + w.n));
        nParams += w.n;
      }
    }

    gpt = new TinyGPT({ config: c, tokenizer: meta.tokenizer, weights, self_check: meta.self_check });
    tok = new WordTokenizer(meta.tokenizer);

    const badges = [
      `<b>${(nParams / 1e6).toFixed(1)}M</b> params`,
      `<b>${c.n_layer}</b> layers`,
      `<b>${c.n_head}</b> heads`,
      `<b>${c.n_embd}</b> dim`,
      `<b>${c.vocab_size}</b> word vocab`,
      `<b>${c.block_size}</b> ctx`,
    ];
    $("badges").innerHTML = badges.map((b) => `<span class="badge">${b}</span>`).join("");

    const loadbar = $("loadbar");
    if (loadbar) loadbar.style.display = "none";
    $("hint").textContent = "Ask me a question to get started.";
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  } catch (err) {
    $("hint").textContent = `Failed to load model (${err.message}).`;
  }
}

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

async function ask(question) {
  if (!gpt || busy || !question.trim()) return;
  busy = true;
  input.disabled = true;
  sendBtn.disabled = true;
  const hint = $("hint");
  if (hint) hint.remove();

  addMessage("user", question);
  const botEl = addMessage("bot", "");
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.textContent = "▋";
  botEl.appendChild(caret);

  const promptIds = tok.buildPrompt(question);
  const answerIds = [];
  try {
    await gpt.generateIds(promptIds, {
      maxNewTokens: gpt.config.block_size,
      temperature: 0.4,
      topK: 20,
      stopToken: tok.specials.eos,
      onToken: (id) => {
        answerIds.push(id);
        // Re-decode the whole answer each token so word spacing stays correct.
        botEl.textContent = tok.decode(answerIds);
        botEl.appendChild(caret);
        chat.scrollTop = chat.scrollHeight;
      },
    });
  } catch (err) {
    botEl.textContent = `Error: ${err.message}`;
  }
  caret.remove();
  if (!botEl.textContent.trim()) botEl.textContent = "…";

  busy = false;
  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.addEventListener("click", () => {
  const q = input.value;
  input.value = "";
  ask(q);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    ask(q);
  }
});
$("suggestions").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip") && !busy) ask(e.target.textContent);
});

loadModel();
