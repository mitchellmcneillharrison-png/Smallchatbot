// app.js -- browser glue: load the exported model, wire up the controls, and
// stream generated text into the page. All the actual model maths lives in
// gpt.js; this file only touches the DOM.

import { TinyGPT } from "./gpt.js";

const $ = (id) => document.getElementById(id);
const els = {
  prompt: $("prompt"),
  temp: $("temp"), tempOut: $("tempOut"),
  topk: $("topk"), topkOut: $("topkOut"),
  len: $("len"), lenOut: $("lenOut"),
  go: $("go"), stop: $("stop"),
  status: $("status"), output: $("output"), badges: $("badges"),
};

// Live-update the little value readouts next to each slider.
const bindReadout = (input, out, fmt) => {
  const sync = () => (out.textContent = fmt(input.value));
  input.addEventListener("input", sync);
  sync();
};
bindReadout(els.temp, els.tempOut, (v) => Number(v).toFixed(2));
bindReadout(els.topk, els.topkOut, (v) => String(v));
bindReadout(els.len, els.lenOut, (v) => String(v));

let gpt = null;
let stopRequested = false;

async function loadModel() {
  try {
    const res = await fetch("./model.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const model = await res.json();
    gpt = new TinyGPT(model);

    const c = gpt.config;
    const nParams = Object.values(model.weights).reduce((s, t) => s + t.data.length, 0);
    els.badges.innerHTML = "";
    const badges = [
      `<b>${(nParams / 1000).toFixed(0)}K</b> params`,
      `<b>${c.n_layer}</b> layers`,
      `<b>${c.n_head}</b> heads`,
      `<b>${c.n_embd}</b> dim`,
      `<b>${c.vocab_size}</b> vocab (chars)`,
      `<b>${c.block_size}</b> ctx`,
      `${c.pos_encoding} pos-enc`,
    ];
    for (const b of badges) {
      const span = document.createElement("span");
      span.className = "badge";
      span.innerHTML = b;
      els.badges.appendChild(span);
    }

    els.go.disabled = false;
    els.status.textContent = "Ready.";
  } catch (err) {
    els.status.textContent = `Failed to load model.json (${err.message}).`;
    els.status.classList.add("err");
  }
}

function setRunning(running) {
  els.go.disabled = running;
  els.stop.disabled = !running;
  els.prompt.disabled = running;
}

async function run() {
  if (!gpt) return;
  stopRequested = false;
  setRunning(true);
  els.status.classList.remove("err");
  els.status.textContent = "Generating…";

  const prompt = els.prompt.value;
  // Render the prompt (highlighted) followed by a blinking caret we replace as
  // tokens stream in.
  els.output.innerHTML = "";
  const promptSpan = document.createElement("span");
  promptSpan.className = "prompt";
  promptSpan.textContent = prompt;
  const genSpan = document.createElement("span");
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.textContent = "▋";
  els.output.append(promptSpan, genSpan, caret);

  const started = performance.now();
  let count = 0;
  try {
    await gpt.generate(prompt, {
      maxNewTokens: Number(els.len.value),
      temperature: Number(els.temp.value),
      topK: Number(els.topk.value),
      onToken: (id) => {
        if (stopRequested) throw new StopSignal();
        genSpan.textContent += gpt.decode([id]);
        count++;
        els.output.scrollTop = els.output.scrollHeight;
      },
    });
  } catch (err) {
    if (!(err instanceof StopSignal)) {
      els.status.textContent = `Error: ${err.message}`;
      els.status.classList.add("err");
      caret.remove();
      setRunning(false);
      return;
    }
  }

  caret.remove();
  const secs = (performance.now() - started) / 1000;
  const rate = count > 0 ? (count / secs).toFixed(0) : "0";
  els.status.textContent = stopRequested
    ? `Stopped after ${count} tokens.`
    : `Done — ${count} tokens in ${secs.toFixed(1)}s (${rate} tok/s).`;
  setRunning(false);
}

class StopSignal extends Error {}

els.go.addEventListener("click", run);
els.stop.addEventListener("click", () => {
  stopRequested = true;
});

loadModel();
