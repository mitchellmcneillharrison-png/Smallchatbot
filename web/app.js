// app.js -- browser glue for the chatbot. Loads the exported model + tokenizer,
// renders a chat UI, and streams the model's answer word by word. All model
// maths is in gpt.js; all tokenization is in tokenizer.js.

import { TinyGPT } from "./gpt.js";
import { WordTokenizer } from "./tokenizer.js";

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const input = $("input");
const sendBtn = $("send");

let gpt = null;
let tok = null;
let busy = false;

async function loadModel() {
  try {
    const res = await fetch("./model.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const model = await res.json();
    gpt = new TinyGPT(model);
    tok = new WordTokenizer(model.tokenizer);

    const c = model.config;
    const nParams = Object.values(model.weights).reduce(
      (s, t) => s + t.shape.reduce((a, b) => a * b, 1), 0
    );
    const badges = [
      `<b>${(nParams / 1e6).toFixed(1)}M</b> params`,
      `<b>${c.n_layer}</b> layers`,
      `<b>${c.n_head}</b> heads`,
      `<b>${c.n_embd}</b> dim`,
      `<b>${c.vocab_size}</b> word vocab`,
      `<b>${c.block_size}</b> ctx`,
    ];
    $("badges").innerHTML = badges.map((b) => `<span class="badge">${b}</span>`).join("");

    $("hint").textContent = "Ask me a question to get started.";
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  } catch (err) {
    $("hint").textContent = `Failed to load model.json (${err.message}).`;
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
