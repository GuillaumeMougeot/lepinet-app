// UI controller for lepinet-app. Home screen → capture → result, per the proposal in
// lepinet/journal/2026-07-lepi-app.md. Vanilla ES modules, no framework (the deploy box has no
// JS toolchain — this ships as static files that GitHub Pages serves directly).

import { loadModel, preprocess, predict, gbifUrl, isReady } from './infer.js';

const el = (id) => document.getElementById(id);
const home = el('home');
const result = el('result');
const status = el('status');

const LEVEL_LABEL = { species: 'Species', genus: 'Genus', family: 'Family' };

// Kick off model loading immediately so the first capture isn't the slow one.
let loadPromise = loadModel('./model/', (msg) => { status.textContent = msg; })
  .then(() => {
    status.textContent = '';
    el('camera-btn').disabled = false;
    el('gallery-btn').disabled = false;
  })
  .catch((e) => { status.textContent = 'Model failed to load: ' + e.message; });

el('camera-btn').addEventListener('click', () => el('camera-input').click());
el('gallery-btn').addEventListener('click', () => el('gallery-input').click());
el('camera-input').addEventListener('change', onFile);
el('gallery-input').addEventListener('change', onFile);
el('back-btn').addEventListener('click', showHome);
window.addEventListener('popstate', showHome);

async function onFile(ev) {
  const file = ev.target.files?.[0];
  ev.target.value = ''; // allow re-picking the same file
  if (!file) return;
  await handleImage(file);
}

async function handleImage(file) {
  showResult();
  el('predictions').innerHTML = '';
  el('result-status').textContent = 'Analysing…';

  const bitmap = await createImageBitmap(file);
  const { tensor, previewCanvas } = preprocess(bitmap);
  showPreview(previewCanvas);

  if (!isReady()) { el('result-status').textContent = 'Loading model…'; await loadPromise; }

  const t0 = performance.now();
  let preds;
  try {
    preds = await predict(tensor);
  } catch (e) {
    el('result-status').textContent = 'Inference failed: ' + e.message;
    return;
  }
  const ms = Math.round(performance.now() - t0);
  render(preds, ms);
}

function showPreview(canvas) {
  const holder = el('preview');
  holder.innerHTML = '';
  canvas.classList.add('preview-img');
  holder.appendChild(canvas);
}

function render(preds, ms) {
  el('result-status').textContent = '';
  const anyAbove = preds.some((p) => p.aboveThreshold);
  const box = el('predictions');
  box.innerHTML = '';

  if (!anyAbove) {
    const note = document.createElement('p');
    note.className = 'lowconf-note';
    note.textContent =
      'Low confidence — this may not be a moth or butterfly, or may be a species outside the training set.';
    box.appendChild(note);
  }

  // finest→coarsest is species,genus,family; display family→genus→species (coarse on top)
  for (const p of [...preds].reverse()) {
    box.appendChild(rowFor(p));
  }
  el('timing').textContent = `${ms} ms · on-device`;
}

function rowFor(p) {
  const pct = Math.min(99, Math.round(p.prob * 100));
  const row = document.createElement('div');
  row.className = 'tax-row' + (p.aboveThreshold ? '' : ' greyed');

  const ring = confidenceRing(pct);

  const mid = document.createElement('div');
  mid.className = 'tax-mid';
  const label = document.createElement('div');
  label.className = 'tax-level';
  label.textContent = LEVEL_LABEL[p.level];
  const name = document.createElement('div');
  name.className = 'tax-name';
  const display = p.name || p.key; // scientific name if the bundle ships one, else the GBIF key
  name.textContent = display;
  if (p.level !== 'family') name.style.fontStyle = 'italic'; // binomials/genera are italicised
  name.title = display;
  mid.append(label, name);

  const actions = document.createElement('div');
  actions.className = 'tax-actions';
  const copy = document.createElement('button');
  copy.className = 'icon-btn'; copy.title = 'Copy name';
  copy.textContent = '⧉';
  copy.addEventListener('click', () => navigator.clipboard?.writeText(display));
  const link = document.createElement('a');
  link.className = 'icon-btn'; link.title = 'Open GBIF page';
  link.href = gbifUrl(p.key); link.target = '_blank'; link.rel = 'noopener';
  link.textContent = '↗';
  actions.append(copy, link);

  row.append(ring, mid, actions);
  return row;
}

// A 0–99 number inside a ring that fills red→green with confidence.
function confidenceRing(pct) {
  const hue = Math.round((pct / 99) * 120); // 0=red, 120=green
  const wrap = document.createElement('div');
  wrap.className = 'ring';
  wrap.style.background = `conic-gradient(hsl(${hue} 70% 45%) ${pct}%, var(--ring-bg) 0)`;
  const inner = document.createElement('span');
  inner.className = 'ring-inner';
  inner.textContent = String(pct).padStart(2, '0');
  wrap.appendChild(inner);
  return wrap;
}

function showResult() {
  home.hidden = true; result.hidden = false;
  if (history.state?.view !== 'result') history.pushState({ view: 'result' }, '');
}
function showHome() {
  result.hidden = true; home.hidden = false;
}
