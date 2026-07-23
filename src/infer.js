// Inference core for lepinet-app.
//
// Loads a model bundle described by model/config.json, preprocesses an image the way the
// training pipeline's validation transform does, runs the model, and derives a consistent
// species/genus/family prediction. Everything runs on-device; no image leaves the phone.
//
// The bundle is model-agnostic: point config.json at a different model.onnx + taxonomy.json and
// the app runs it unchanged (see DEVELOPER.md). Two decisions come from the lepinet compression
// experiments (lepinet/journal/2026-07-lepi-app-compression.md):
//   * Preprocessing = shorter-side resize to `imageSize` + center crop (dev/041: matches fastai
//     validation within 0.1 pp; only aspect ratio matters, so a plain canvas draw is fine).
//   * Genus/family are marginalized from the species posterior (dev/042: more accurate AND
//     consistent), so the three levels can never contradict each other.
//
// The model must emit raw logits for each level and bake its own input normalization into the
// graph (input = RGB [1,3,S,S] in [0,1]).

import * as ort from '../ort/ort.webgpu.mjs';

ort.env.wasm.wasmPaths = new URL('../ort/', import.meta.url).href;
// GitHub Pages is not cross-origin isolated (no COOP/COEP) → no SharedArrayBuffer/threads.
ort.env.wasm.numThreads = 1;

let session = null;
let cfg = null;
let taxonomy = null;
let names = null;         // { species:[…], genus:[…], family:[…] } aligned to vocab order, or null
let calibration = null;   // { temperatures: {…} } or null
let thresholds = null;    // { species, genus, family } or null

const LEVELS = ['species', 'genus', 'family'];

async function getJSON(url) {
  return fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}

/** Load the bundle described by `<base>config.json`. Idempotent. */
export async function loadModel(base = './model/', onProgress = () => {}) {
  if (session) return;
  onProgress('Loading bundle…');
  cfg = (await getJSON(base + 'config.json')) || {};
  cfg.imageSize = cfg.imageSize || 256;
  cfg.inputName = cfg.inputName || 'image';
  cfg.outputs = cfg.outputs || { species: 'logits_species', genus: 'logits_genus', family: 'logits_family' };
  cfg.gbifBase = cfg.gbifBase || 'https://www.gbif.org/species/';

  taxonomy = await getJSON(base + (cfg.taxonomy || 'taxonomy.json'));
  if (!taxonomy) throw new Error('taxonomy.json missing or invalid');
  const namesDoc = cfg.names ? await getJSON(base + cfg.names) : null;
  names = namesDoc?.names || null;
  calibration = cfg.calibration ? await getJSON(base + cfg.calibration) : null;
  const thr = cfg.thresholds ? await getJSON(base + cfg.thresholds) : null;
  if (thr?.levels) {
    thresholds = Object.fromEntries(LEVELS.map((l) => [l, thr.levels[l]?.threshold ?? null]));
  }

  onProgress('Loading model…');
  // Primary model (config.model) first; if it can't run on this device, fall back to
  // config.fallback (e.g. the fp32 model on the GitHub release — larger but universally
  // supported). Within each, prefer WebGPU then WASM-only.
  const modelUrls = [base + (cfg.model || 'model.onnx'), cfg.fallback].filter(Boolean);
  session = await createSession(modelUrls, onProgress);

  onProgress('Warming up…');
  await warmup(); // pay the first-inference graph-init cost now, not on the user's first photo
  onProgress('Ready');
}

async function createSession(modelUrls, onProgress) {
  let lastErr;
  for (let i = 0; i < modelUrls.length; i++) {
    if (i > 0) onProgress('Retrying with fallback model…');
    // Prefer WebGPU (fast) but fall back to WASM-only — some browsers report navigator.gpu yet
    // fail to create a context, and QDQ-quantized ops may not run on the GPU backend.
    const attempts = [];
    if (navigator.gpu) attempts.push(['webgpu', 'wasm']);
    attempts.push(['wasm']);
    for (const executionProviders of attempts) {
      try {
        return await ort.InferenceSession.create(modelUrls[i], {
          executionProviders, graphOptimizationLevel: 'all',
        });
      } catch (err) { lastErr = err; }
    }
  }
  throw lastErr;
}

async function warmup() {
  const S = cfg.imageSize;
  const dummy = new ort.Tensor('float32', new Float32Array(3 * S * S), [1, 3, S, S]);
  await session.run({ [cfg.inputName]: dummy });
}

/**
 * Preprocess an ImageBitmap/HTMLImageElement to a [1,3,S,S] float tensor in [0,1] RGB, via
 * shorter-side resize + center crop. Returns { tensor, previewCanvas }.
 */
export function preprocess(img) {
  const S = cfg.imageSize;
  const w = img.width, h = img.height;
  const scale = S / Math.min(w, h);
  const nw = Math.max(S, Math.round(w * scale));
  const nh = Math.max(S, Math.round(h * scale));

  const c = document.createElement('canvas');
  c.width = S; c.height = S;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, (S - nw) / 2, (S - nh) / 2, nw, nh); // resize + center-crop in one draw

  const { data } = ctx.getImageData(0, 0, S, S);
  const chw = new Float32Array(3 * S * S);
  const plane = S * S;
  for (let i = 0; i < plane; i++) {
    chw[i] = data[i * 4] / 255;
    chw[plane + i] = data[i * 4 + 1] / 255;
    chw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', chw, [1, 3, S, S]), previewCanvas: c };
}

function softmaxLog(logits) {
  let m = -Infinity;
  for (const v of logits) if (v > m) m = v;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - m);
  const logsum = Math.log(sum) + m;
  const logp = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) logp[i] = logits[i] - logsum;
  return logp;
}

function marginalizeLog(childLogP, childToParent, nParents) {
  const maxes = new Float64Array(nParents).fill(-Infinity);
  for (let i = 0; i < childLogP.length; i++) {
    const p = childToParent[i];
    if (childLogP[i] > maxes[p]) maxes[p] = childLogP[i];
  }
  const sums = new Float64Array(nParents);
  for (let i = 0; i < childLogP.length; i++) sums[childToParent[i]] += Math.exp(childLogP[i] - maxes[childToParent[i]]);
  const out = new Float64Array(nParents);
  for (let p = 0; p < nParents; p++) out[p] = Math.log(sums[p]) + maxes[p];
  return out;
}

function argmax(arr) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
  return bi;
}

function calibratedTopProb(logp, idx, temperature) {
  if (!temperature || temperature === 1) return Math.exp(logp[idx]);
  let m = -Infinity;
  for (const v of logp) if (v / temperature > m) m = v / temperature;
  let sum = 0;
  for (const v of logp) sum += Math.exp(v / temperature - m);
  return Math.exp(logp[idx] / temperature - m) / sum;
}

/**
 * Run the model and return the consistent 3-level prediction, finest→coarsest:
 *   [{ level, key, name, prob, aboveThreshold }, …]
 */
export async function predict(tensor) {
  const out = await session.run({ [cfg.inputName]: tensor });
  const speciesLogits = out[cfg.outputs.species].data;

  const spLogP = softmaxLog(speciesLogits);
  const p = taxonomy.parents;
  const gnLogP = marginalizeLog(spLogP, p.species_to_genus, taxonomy.vocabs.genus.length);
  const fmLogP = marginalizeLog(gnLogP, p.genus_to_family, taxonomy.vocabs.family.length);
  const perLevel = { species: spLogP, genus: gnLogP, family: fmLogP };

  return LEVELS.map((level) => {
    const logp = perLevel[level];
    const idx = argmax(logp);
    const T = calibration?.temperatures?.[level]?.temperature ?? 1;
    const key = taxonomy.vocabs[level][idx];
    const name = names?.[level]?.[idx] || '';
    const thr = thresholds?.[level] ?? null;
    return {
      level, key, name,
      prob: calibratedTopProb(logp, idx, T),
      aboveThreshold: thr == null ? true : calibratedTopProb(logp, idx, T) >= thr,
    };
  });
}

export function gbifUrl(key) { return (cfg?.gbifBase || 'https://www.gbif.org/species/') + key; }
export function modelName() { return cfg?.name || ''; }
export function isReady() { return session != null; }
