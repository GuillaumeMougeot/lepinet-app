// Inference core for lepinet-app.
//
// Loads the ONNX model + taxonomy, preprocesses an image the way the training pipeline's
// validation transform does, runs the model, and derives a consistent species/genus/family
// prediction. Everything here runs on-device; no image ever leaves the phone.
//
// Two decisions come straight from the lepinet compression experiments
// (lepinet/journal/2026-07-lepi-app-compression.md):
//   * Preprocessing = shorter-side resize to 256 + center crop. dev/041 measured that this
//     matches fastai's validation pipeline to within 0.1 pp; the resampling kernel is a
//     non-issue, only aspect ratio matters, so a plain canvas drawImage is fine.
//   * Genus and family are *marginalized* from the species posterior, not read from the model's
//     own genus/family heads. dev/042 proved this is both more accurate (+0.7 / +3.1 pp) and
//     consistent by construction — the three levels can never contradict each other, which is
//     exactly what the stacked UI needs.
//
// The model outputs raw logits for all three levels; we use the species logits and roll them up.

import * as ort from '../ort/ort.webgpu.mjs';

ort.env.wasm.wasmPaths = new URL('../ort/', import.meta.url).href;

const IMG_SIZE = 256;
const MEAN = [0.485, 0.456, 0.406]; // baked into the graph too, but kept for reference
const STD = [0.229, 0.224, 0.225];

let session = null;
let taxonomy = null;
let calibration = null; // { temperatures: {species,genus,family} } or null
let thresholds = null;  // { species, genus, family } or null

/** Load model + sidecars. Idempotent. `base` is the model dir URL. */
export async function loadModel(base = './model/', onProgress = () => {}) {
  if (session) return;
  onProgress('Loading taxonomy…');
  taxonomy = await fetch(base + 'taxonomy.json').then((r) => r.json());
  // Optional sidecars — the app degrades gracefully if a bundle ships without them.
  calibration = await fetch(base + 'calibration.json').then((r) => r.ok ? r.json() : null).catch(() => null);
  const thr = await fetch(base + 'thresholds.json').then((r) => r.ok ? r.json() : null).catch(() => null);
  if (thr && thr.levels) {
    thresholds = {
      species: thr.levels.species?.threshold ?? null,
      genus: thr.levels.genus?.threshold ?? null,
      family: thr.levels.family?.threshold ?? null,
    };
  }

  onProgress('Loading model…');
  // WebGPU where available (fast), WASM-SIMD everywhere else. iOS Safari falls back to WASM.
  const providers = [];
  if (navigator.gpu) providers.push('webgpu');
  providers.push('wasm');
  session = await ort.InferenceSession.create(base + 'model.onnx', {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });

  onProgress('Warming up…');
  await warmup(); // pay the first-inference graph-init cost now, not on the user's first photo
  onProgress('Ready');
}

async function warmup() {
  const dummy = new ort.Tensor('float32', new Float32Array(3 * IMG_SIZE * IMG_SIZE), [1, 3, IMG_SIZE, IMG_SIZE]);
  await session.run({ image: dummy });
}

/**
 * Preprocess an ImageBitmap/HTMLImageElement to a [1,3,256,256] float tensor in [0,1], RGB,
 * via shorter-side resize + center crop (see module header). Returns { tensor, previewCanvas }.
 */
export function preprocess(img) {
  const w = img.width, h = img.height;
  const scale = IMG_SIZE / Math.min(w, h);
  const nw = Math.max(IMG_SIZE, Math.round(w * scale));
  const nh = Math.max(IMG_SIZE, Math.round(h * scale));

  const c = document.createElement('canvas');
  c.width = IMG_SIZE; c.height = IMG_SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  // Draw the resized image and center-crop in one drawImage: source is the full image, dest is
  // offset so the IMG_SIZE×IMG_SIZE window sits at the center of the nw×nh resized image.
  const dx = (IMG_SIZE - nw) / 2, dy = (IMG_SIZE - nh) / 2;
  ctx.drawImage(img, dx, dy, nw, nh);

  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const chw = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    chw[i] = data[i * 4] / 255;               // R
    chw[plane + i] = data[i * 4 + 1] / 255;    // G
    chw[2 * plane + i] = data[i * 4 + 2] / 255; // B
  }
  return { tensor: new ort.Tensor('float32', chw, [1, 3, IMG_SIZE, IMG_SIZE]), previewCanvas: c };
}

function softmaxLog(logits) {
  let m = -Infinity;
  for (const v of logits) if (v > m) m = v;
  let sum = 0;
  const exp = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) { exp[i] = Math.exp(logits[i] - m); sum += exp[i]; }
  const logsum = Math.log(sum) + m;
  const logp = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) logp[i] = logits[i] - logsum;
  return logp; // log-probabilities
}

/** log P(parent) = logsumexp over children, given a child→parent index array. */
function marginalizeLog(childLogP, childToParent, nParents) {
  const maxes = new Float64Array(nParents).fill(-Infinity);
  for (let i = 0; i < childLogP.length; i++) {
    const p = childToParent[i];
    if (childLogP[i] > maxes[p]) maxes[p] = childLogP[i];
  }
  const sums = new Float64Array(nParents);
  for (let i = 0; i < childLogP.length; i++) {
    const p = childToParent[i];
    sums[p] += Math.exp(childLogP[i] - maxes[p]);
  }
  const out = new Float64Array(nParents);
  for (let p = 0; p < nParents; p++) out[p] = Math.log(sums[p]) + maxes[p];
  return out;
}

function argmax(arr) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
  return bi;
}

/** Apply a temperature to log-probs and return the max as a calibrated probability. */
function calibratedTopProb(logp, idx, temperature) {
  if (!temperature || temperature === 1) return Math.exp(logp[idx]);
  // Re-softmax the underlying logits at temperature T. logp are log-probs, i.e. logits shifted
  // by a constant; dividing by T and renormalizing is the standard temperature rescaling.
  let m = -Infinity;
  for (const v of logp) if (v / temperature > m) m = v / temperature;
  let sum = 0;
  for (const v of logp) sum += Math.exp(v / temperature - m);
  return Math.exp(logp[idx] / temperature - m) / sum;
}

/**
 * Run the model on a preprocessed tensor and return the consistent 3-level prediction:
 *   [{ level, key, prob, aboveThreshold }, …] finest→coarsest, plus the raw species logp.
 */
export async function predict(tensor) {
  const out = await session.run({ image: tensor });
  const speciesLogits = out.logits_species.data; // Float32Array, length = #species

  const spLogP = softmaxLog(speciesLogits);
  const p = taxonomy.parents;
  const nGenus = taxonomy.vocabs.genus.length;
  const nFamily = taxonomy.vocabs.family.length;
  const gnLogP = marginalizeLog(spLogP, p.species_to_genus, nGenus);
  const fmLogP = marginalizeLog(gnLogP, p.genus_to_family, nFamily);

  const levels = [
    { name: 'species', logp: spLogP },
    { name: 'genus', logp: gnLogP },
    { name: 'family', logp: fmLogP },
  ];

  return levels.map(({ name, logp }) => {
    const idx = argmax(logp);
    const T = calibration?.temperatures?.[name]?.temperature ?? 1;
    const prob = calibratedTopProb(logp, idx, T);
    const thr = thresholds?.[name] ?? null;
    return {
      level: name,
      key: taxonomy.vocabs[name][idx],
      prob,
      aboveThreshold: thr == null ? true : prob >= thr,
    };
  });
}

export function gbifUrl(key) {
  return `https://www.gbif.org/species/${key}`;
}

export function isReady() { return session != null; }
