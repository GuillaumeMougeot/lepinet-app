# Developer guide — bring your own model

lepinet-app is **model-agnostic**. Everything the app needs to run a model lives in one folder,
`model/`, described by `model/config.json`. Swapping in a model you trained is: produce a bundle,
drop it in `model/`, bump the service-worker cache, deploy. No code changes, no build step.

This guide covers the bundle contract, how the app uses it, and how to generate a bundle from a
model trained in the [`lepinet`](https://github.com/GuillaumeMougeot/lepinet) repo (or any model
that meets the contract).

---

## 1. The bundle contract

A bundle is the `model/` folder. `config.json` is the manifest; everything else is referenced
from it.

```
model/
  config.json        # manifest (below)
  model.onnx         # the network (required)
  taxonomy.json      # class vocab + hierarchy (required)
  names.json         # display names (optional; falls back to taxonomy keys)
  calibration.json   # per-level temperature (optional)
  thresholds.json    # per-level confidence thresholds (optional)
```

### `config.json`

```jsonc
{
  "name": "EfficientNetV2-B2 · Lepidoptera v1",   // shown in the UI
  "model": "model.onnx",                           // primary model, relative to model/
  "fallback": "https://…/model.onnx",              // optional: tried if the primary won't run
  "taxonomy": "taxonomy.json",
  "names": "names.json",                           // omit to show taxonomy keys
  "calibration": "calibration.json",               // omit for raw softmax confidence
  "thresholds": "thresholds.json",                 // omit to never grey a level
  "imageSize": 256,                                 // model input side length
  "inputName": "image",                            // ONNX input tensor name
  "outputs": { "species": "logits_species",         // ONNX output tensor names, finest→coarsest
               "genus":   "logits_genus",
               "family":  "logits_family" },
  "gbifBase": "https://www.gbif.org/species/"       // <base><key> forms each taxon's link
}
```

`fallback` is how this app ships **QDQ int8 as primary (small, ~15 MB) with fp32 as a safety net**
(the fp32 model on the lepinet GitHub release). If the primary model fails to create a session on
a device, the loader tries the fallback URL.

### The model (`model.onnx`)

Hard requirements — the app assumes all of these:

1. **Input:** one tensor named `config.inputName`, shape `[1, 3, S, S]`, dtype `float32`, **RGB,
   values in `[0, 1]`**, where `S = config.imageSize`.
2. **Normalization is baked into the graph.** The app feeds raw `[0,1]` pixels; the model applies
   its own mean/std internally. (Do not expect the app to normalize.)
3. **Outputs:** raw logits (pre-softmax) for each level, named per `config.outputs`. The app only
   strictly needs the **species** output — genus and family are *marginalized* from it (§3), so
   the genus/family outputs may be present but are not read.
4. **Preprocessing the app performs:** shorter-side resize to `S` + center crop. If your model was
   trained with a different validation transform, either retrain to match or expect a small
   accuracy drop (see `lepinet/dev/041` for why this specific transform).
5. **ONNX Runtime Web compatibility:** the op set must be one ORT Web implements on the target
   devices. This is the trickiest constraint:
   - **fp32 always works** — the safe default (v1 ships fp32 for exactly this reason).
   - **Dynamic int8 never works** — it emits `ConvInteger`/`MatMulInteger`, unsupported on every
     ORT Web backend (fails with *"Could not find an implementation for ConvInteger"*).
   - **Static QDQ int8** (§4) is smaller (~3.5×) and *should* work (only `QuantizeLinear`/
     `DequantizeLinear` + float `Conv`), but has been observed to fail at session creation on
     some devices (a raw numeric WASM error) — **validate it in a real browser on your target
     devices before shipping it as primary.** Keep fp32 as a same-origin `fallback` if you do.
   - **fp16** (~2×) is untested here and risky for this model — the cosine head (`normalize` +
     `acos`) is fp16-sensitive; if you try it, keep the head fp32 via the converter's op
     block-list and validate.

### `taxonomy.json`

```jsonc
{
  "vocabs": {
    "species": ["10000103", …],   // taxon keys in MODEL OUTPUT ORDER (index = logit index)
    "genus":   ["1738259", …],
    "family":  ["3266", …]
  },
  "parents": {
    "species_to_genus": [216, …], // for species index i, the genus index of its parent
    "genus_to_family":  [12, …]   // for genus index i, the family index of its parent
  }
}
```

The vocab arrays **must** be in the model's output order — `vocabs.species[i]` is the taxon for
logit `i`. The parent arrays drive marginalization and must index into the coarser vocab.

### `names.json` (optional)

```jsonc
{ "names": {
    "species": ["Eucosma raracana", …],  // aligned to taxonomy.vocabs.species order
    "genus":   ["Eucosma", …],
    "family":  ["Tortricidae", …] } }
```

Aligned index-for-index with the taxonomy vocabs. Empty string → the app shows the key instead.

### `calibration.json` / `thresholds.json` (optional)

```jsonc
// calibration.json
{ "temperatures": { "species": { "temperature": 0.80 }, "genus": {…}, "family": {…} } }

// thresholds.json
{ "levels": { "species": { "threshold": 0.51 }, "genus": {…}, "family": {…} } }
```

The app divides each level's logits by the temperature to get a calibrated probability, then greys
a level whose top probability is below its threshold. Omit both for raw softmax and no greying.

---

## 2. Swapping in a bundle

```bash
# 1. Replace the folder with your bundle
rm -rf model && cp -r /path/to/your-bundle model

# 2. Bump the cache version so clients pick up the new files
#    (edit sw.js: const CACHE = 'lepinet-vN' -> 'lepinet-vN+1')

# 3. Commit + push; GitHub Actions redeploys to Pages
git add -A && git commit -m "model: <name>" && git push
```

Test locally first with any static server (single-thread WASM needs no special headers):

```bash
python -m http.server 8000   # then open http://localhost:8000
```

---

## 3. What the app does with the bundle (so you can reason about it)

`src/infer.js`, per image:

1. **Preprocess** → `[1,3,S,S]` `[0,1]` RGB (shorter-side resize + center crop).
2. **Run** the model; take the **species** logits.
3. **Marginalize**: `log P(genus) = logsumexp over child species`, then family from genus, using
   `taxonomy.parents`. This is why genus/family are always *consistent* with species — and it is
   measurably more accurate than reading separate genus/family heads (`lepinet/dev/042`). Your
   model does not need to output good genus/family logits; species is enough.
4. **Calibrate + threshold** per level, look up **name** and **key**, render.

---

## 4. Generating a bundle from a lepinet model

If you trained with `lepinet/dev/030`, the bundle is four scripts:

```bash
cd lepinet
# 0. (train) -> data/global/models/<run>/<name>.pt

# 1. Export to ONNX + taxonomy + manifest (fp32, browser-parity checked)
python dev/040_onnx_export.py -c "data/global/models/*<name>/*.pt" -o bundle/

# 2. Static QDQ int8 (browser-safe; ~3.5x smaller than fp32, same accuracy)
python - <<'PY'
import importlib; q = importlib.import_module("dev.043_quantize".replace("dev.","")) # see dev/043
# q.quantize_static_qdq(src, dst, calib_image_paths)   # ~200 preprocessed images
PY

# 3. Calibration + precision-targeted thresholds
python dev/044_calibrate.py --onnx bundle/model.onnx --out-dir bundle/ --target-precision 0.95

# 4. Names map from the dataset parquet (scientificName / genus / family columns),
#    aligned to taxonomy vocab order  ->  names.json
```

Then assemble `config.json` (point `model` at the QDQ file, `fallback` at the fp32 on a release),
and drop the folder into the app's `model/`.

The reasoning behind every step — preprocessing choice, marginalization, int8 vs fp16, the
ConvInteger gotcha, calibration — is written up in
[`lepinet/journal/2026-07-lepi-app-compression.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-lepi-app-compression.md).

---

## 5. Constraints & gotchas

- **No JS build step.** The app is plain ES modules; keep it that way (the deploy box has no
  node/bun). Add libraries as vendored ES modules under a folder, like `ort/`.
- **ORT Web op support** is the main compatibility limit. When in doubt, fp32 always works; static
  QDQ int8 works; dynamic int8 does not.
- **Size vs offline.** Everything in `model/` is precached by the service worker for offline use,
  so bundle size is download-and-storage cost. The ORT `.wasm` (~12–25 MB per variant) is cached
  on first use, not precached.
- **Cross-origin isolation** is off on GitHub Pages, so ORT runs single-threaded. That's set in
  `infer.js` (`ort.env.wasm.numThreads = 1`); leave it unless you serve with COOP/COEP headers.
