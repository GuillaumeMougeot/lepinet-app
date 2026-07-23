# Lepinet — moth & butterfly ID (PWA)

An offline-first web app that identifies a Lepidoptera photo to **species, genus, and family**,
running the model **entirely on your device** — no photo ever leaves your phone.

Companion to the [`lepinet`](https://github.com/GuillaumeMougeot/lepinet) model repo. The model
and its compression story (ONNX export, int8, bottleneck + backbone search, calibration) live
there under `journal/2026-07-lepi-app-*.md`.

## How it works

- **Model:** an EfficientNetV2-B2 backbone with a 256-d bottleneck classifier head, trained on
  ~5.7 M GBIF images across 12,041 species. Exported to ONNX, int8-quantized (**~14 MB**).
- **Runtime:** [ONNX Runtime Web](https://onnxruntime.ai/), WebGPU where available and
  WASM-SIMD everywhere else (iOS Safari).
- **Preprocessing:** shorter-side resize to 256 + center crop — measured to match the training
  pipeline within 0.1 pp (`lepinet/dev/041`).
- **Consistency:** genus and family are **marginalized** from the species posterior, so the
  three levels can never contradict each other (`lepinet/dev/042`).
- **Confidence:** each level shows a calibrated probability; below a per-level threshold the
  name is greyed. If nothing clears threshold, the app says the subject may not be a
  lepidopteran or may be outside the training set.

## No build step

The deploy environment has no JS toolchain, so this ships as **plain static files** —
ES modules, a vendored copy of ONNX Runtime Web under `ort/`, and the model bundle under
`model/`. GitHub Pages serves the repo root directly; `.github/workflows/deploy.yml` just
publishes it. To run locally, serve the folder with any static server, e.g.
`python -m http.server` (WebGPU/WASM threads may need cross-origin isolation headers; WASM
single-thread works without).

## Layout

```
index.html              app shell (home + result screens)
src/infer.js            model load, preprocess, inference, marginalization, calibration
src/app.js              UI controller
src/style.css           styles (light/dark)
sw.js                   service worker — precaches shell + model + runtime for offline
manifest.webmanifest    PWA manifest
ort/                    vendored ONNX Runtime Web (mjs + wasm)
model/                  model.onnx (int8) + taxonomy.json (+ calibration/thresholds)
```

## Model bundle

`model/` is a pinned copy of an artifact bundle emitted from the `lepinet` repo
(`dev/040`–`044`): `model.onnx`, `taxonomy.json`, and optional `calibration.json` /
`thresholds.json`. Swapping in a newer/better model is a matter of replacing that folder and
bumping the cache version in `sw.js`.

**Not for conservation, pest, or toxicity decisions** — predictions can be wrong.
