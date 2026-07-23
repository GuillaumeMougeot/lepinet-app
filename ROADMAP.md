# Lepinet — development roadmap

The single place for **what's next**, across the model ([`lepinet`](https://github.com/GuillaumeMougeot/lepinet))
and the app (this repo). Reasoning and measurements behind everything already built live in
`lepinet/journal/2026-07-lepi-app-*.md`; this document is the forward-looking companion.

Status legend: **[done]** shipped · **[next]** clear next step · **[later]** wanted, not urgent ·
**[blocked]** needs something we don't have yet.

---

## Where we are (v1, 2026-07)

- Model: EfficientNetV2-B2 + 256-d bottleneck cosine head, 12,041 species. Test species
  macro-F1 **0.887**. Static-QDQ int8 ONNX, **15.5 MB**.
- App: offline PWA, on-device inference, marginalized & consistent 3-level output, calibrated
  confidence with greying, scientific names, camera/gallery, installable, config-driven bundle.
- Model published as a GitHub release; app pins its own copy.

The chain that got here: `journal/2026-07-lepi-app-compression.md` (A→D), with the plan and the
open product decisions in `journal/2026-07-lepi-app-claude.md`.

---

## Model quality

### [next] C3 — knowledge distillation
The one deliberately-skipped compression step. The teacher (effnetv2_s, **0.9148**) is 2.8 pp
above the shipped student (effnetv2b2, 0.887). Distilling soft targets from the teacher — ideally
over *unlabelled* extra images, since soft targets over 12 k classes carry the hierarchy — should
recover a good chunk of that gap at no inference cost. Needs a new `dev/045` that runs the teacher
per batch and adds a KL term to the existing `MultiLevelWeightedCrossEntropyLoss`. Estimated
lift: +0.5–1.5 pp. **Highest-value model work.**

### [later] Push the recipe further
5-epoch runs were chosen for iteration speed. A longer schedule, or a bigger student (effnetv2_s
at hidden=256 is already 0.9002 at ~24 MB — an "accuracy-first" bundle option now that size is
relaxed), trades size for accuracy. The fastvit_sa12 (0.892, reparameterizes to conv) remains a
viable browser-friendly alternative if we ever want max accuracy per MB.

### [blocked] Geographic prior
Likely the largest single accuracy lever (iNaturalist's biggest win): multiply the visual
posterior by a coarse per-species occurrence prior so allopatric look-alikes stop being confused.
**Blocked:** needs species×location co-occurrence data we don't currently have. When available:
build a coarse grid (H3 res-2 or ~5°), quantize to int8 log-priors (~hundreds of KB), ship in the
bundle, opt-in (location permission), tempered (`P_visual · P_geo^α`, α<1) so it never buries a
genuine vagrant/range-shift record. Also the softer, single-artifact version of a regional build.

### [later] Open-set / "is this even a lepidopteran?"
The model confidently classifies beetles, leaves, coffee cups — softmax normalizes over the
classes it has. Proper handling needs an OOD score (energy / max-logit) thresholded against an
explicit **negative image set** (other insect orders, plants, indoor scenes, faces) that must be
built, or a cheap binary gate in front. v1 ships honest copy ("may not be a moth/butterfly or may
be outside the training set") instead. Deferred as its own research task.

### [later] Regional / smaller-class builds
Filtering to a region (e.g. Denmark, N. Europe) cuts the class set from 12 k to a few
hundred/thousand: much higher accuracy (most confusions are between species that never co-occur)
and genuinely small (**the sub-1 MB build only exists here** — 12 k species have an information
floor of ~385 KB at 64-d/4-bit prototypes). Cost: a build matrix + a "which region?" choice. The
config-driven bundle already supports shipping multiple models; this is a data-pipeline task.

---

## Model size / efficiency

### [later] fp16 variant
27 MB, no quantization-op pitfalls. Risk: the cosine head (F.normalize + acos) is fp16-sensitive
(it's why training runs the head in fp32) — would need the head kept fp32 via the float16
converter's op block-list, which then dominates size. Static-QDQ int8 (current) is the better
point; fp16 only interesting if a device rejects QDQ.

### [later] 4-bit head prototypes
The cosine prototype rows share a dynamic range (per-row 4-bit is plausible). Would trim another
~1–2 MB off the head. Measure the F1 cost; abandon if >0.5 pp.

### [later] Slim the ORT runtime
The vendored ORT `.wasm` is 12–25 MB per variant — larger than the model. Options: build a
custom ORT-web with only the ops this graph uses; or drop WebGPU (ship the plain wasm only) if
WebGPU never wins on target devices. Meaningful download reduction.

---

## App features

### [next] Pull the model from the GitHub release
Decouple the model from the app repo (owner's request). The app fetches the bundle from the
`lepinet` release (or a pinned version), caches it in the service worker, and shows the version in
the UI. Turns a model update into a release + a version bump — no app redeploy — and shrinks the
app repo. `config.json` already has the seam (`fallback` is a release URL today).

### [later] Sample images per taxon
The proposal's bottom-tier gallery. Deferred in v1 to stay light & copyright-free. Best form: a
"show examples" button that **fetches from GBIF on demand** (offline ID, online illustration) with
attribution — avoids bundling ~289 MB of thumbnails and the mixed-licence problem.

### [later] Top-k / "other possibilities"
macro-F1 is the research metric; the product metric is closer to **top-5 recall** — a user shown
5 candidates can disambiguate. Add an expandable "other likely species" list under a low-confidence
top-1. Also report top-1/top-5 alongside macro-F1 when evaluating future models.

### [later] Identification history
Store captures + predictions locally (IndexedDB), a simple list view. From the proposal's future
avenues. Purely on-device, preserves the privacy property.

### [later] Confidence UX
Current greying targets 0.95 precision (species greys ~6%). Worth revisiting per real usage:
maybe a 3-state (confident / uncertain / not-a-lep) display, or a user-tunable strictness.

---

## App infrastructure & polish

### [next] Real-device test matrix
The one gap Python validation can't cover. Priorities: **iOS Safari** (WASM path, storage
eviction, `Add to Home Screen`), Android Chrome (WebGPU vs WASM), desktop. A short manual
checklist per release until automated browser tests exist.

### [later] Automated browser tests
Once a JS toolchain is available on a build box, a Playwright smoke test (load model, run the
`lepinet/dev/041` fixture images, assert top-1) would catch runtime regressions the Python
validation can't. The fixture already exists.

### [later] Robustness & a11y
EXIF orientation is intentionally *not* handled (moths sit in any orientation) — revisit only if
the pipeline is reused for taxa with a canonical pose. Add: accessible labels/focus states,
larger tap targets, and possibly i18n of the UI chrome.

### [later] Privacy-preserving feedback
A "was this right?" affordance that (opt-in) logs only the prediction + user correction, never the
image, to improve future training sets. Must be explicit and off by default.

---

## Dev / ops

- **[later] CI bundle build.** A `lepinet` workflow that, on a tagged model release, runs
  `dev/040`→`044`+`047` and attaches the full bundle — so releasing a model is one action and the
  app can pull it (ties into "pull from release" above).
- **[later] Versioned bundle format.** `MANIFEST.json` already records provenance; formalize a
  `bundleVersion` the app displays and can pin.
- **[done] Modular bundle contract** — `DEVELOPER.md` + `config.json`; swapping a model is a
  folder replace.

---

## Suggested order

1. **C3 distillation** — biggest model-quality win, GPU is reserved.
2. **Pull model from release** — decouples model/app, enables painless updates.
3. **Real-device test matrix** — close the one validation gap (esp. iOS).
4. Then pick from features (top-k, history, sample images) and the size levers as priorities shift.

Geographic prior and open-set are the two highest-*ceiling* items but are gated on data that
doesn't exist yet; start them when the data does.
