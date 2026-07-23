// Service worker: makes Lepinet installable and fully offline.
//
// Strategy: precache the app shell + model bundle + ORT runtime on install, then serve
// cache-first (this is a static app whose only "data" is the bundled model, so freshness is a
// deploy-time concern, not a runtime one). Bumping CACHE on each deploy evicts the old set.
//
// The model (~14 MB) and the ORT wasm (~23 MB) dominate the precache; that is the deliberate
// cost of offline inference, and it is why they live behind an explicit install step the user
// triggers by adding the app to their home screen.

const CACHE = 'lepinet-v6';

// Precache the shell + model + the loader mjs (tiny). The big .wasm variants (~59 MB across
// jsep/plain/asyncify) are NOT precached — ORT loads exactly one at runtime depending on the
// device, and the fetch handler below caches whatever it fetches, so offline works after the
// first successful run without forcing a 70 MB install.
const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './src/app.js',
  './src/infer.js',
  './src/install.js',
  './src/style.css',
  './assets/icon.svg',
  './assets/icon-192.png',
  './assets/icon-512.png',
  './ort/ort.webgpu.mjs',
  './ort/ort-wasm-simd-threaded.mjs',
  './ort/ort-wasm-simd-threaded.jsep.mjs',
  './ort/ort-wasm-simd-threaded.asyncify.mjs',
  './model/config.json',
  './model/model.onnx',
  './model/taxonomy.json',
  './model/names.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) =>
      // Sidecars (calibration/thresholds) are optional — don't fail the whole install if absent.
      Promise.all([
        c.addAll(SHELL),
        c.add('./model/calibration.json').catch(() => {}),
        c.add('./model/thresholds.json').catch(() => {}),
      ])
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // Never intercept cross-origin (e.g. GBIF pages open in a new tab, image fetches later).
  if (url.origin !== location.origin || e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((hit) => {
      if (hit) return hit;
      // Cache-on-fetch: the first run pulls one big .wasm variant; store it so the app is
      // fully offline afterwards, without precaching all 59 MB of variants up front.
      return fetch(e.request).then((res) => {
        if (res.ok && res.type === 'basic') {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      });
    })
  );
});
