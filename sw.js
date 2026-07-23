// Service worker: makes Lepinet installable and fully offline.
//
// Strategy: precache the app shell + model bundle + ORT runtime on install, then serve
// cache-first (this is a static app whose only "data" is the bundled model, so freshness is a
// deploy-time concern, not a runtime one). Bumping CACHE on each deploy evicts the old set.
//
// The model (~14 MB) and the ORT wasm (~23 MB) dominate the precache; that is the deliberate
// cost of offline inference, and it is why they live behind an explicit install step the user
// triggers by adding the app to their home screen.

const CACHE = 'lepinet-v8';

// Precache ONLY the small shell (HTML/JS/CSS/config, all a few KB). The heavy assets — the model
// (~15 MB) and the ORT .wasm (~12–25 MB) — are deliberately NOT in the install set: a single
// flaky download would reject the whole `addAll` and leave the service worker stuck on a stale
// version (that is what broke the app after a cache bump). Instead the fetch handler caches those
// big files on first successful load, so the app becomes fully offline after one online run
// without a fragile 70 MB install step.
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
  './model/config.json',
  './model/taxonomy.json',
  './model/names.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    // Cache shell items individually so one 404/hiccup can't fail the whole install and wedge
    // the SW; the big model + wasm are picked up by the fetch handler on first load.
    caches.open(CACHE)
      .then((c) => Promise.all(SHELL.map((u) => c.add(u).catch((err) => console.warn('precache skip', u, err)))))
      .then(() => self.skipWaiting())
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
