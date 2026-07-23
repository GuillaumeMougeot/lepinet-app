// Install affordance for the home screen.
//
// Three cases:
//   * Already installed (running standalone) → show nothing.
//   * Installable via the browser prompt (Chrome/Edge on Android & desktop) → capture the
//     `beforeinstallprompt` event and offer a real "Install app" button.
//   * Everything else, notably iOS Safari (no prompt API) → show written instructions, since
//     the user's request was: if the pop-up is skipped, the home page should explain how to
//     get the app.

let deferredPrompt = null;

export function initInstall() {
  const section = document.getElementById('install');
  const btn = document.getElementById('install-btn');
  const help = document.getElementById('install-help');
  if (!section) return;

  // Installed already? Nothing to do.
  const standalone =
    window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;
  if (standalone) { section.hidden = true; return; }

  section.hidden = false;

  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    btn.hidden = false;
    help.hidden = true; // prefer the one-tap button when the browser supports it
  });

  btn.addEventListener('click', async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    await deferredPrompt.userChoice;
    deferredPrompt = null;
    btn.hidden = true;
  });

  window.addEventListener('appinstalled', () => { section.hidden = true; });

  // Written fallback instructions, tailored to the platform.
  help.innerHTML = instructionsFor();
}

function instructionsFor() {
  const ua = navigator.userAgent;
  const iOS = /iPhone|iPad|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (iOS) {
    return `<strong>Install on iPhone/iPad:</strong> tap the Share button
      <span aria-hidden="true">⬆️</span> in Safari, then <em>Add to Home Screen</em>.`;
  }
  if (/Android/.test(ua)) {
    return `<strong>Install on Android:</strong> open the browser menu
      <span aria-hidden="true">⋮</span> and tap <em>Install app</em> (or <em>Add to Home screen</em>).`;
  }
  return `<strong>Install:</strong> in Chrome or Edge, use the install icon
    <span aria-hidden="true">⊕</span> in the address bar, or the menu → <em>Install Lepinet</em>.
    Installing lets it run offline.`;
}
