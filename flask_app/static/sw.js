//
// Fraud Shield AI — Service Worker
// Strategy:
//   - Static assets (CSS/JS/images/fonts): cache-first
//   - Navigation (HTML pages):              network-first, fall back to cache
//   - API endpoints (/api/..., /fraud/.../predict, ..._stream): network-only
//     (never cache, fraud data must be fresh)
// Bump CACHE_VERSION whenever static assets change to force update.
//

const CACHE_VERSION = 'fraudshield-v3';
const STATIC_CACHE  = `${CACHE_VERSION}-static`;
const PAGES_CACHE   = `${CACHE_VERSION}-pages`;

// Pre-cache the shell — keep this list small
const PRECACHE_URLS = [
  '/',
  '/offline',
  '/static/css/style.css',
  '/static/css/fraud.css',
  '/static/js/icons.js',
  '/static/favicon.svg',
  '/manifest.json',
  '/static/images/favicon.ico',
  '/static/images/icon-192.png',
];

// Install: prefetch the app shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS).catch(() => null))
      .then(() => self.skipWaiting())
  );
});

// Activate: nuke stale versioned caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => !k.startsWith(CACHE_VERSION)).map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// Helpers
const isStaticAsset = (url) =>
  url.pathname.startsWith('/static/') ||
  /\.(?:css|js|png|jpg|jpeg|webp|svg|ico|woff2?|ttf|otf)$/i.test(url.pathname);

const isApiOrStream = (url) =>
  url.pathname.startsWith('/api/') ||
  url.pathname.endsWith('/predict') ||
  url.pathname.endsWith('/analyze') ||
  url.pathname.endsWith('/stream') ||
  url.pathname.endsWith('/data') ||
  url.pathname.includes('/pdf');

// Fetch handler
self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return; // never cache POST/PUT/DELETE

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // ignore cross-origin

  // 1) APIs / live streams: bypass cache completely
  if (isApiOrStream(url)) return;

  // 2) Static assets: cache-first
  if (isStaticAsset(url)) {
    event.respondWith(
      caches.match(req).then((cached) =>
        cached ||
        fetch(req).then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(STATIC_CACHE).then((c) => c.put(req, copy));
          }
          return res;
        }).catch(() => cached)
      )
    );
    return;
  }

  // 3) Navigations / HTML: network-first, fall back to cache, fall back to offline page
  if (req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html')) {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(PAGES_CACHE).then((c) => c.put(req, copy));
          }
          return res;
        })
        .catch(() =>
          caches.match(req).then((cached) => cached || caches.match('/offline'))
        )
    );
  }
});

// Listen for skipWaiting message from page (for instant updates)
self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});

// ═══════════════════════════════════════════════════════════════
// Web Push: incoming notification + click handler
// ═══════════════════════════════════════════════════════════════
self.addEventListener('push', (event) => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch (_) {
    data = { title: 'Fraud Shield', body: event.data ? event.data.text() : '' };
  }

  const title = data.title || 'Fraud Shield Alert';
  const options = {
    body:    data.body  || 'A new fraud alert has been received.',
    icon:    data.icon  || '/static/images/icon-192.png',
    badge:   data.badge || '/static/images/icon-192.png',
    tag:     data.tag   || 'fraudshield',
    vibrate: data.vibrate || [200, 100, 200],
    requireInteraction: !!data.requireInteraction,
    renotify: data.renotify !== false,
    data:    { url: data.url || '/fraud/alerts', ts: data.ts || Date.now() },
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || '/fraud/alerts';

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      // If the app is already open, focus it and navigate
      for (const client of clientList) {
        if ('focus' in client) {
          client.focus();
          if ('navigate' in client) client.navigate(targetUrl);
          return;
        }
      }
      // Otherwise open a new window
      if (self.clients.openWindow) return self.clients.openWindow(targetUrl);
    })
  );
});
