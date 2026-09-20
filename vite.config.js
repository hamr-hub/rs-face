// rs-face platform — Vite dev server (at the repo root).
//
// The production bundle is shipped inside the rsface-server Docker image
// (see platform/Dockerfile); this config exists only for the dev workflow:
//
//   1. Backend:  `docker compose -f platform/docker-compose.yml up -d --build`
//                (rsface-server on http://localhost:20080)
//   2. Frontend: `pnpm dev`  (Vite on http://localhost:5173, HMR on)
//
// Vite serves `platform/web/` as a static root (see `root:` below), then
// proxies /api/* and /events to the Docker backend. Edits to app.js /
// *.css / index.html reflect immediately via HMR — no rebuild of the
// backend image needed.
//
// To point at a different backend (e.g. a remote dev box), set
// RSFACE_BACKEND=http://host:port  before running pnpm dev.

import { defineConfig } from 'vite';

const BACKEND = process.env.RSFACE_BACKEND || 'http://localhost:20080';

export default defineConfig({
  // platform/web/ is the static root (relative to the repo root where this
  // config file lives). Production ships these same files from /app/web/
  // inside the rsface-server container.
  root: 'platform/web',

  server: {
    host: '0.0.0.0',       // listen on all interfaces so LAN peers can hit it
    port: 5173,
    strictPort: true,       // fail loudly if 5173 is taken rather than auto-bumping
    open: false,            // don't auto-open browser; tail docker-logs in another tab

    proxy: {
      // Every API call in app.js uses a relative '/api/...' path; forward them
      // all to the docker-deployed rsface-server.
      '/api': {
        target: BACKEND,
        changeOrigin: false,
      },
      // SSE endpoint for live task progress.
      '/events': {
        target: BACKEND,
        changeOrigin: false,
        // SSE needs the response stream passed through verbatim. Vite's
        // http-proxy does this by default, but we make it explicit.
        ws: false,
        selfHandleResponse: false,
      },
    },
  },

  build: {
    // Optional: `pnpm build` writes a production bundle here for inspection.
    // The Docker image rebuilds its own copy of platform/web/, so this is
    // mainly for `vite preview` smoke-tests.
    outDir: 'web-dist',
    emptyOutDir: true,
    sourcemap: true,
  },

  // platform/web/ is a ~5,000-line zero-dependency vanilla-JS codebase — Vite
  // must NOT try to process it (no JSX, no TS, no bundling beyond what the
  // browser would do anyway).
  esbuild: {
    target: 'es2020',
    legalComments: 'none',
  },
});
