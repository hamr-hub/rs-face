# Repository Structure — Conventions & Hard Constraints

> **This document is read first by every AI agent and human contributor.**
> It defines the canonical directory layout, what each top-level path is *for*,
> and what an agent **must not** do. If a change conflicts with this file, this
> file wins. See also [`CONTRIBUTING.md`](CONTRIBUTING.md) for the longer
> rationale and [`CLAUDE.md`](CLAUDE.md) for AI-facing operational rules.

## 1. Canonical layout

Every file in the repo has exactly one home. Top-level directories are sparse
by design — if a new top-level dir is being proposed, that's a smell.

```
rs-face/
├── Cargo.toml / Cargo.lock / clippy.toml / rustfmt.toml
│   Cargo workspace metadata + tooling config. **No code here.**
│   **There is no Makefile** — convenience wrappers are inline cargo /
│   docker compose / pnpm commands in the docs, or shell scripts under
│   `platform/scripts/` and `tools/`.
├── README.md / CHANGELOG.md / CONTRIBUTING.md / STRUCTURE.md / LICENSE / CLAUDE.md
│   Public-facing + agent-facing documentation. One file per concern:
│   - README.md          — what rs-face is, quick start, accuracy tables
│   - CHANGELOG.md       — release history (Keep a Changelog format)
│   - CONTRIBUTING.md    — dev workflow, full tree, module-split rationale
│   - STRUCTURE.md       — this file: layout + hard constraints
│   - CLAUDE.md          — AI agent operational rules
│   - LICENSE            — MIT
│
├── src/                 — library + CLI source (Rust, zero-dep default)
│   ├── lib.rs           — library entrypoint; declares every pub mod
│   ├── main.rs          — `rs-face` CLI binary
│   ├── <single-file modules>.rs | <multi-file modules>/haar/, cnn/, gpu/, image/, ...
│   └── bin/             — explicit `[[bin]]` targets (autobins=false in Cargo.toml)
│
├── tests/               — integration tests + ignored bench runners
├── benches/             — criterion-style benches (harness = false)
├── examples/            — `cargo run --example <NAME>` cookbook (public)
│
├── docs/                — long-form design + accuracy documentation
│   ├── INDEX.md         — entry point; read this first
│   └── *.md             — per-topic deep dives (algorithms, format, GPU, ...)
│
├── tools/               — Python + shell helpers (NOT part of the Rust crate)
│
├── platform/            — deployment-grade server + web UI (Docker-only)
│   ├── Cargo.toml / Cargo.lock   — platform-server crate (separate crate)
│   ├── Dockerfile / Dockerfile.gpu / docker-compose.yml / docker-compose.gpu.yml
│   ├── DOCKER.md / DOCKER_SIZING.md / MINIMUM_CONFIG.md / PROFILE.md / README.md
│   ├── docs/            — platform design + roadmap + SDK notes
│   ├── migrations/      — SQL migrations (0001_init.sql, 0002_*.sql, ...)
│   ├── scripts/         — bash + node smoke / screenshot scripts
│   ├── server/src/      — axum HTTP API + worker pool (Rust)
│   ├── testdata/        — small images checked in; large mp4/HLS gitignored
│   └── web/             — zero-build vanilla-JS frontend (served by the server)
│       └── index.html + *.js + *.css + theme.css
│
├── .github/workflows/   — CI: build / test / fmt / clippy (ubuntu + macOS)
│
└── data/                — gitignored, bind-mounted runtime state
    ├── rustfs/          — object store contents
    ├── pg/pgdata/       — postgres data directory
    └── media/           — raw uploads + extracted frames mirror
```

### 1.1 Where a new file belongs — quick lookup

| You're adding | Home directory |
|---|---|
| New algorithm module | `src/<algo>.rs` or `src/<algo>/` (multi-file) |
| New `[[bin]]` target | `src/bin/<name>.rs` (and list it in `Cargo.toml`) |
| Integration test | `tests/<name>.rs` |
| Criterion bench | `benches/<name>.rs` (`harness = false`) |
| Cookbook snippet | `examples/<name>.rs` |
| Long-form doc | `docs/<topic>.md` (link from `docs/INDEX.md`) |
| Python / shell helper | `tools/<name>.{py,sh}` |
| Platform API handler | `platform/server/src/<module>.rs` |
| Platform SQL migration | `platform/migrations/NNNN_<topic>.sql` |
| Platform frontend module | `platform/web/<name>.{js,css}` (load in `index.html`) |
| Platform ops script | `platform/scripts/<name>.{sh,py,cjs}` |
| Small reference image | `platform/testdata/<name>.{jpg,png,pgm}` |
| Release note | append to `CHANGELOG.md` (Keep a Changelog) |
| Agent scratch / working plan | **DO NOT commit.** Use your local branch. |

---

## 2. Hard constraints — Do NOT

> Each rule below has been broken in this repo's git history. Read them
> as **enforced** rather than advisory.

### 2.1 Do NOT add a new top-level directory

`s/`、`examples/`、`tests/`、`benches/`、`docs/`、`tools/`、`platform/`
`data/`、`scripts/` 覆盖全部合法用途。Rejected names: `core/`, `misc/`,
`notes/`, `scratch/`, `helpers/`, `common/`, `shared/`, `lib2/`, `tmp/`.

If a file can't be placed under one of the existing top-level dirs, the
proposal needs an explicit PR-level justification.

### 2.2 Do NOT commit working plans / scratch notes / AI internal docs

Rejected filenames anywhere in the repo:

```
TASK_PLAN.md         *_plan.md        *_notes.md            NOTES.md
TODO.md              *_todo.md        *_fix.md              *_wip.md
CHANGELOG_<TOPIC>.md (in platform/*)
```

The historical ones were:
- `TASK_PLAN.md` (root) — deleted 2026-09
- `core/MULTI_ALGO.md` — deleted 2026-09 (along with empty `core/`)
- `platform/CHANGELOG_CNN.md`, `platform/CHANGELOG_PERF.md` — deleted 2026-09
- `platform/web/CHANGELOG_DOUBAO.md`, `CHANGELOG_ENHANCE.md`, `CHANGELOG_FIXES.md` — deleted 2026-09

Their content is either superseded by `docs/algorithms.md`, `docs/architecture.md`,
and the git history, or it was personal working notes that should never have been
committed. **Release history lives in `CHANGELOG.md` at the repo root.**

### 2.3 Do NOT put a `CHANGELOG_*.md` / `NOTES.md` / `TODO.md` under `platform/`

Same rule, scoped: history → `CHANGELOG.md`, roadmap → `platform/docs/ROADMAP.md`,
TODO → GitHub issues.

### 2.4 Do NOT add deps to `[lib]` without prior discussion

The library is **zero-dep by default**. New dependencies must go behind a
Cargo `feature` flag. The CI `zero-dep-build` job (`cargo build --no-default-features --lib`)
enforces this and will fail on a default-build dep.

### 2.5 Do NOT edit `platform/web/` inside the Docker container

The Docker image bundles `platform/web/` at build time. **All frontend work
happens on the host** under `platform/web/`, with Vite HMR via `pnpm dev`
(repo root, since `package.json` lives there). Edits to `/app/web/` inside
the container are wiped on the next `docker compose up -d --build`.

### 2.6 Do NOT run `cargo run` for `rsface-server` on the host

The platform server needs ffmpeg + a running S3 endpoint + a running Postgres,
all of which the compose stack wires together. Use
`docker compose -f platform/docker-compose.yml up -d --build` then
`bash platform/scripts/docker-smoke.sh`. `cargo run -p rsface-server` is **not**
a supported path.

### 2.7 Do NOT break the zero-dep build

If your patch adds a `use` statement for a non-std crate at the top of a
`src/*.rs` file, gate it behind `#[cfg(feature = "...")]`. CI verifies.

### 2.8 Do NOT introduce a Makefile back into the repo

This repo deliberately has no Makefile. Convenience wrappers live as plain
shell scripts under `platform/scripts/`, `tools/`, or as inline `docker compose`
/ `cargo` / `pnpm` commands in the docs. If a Makefile reappears, delete it.

---

## 3. Pre-commit self-check (for AI agents)

Before `git add`, run through this list. If any item fails, **don't commit**.

```
[ ] git status does not show new *.rfcf / .vite/ / __pycache__/ / node_modules/
[ ] Every new file fits one row of §1.1 lookup table
[ ] No file deletion without `git log -- <path>` first
[ ] If src/ touched: cargo build --no-default-features --lib still passes
[ ] If src/ touched: cargo clippy --release --all-targets -- -D warnings passes
[ ] If platform/server/ touched: cargo build --manifest-path platform/Cargo.toml passes
[ ] If platform/web/ touched: index.html loads every new <script>/<link>
[ ] No CHANGELOG_*.md / NOTES.md / TODO.md / *_plan.md / *_notes.md added
[ ] No new top-level directory
[ ] CHANGELOG.md updated if this is a user-visible change
```

If a test fails or a check is unclear, **STOP** and ask before committing.

---

## 4. Why this file exists

In August–September 2026 this repo accumulated:

- an empty `core/` directory with no purpose
- 5 AI-scratch CHANGELOG files under `platform/`
- a Makefile that references a non-existent `platform/web-dev/` directory
- a `.vite/` cache directory that an agent forgot to gitignore
- a `TASK_PLAN.md` root file with personal Chinese working notes

Every one of these was preventable. `STRUCTURE.md` is the contract that makes
the next one impossible to introduce without it being a deliberate, flagged
violation.