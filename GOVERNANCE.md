# Governance — how changes land in rs-face

> Single source of truth for **the process**, not the code layout (that is
> [`STRUCTURE.md`](STRUCTURE.md)) or day-to-day setup
> ([`CONTRIBUTING.md`](CONTRIBUTING.md)). When these disagree, this file says
> how to reconcile them.

This exists because the project grew through direct, unreviewed pushes to
`main` (at one point ~77% of commits were automated bot commits) with no CI
and no consistent commit style. The rules below make the intended quality
bar **enforced**, not merely documented.

## 1. Branches and integration

- **Default branch:** `main`. It must always be green.
- Small, verified changes may be pushed directly to `main` **by the owner**;
  everyone else (including automation/bots) opens a pull request.
- **Automated agents do not push to `main` directly.** They push a branch and
  open a PR; a human merges after CI + review. A bot that only updates data /
  generated artifacts still opens a PR so the diff is inspectable.
- Long-lived feature branches are merged (or rebased) back promptly; there is
  no "develop" branch.

## 2. The gate (what "green" means)

CI (`.github/workflows/ci.yml`) and the local `pre-push` hook run the same
checks for both crates:

| check | core (`rs-face`) | platform (`platform/`) |
|---|---|---|
| format | `cargo fmt --all --check` | `cargo fmt --all --check` |
| lints | `cargo clippy --release --all-targets -- -D warnings` | `cargo clippy --all-targets -- -D warnings` |
| zero-dep | `cargo build --no-default-features --lib` | — |
| tests | `cargo test --lib` | `cargo test` |

Optional GPU/ONNX backends (`metal`, `cuda`, `ort`, `tract`) are **not** built
in CI (they need platform toolchains); validate them manually. The platform
server is run only via Docker; `cargo run -p rsface-server` is unsupported
(see [`CLAUDE.md`](CLAUDE.md)).

### Install the local gates once per clone

```bash
bash tools/install-hooks.sh
```

This symlinks `.git/hooks/pre-push` → `tools/pre-push.sh` and
`.git/hooks/commit-msg` → `tools/commit-msg.sh`. Hooks are scripts in the
repo (reviewable), not untracked machine state. Emergency bypass only:
`git commit --no-verify` / `git push --no-verify`, with a follow-up note.

## 3. Commit discipline

- Subject: **`<type>(<scope>): <summary>`** (scope optional), types:
  `feat fix perf refactor docs style test build ci chore revert`.
  Examples: `fix(s3): send Host header`, `feat(api): block fe80::/10`.
- Present-tense, no trailing period in the subject; keep summary ≤ ~72 chars.
- One logical change per commit. Pure reformatting is its own commit
  (e.g. the `style: cargo fmt` baseline) and must not mix logic.
- **User-visible changes** get an entry under `## [Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md) in the same commit/PR.
- Commit messages state **what and why**, and for detection/model changes the
  measured evidence (see §5). "Works on my machine" is not evidence.

## 4. Dependencies and structure (hard rules)

- The core library is **zero-dependency by default**
  (`default = []`). New crates go behind a cargo feature; CI's
  `--no-default-features --lib` build enforces it (`STRUCTURE.md` §2.4/2.7).
- No new top-level directory; no `Makefile`; no scratch/TODO/plan notes in
  git. Release notes live only in root `CHANGELOG.md` (`STRUCTURE.md` §2).
- Frontend edits happen on the host under `platform/web/`, never inside the
  running container.

## 5. Accuracy claims must be measured

This is a detection project and its reputation rests on not overstating
accuracy. Any change that touches detection thresholds, NMS, recognisers,
cascades, or models must report:

- the cascade/model and images/bench used,
- before/after detection or recognition numbers (and false positives),
- where the evidence is stored (a `research/results`-style artifact, issue,
  or PR description).

Do not widen a cascade or relax a threshold without showing the FP/miss
trade-off. This matches the "honest disclosures" section in CONTRIBUTING.

## 6. Releases

- Version `Cargo.toml`, move `## [Unreleased]` to a dated `## [x.y.z]`, tag
  `vx.y.z`. Semantic versioning; pre-1.0, so bumps stay conservative.
- Release only from a green `main` after the platform Docker build
  (`docker compose -f platform/docker-compose.yml up -d --build`) and
  `bash platform/scripts/docker-smoke.sh` pass.

## 7. Scope of authority

| Action | Owner | Bots/agents | External PRs |
|---|---|---|---|
| push to `main` | yes (small verified) | no — branch + PR | no — PR |
| open PR / branch push | yes | yes | yes |
| merge PR | owner | no | — |
| bump deps / structure rules | owner (edit this file) | no | discussed in issue |
| edit `.github/`, hooks, CI | owner | no | review only |
