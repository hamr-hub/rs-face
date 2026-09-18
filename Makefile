# rs-face developer Makefile
#
# Common dev tasks in one place. Every target is a thin wrapper around a
# cargo invocation — no magic. Run `make help` for the list.
#
# These are *not* required; the same commands work directly. The point is
# to make the common workflow discoverable and uniform.

.DEFAULT_GOAL := help

.PHONY: help build build-zero-dep test test-zero test-doc lint fmt \
        docs bench smoke list-algos list-features clean \
        docker-up docker-down docker-ps docker-logs docker-test \
        docker-restore-pg docker-clean \
        web-dev web-build web-install \
        ci-build ci-test ci-zero ci-features ci-clippy

help: ## show this help
	@awk 'BEGIN {FS = ":.*## "; printf "rs-face dev targets:\n\n"} \
	     /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-15s\033[0m  %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)

build: ## cargo build --release
	cargo build --release

build-zero-dep: ## cargo build --no-default-features (verify the zero-dep claim)
	cargo build --no-default-features --lib

test: ## cargo test (full suite)
	cargo test

test-zero: ## cargo test --no-default-features (zero-dep smoke)
	cargo test --lib --no-default-features

test-doc: ## cargo test --doc (doctests in src/lib.rs + every module //! block)
	cargo test --doc

lint: ## cargo clippy --release --all-targets -- -D warnings
	cargo clippy --release --all-targets -- -D warnings

fmt: ## cargo fmt --all
	cargo fmt --all

docs: ## cargo doc --lib --no-deps (must emit 0 warnings)
	cargo doc --lib --no-deps

bench: ## cargo bench (perfbench; not a CI gate)
	cargo bench

smoke: ## end-to-end smoke test on synthetic input
	./target/release/rs-face test://5 --out /tmp/rsface-smoke
	@if [ -f /tmp/rsface-smoke/manifest.json ]; then echo "smoke ok: yes"; else echo "smoke ok: NO"; fi

list-algos: ## list every algorithm this binary was compiled with
	./target/release/rs-face --list-algos

list-features: ## list every Cargo feature this binary was compiled with
	./target/release/rs-face --list-features

clean: ## cargo clean (also wipes /tmp/rsface-smoke)
	cargo clean
	rm -rf /tmp/rsface-smoke

# --- Docker (platform services) -----------------------------------------------
# Docker is the canonical way to run / test / tear down the platform server and
# its sidecars (rustfs + postgres). The Makefile targets below are thin wrappers
# around `docker compose -f platform/docker-compose.yml`. See CLAUDE.md and
# platform/DOCKER.md for the full ops guide.

COMPOSE_FILE := platform/docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)

docker-up: ## start rustfs + postgres + rsface-server (binds host 0.0.0.0:20080)
	$(COMPOSE) up -d --build
	@echo "Web: http://localhost:20080/  S3: http://localhost:19000/  PG: localhost:15432"

docker-down: ## stop the stack (keeps data/ bind mounts intact)
	$(COMPOSE) down

docker-ps: ## container status + health
	$(COMPOSE) ps

docker-logs: ## tail logs from all 3 services
	$(COMPOSE) logs -f --tail=100

docker-test: docker-up ## e2e smoke: /api/health + image upload + PG count
	@bash platform/scripts/docker-smoke.sh

docker-restore-pg: ## restore PG from data/pg/rsface_dump.sqlc
	@docker exec -i rsface-postgres pg_restore \
		-U rsface -d rsface --clean --if-exists --no-owner --role=rsface \
		< data/pg/rsface_dump.sqlc

docker-clean: ## DESTRUCTIVE: stop stack and wipe data/ bind mounts (NOT dump file)
	$(COMPOSE) down
	rm -rf data/rustfs data/pg/pgdata data/media

# --- Frontend dev (Vite + proxy to docker backend) ---------------------------
# `make docker-up` must be running first. Then `make web-dev` starts a Vite dev
# server on http://localhost:5173/ that serves platform/web/ and proxies
# /api/* + /events to the docker backend at :20080. Edits to any file under
# platform/web/ are picked up by Vite HMR immediately.
#
# All pnpm/Vite tooling lives under platform/web-dev/ (vite.config.js,
# package.json, pnpm-lock.yaml, pnpm-workspace.yaml, .npmrc). This keeps the
# repo root for Rust core + meta only.

WEB_DEV_DIR := platform/web-dev

web-install: ## install Vite + plugins (once, into platform/web-dev/)
	cd $(WEB_DEV_DIR) && pnpm install

web-dev: docker-up ## vite dev server :5173 → proxies /api to docker :20080 (HMR on)
	cd $(WEB_DEV_DIR) && pnpm dev

web-build: ## produce production bundle to web-dist/ (optional; docker builds its own)
	cd $(WEB_DEV_DIR) && pnpm build

# --- CI parity --------------------------------------------------------------
# Targets that match what CI runs, so a local `make ci-*` is a faithful
# pre-push check. See .github/workflows/ci.yml.

ci-build: build smoke ## local proxy for the build + smoke job
ci-test: test test-doc ## local proxy for the lib + doctest job
ci-features: ## verify every documented Cargo feature still builds
	cargo check --features tract-backend --lib
	cargo check --features cuda-backend --lib
ci-clippy: lint ## local proxy for the clippy job