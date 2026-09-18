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

# --- CI parity --------------------------------------------------------------
# Targets that match what CI runs, so a local `make ci-*` is a faithful
# pre-push check. See .github/workflows/ci.yml.

ci-build: build smoke ## local proxy for the build + smoke job
ci-test: test test-doc ## local proxy for the lib + doctest job
ci-features: ## verify every documented Cargo feature still builds
	cargo check --features tract-backend --lib
	cargo check --features cuda-backend --lib
ci-clippy: lint ## local proxy for the clippy job