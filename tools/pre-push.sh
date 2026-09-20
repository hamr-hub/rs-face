#!/usr/bin/env bash
# Pre-push quality gate for rs-face.
#
# Mirrors .github/workflows/ci.yml so a push to main cannot land code that
# CI would reject. Runs fast (incremental). Two independent crates are
# checked: the zero-dep core library and the platform server.
#
# Install (from repo root):
#   ln -sf ../../tools/pre-push.sh .git/hooks/pre-push
#
# Override in an emergency:
#   git push --no-verify
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root" || exit 2

fail=0
say() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*"; fail=1; }

echo "▶ cargo fmt --check (core)"
cargo fmt --all --check || say "core not formatted: run 'cargo fmt'"

echo "▶ cargo fmt --check (platform)"
( cd platform && cargo fmt --all --check ) || say "platform not formatted: 'cargo fmt --manifest-path platform/Cargo.toml'"

echo "▶ clippy -D warnings (core, release/all-targets)"
cargo clippy --release --all-targets -- -D warnings || say "core clippy failed"

echo "▶ clippy -D warnings (platform)"
cargo clippy --manifest-path platform/Cargo.toml --all-targets -- -D warnings \
  || say "platform clippy failed"

echo "▶ zero-dep core lib build"
cargo build --no-default-features --lib || say "zero-dep lib build broke (STRUCTURE §2.4/2.7)"

echo "▶ rustdoc deny warnings (core)"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items \
  || say "core rustdoc warnings: broken intra-doc links"

echo "▶ rustdoc deny warnings (platform)"
( cd platform && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items ) \
  || say "platform rustdoc warnings"

echo "▶ core lib tests"
cargo test --lib || say "core tests failed"

echo "▶ platform tests"
cargo test --manifest-path platform/Cargo.toml || say "platform tests failed"

if [ "$fail" -ne 0 ]; then
  printf '\n\033[1;31mPush rejected. Fix the above (or push --no-verify in a true emergency).\033[0m\n'
  exit 1
fi
printf '\n\033[1;32m✓ pre-push gates passed\033[0m\n'
