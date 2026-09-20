#!/usr/bin/env bash
# Install/refresh the local git hooks shipped under tools/.
#
# Git never commits .git/hooks, so each clone must run this once:
#   bash tools/install-hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ln -sf ../../tools/pre-push.sh   .git/hooks/pre-push
ln -sf ../../tools/commit-msg.sh .git/hooks/commit-msg
chmod +x tools/pre-push.sh tools/commit-msg.sh
echo "installed:"
echo "  .git/hooks/pre-push   -> tools/pre-push.sh   (fmt/clippy/zero-dep/test gate)"
echo "  .git/hooks/commit-msg -> tools/commit-msg.sh (conventional subject)"
echo "emergency bypass: git commit/push --no-verify"
