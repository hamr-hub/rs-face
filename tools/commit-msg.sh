#!/usr/bin/env bash
# Enforce conventional-commit-ish subjects.
#
# Allowed types (lowercase):
#   feat fix perf refactor docs style test build ci chore revert
# Accept:  "fix(s3): send Host header"   "feat: X"   "docs: X"
# Reject:  "update"  "wip"  "fix stuff"  merge noise.
#
# Install: ln -sf ../../tools/commit-msg.sh .git/hooks/commit-msg
# Override: git commit --no-verify
set -uo pipefail
msg_file="$1"
subject="$(head -1 "$msg_file")"

# Merge / fixup / revert commits git generates are fine as-is.
case "$subject" in
  "Merge "*|"Revert "*|fixup!*|squash!*|"chore(merge)"*) exit 0;;
esac

pattern='^(feat|fix|perf|refactor|docs|style|test|build|ci|chore|revert)(\([a-zA-Z0-9._/-]+\))?!?: .+'
if ! [[ "$subject" =~ $pattern ]]; then
  cat >&2 <<EOF

✗ Bad commit subject:
  $subject

Use conventional format:  <type>(<scope>): <summary>
  types: feat fix perf refactor docs style test build ci chore revert
  e.g.   fix(s3): send Host header to match SigV4 canonical headers
         feat(api): add IPv6 link-local SSRF block
         docs: document the pre-push gate

Override only in emergencies: git commit --no-verify
EOF
  exit 1
fi
