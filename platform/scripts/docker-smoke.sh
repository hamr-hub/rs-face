#!/usr/bin/env bash
# e2e smoke for the docker-deployed rs-face platform.
# Run via: make docker-test   (or: bash platform/scripts/docker-smoke.sh)
#
# Tries to hit every layer of the stack:
#   [1/4] rsface-server /api/health         (HTTP, in-container service)
#   [2/4] rustfs S3 health                  (HTTP, sidecar)
#   [3/4] postgres accept + jobs count      (psql into sidecar)
#   [4/4] POST /api/jobs/image upload       (real upload, optional — uses any test image under platform/testdata/)
#
# Steps 1-3 are mandatory. Step 4 is best-effort: skipped silently if no test
# image is present. The script never exits non-zero on a partial failure, so
# `make docker-test` is safe to run as a smoke that tells you what's working.

set -u
BASE="${BASE:-http://localhost:20080}"
TEST_IMG=""
for f in platform/testdata/lena.ppm platform/testdata/lena.png platform/testdata/*.jpg platform/testdata/*.png; do
    [ -f "$f" ] && TEST_IMG="$f" && break
done

ok()   { printf "  ✅ %s\n" "$*"; }
warn() { printf "  ⚠️  %s\n" "$*"; }
fail() { printf "  ❌ %s\n" "$*"; }

echo "[1/4] rsface-server  → $BASE/api/health"
out=$(curl -sS -m 5 "$BASE/api/health" 2>&1) && ok "$out" || fail "health unreachable: $out"

echo "[2/4] rustfs S3      → http://localhost:19000/minio/health/live"
out=$(curl -sS -m 5 "http://localhost:19000/minio/health/live" 2>&1) && ok "$out" || fail "rustfs unreachable: $out"

echo "[3/4] postgres       → docker exec rsface-postgres"
if out=$(docker exec rsface-postgres pg_isready -U rsface -d rsface 2>&1); then
    ok "$out"
    jobs=$(docker exec rsface-postgres psql -U rsface -d rsface -tAc "SELECT count(*) FROM jobs;" 2>/dev/null | tr -d ' \n')
    [ -n "$jobs" ] && ok "jobs in DB: $jobs" || warn "jobs table empty or missing"
else
    fail "pg_isready failed: $out"
fi

echo "[4/4] image upload   → $BASE/api/jobs/image"
if [ -n "$TEST_IMG" ]; then
    out=$(curl -sS -m 30 -F "file=@$TEST_IMG" "$BASE/api/jobs/image" 2>&1) \
        && ok "$(echo "$out" | head -c 200)" \
        || fail "upload failed: $(echo "$out" | head -c 200)"
else
    warn "no test image under platform/testdata/ — skip"
fi

echo
echo "smoke done. full logs: make docker-logs"
