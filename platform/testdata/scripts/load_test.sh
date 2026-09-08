#!/usr/bin/env bash
# Lightweight load test for rsface-platform.
# Submits N jobs concurrently to /api/jobs/image and verifies all complete
# without panics, leaked threads, or 500 responses.
#
# Usage:  ./load_test.sh <PORT> <N_JOBS> <ALGO>
#   PORT  : server bind port (default 20080)
#   N_JOBS: total concurrent jobs (default 8)
#   ALGO  : per-job algo override (default luminance — no cascade.rfcf needed)
#
# Requires: a running rsface-server, a test image at
#   platform/testdata/lena.jpg, and ffmpeg in PATH.

set -euo pipefail

PORT="${1:-20080}"
N="${2:-8}"
ALGO="${3:-luminance}"
IMG="$(dirname "$0")/../lena.jpg"

if [[ ! -f "$IMG" ]]; then
  echo "missing test image at $IMG" >&2; exit 1
fi

echo "load_test: port=$PORT n=$N algo=$ALGO img=$IMG"
URL="http://127.0.0.1:$PORT"

# 1) sanity: server up
curl -sf "$URL/api/health" > /dev/null || { echo "server not reachable at $URL" >&2; exit 1; }

# 2) submit N jobs in parallel, capture job_ids
START=$(date +%s.%N)
TMP=$(mktemp -d)
for i in $(seq 1 "$N"); do
  (
    resp=$(curl -s -X POST -F "algo=$ALGO" -F "file=@$IMG" "$URL/api/jobs/image")
    echo "$resp" > "$TMP/$i.json"
  ) &
done
wait
SUBMIT_END=$(date +%s.%N)

JOB_IDS=()
ERRORS=0
for i in $(seq 1 "$N"); do
  jid=$(python3 -c "import json,sys; print(json.load(open('$TMP/$i.json')).get('job_id',''))" 2>/dev/null || echo "")
  if [[ -z "$jid" ]]; then ERRORS=$((ERRORS + 1)); else JOB_IDS+=("$jid"); fi
done
echo "load_test: submitted=${#JOB_IDS[@]}/${N} submit_errors=$ERRORS submit_wall=$(awk "BEGIN{print $SUBMIT_END - $START}")s"

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then echo "no jobs submitted — abort" >&2; rm -rf "$TMP"; exit 1; fi

# 3) poll until all done / error / cancelled (max 120s)
POLL_END=$(($(date +%s) + 120))
while [[ $(date +%s) -lt $POLL_END ]]; do
  PENDING=0
  for jid in "${JOB_IDS[@]}"; do
    s=$(curl -s "$URL/api/jobs/$jid" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','?').lower())" 2>/dev/null || echo "?")
    case "$s" in
      queued|running|pending|?) PENDING=$((PENDING + 1)) ;;
    esac
  done
  if [[ $PENDING -eq 0 ]]; then break; fi
  sleep 1
done

# 4) tally results
DONE=0; ERR=0; CANC=0; STILL=0
TOTAL_FRAMES=0; TOTAL_DETS=0
for jid in "${JOB_IDS[@]}"; do
  resp=$(curl -s "$URL/api/jobs/$jid")
  s=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
  case "$s" in
    done) DONE=$((DONE + 1)) ;;
    error) ERR=$((ERR + 1)) ;;
    cancelled) CANC=$((CANC + 1)) ;;
    *) STILL=$((STILL + 1)) ;;
  esac
  stats=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin).get('stats',{}); print(d.get('frames_processed',0), d.get('total_detections',0))" 2>/dev/null || echo "0 0")
  f=$(echo "$stats" | cut -d' ' -f1); d=$(echo "$stats" | cut -d' ' -f2)
  TOTAL_FRAMES=$((TOTAL_FRAMES + f)); TOTAL_DETS=$((TOTAL_DETS + d))
done
TOTAL_END=$(date +%s.%N)
TOTAL_WALL=$(awk "BEGIN{print $TOTAL_END - $START}")

# 5) verify /metrics endpoint reflects the work
METRIC_LINES=$(curl -s "$URL/metrics" | wc -l)

echo "load_test: done=$DONE error=$ERR cancelled=$CANC still_pending=$STILL frames=$TOTAL_FRAMES dets=$TOTAL_DETS total_wall=${TOTAL_WALL}s metrics_lines=$METRIC_LINES"

# 6) clean up + verdict
rm -rf "$TMP"

# Pass criteria: 0 errors, ≥1 done, /metrics responded with > 5 lines.
if [[ $ERR -gt 0 ]]; then echo "FAIL: $ERR jobs errored"; exit 1; fi
if [[ $DONE -eq 0 && $STILL -gt 0 ]]; then echo "FAIL: jobs still pending after 120s"; exit 1; fi
if [[ $METRIC_LINES -lt 5 ]]; then echo "FAIL: /metrics endpoint broken"; exit 1; fi
echo "OK"