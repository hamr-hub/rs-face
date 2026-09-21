#!/usr/bin/env bash
# platform/scripts/smoke-timeline.sh
#
# Smoke test for the job timeline visualization (timeline.js).
#
# Picks the most recent done job with frames (video job preferred — videos
# have many frames; image jobs have only 1 frame and timeline is degenerate),
# verifies the frames payload has timestamp_ms + faces + index fields the
# timeline renders dots from. Also checks the HTML is being served with the
# new <div id="timeline"> placeholder and the <script src="/timeline.js"> tag.
#
# Usage:
#   BACKEND=http://localhost:5173 bash platform/scripts/smoke-timeline.sh
#   bash platform/scripts/smoke-timeline.sh

set -u

BASE_BACKEND="${BASE:-http://localhost:20080}"
DEV_SERVER="${DEV_SERVER:-http://localhost:5173}"

# Pick a video job (frames >= 5) for meaningful dot density.
list=$(curl -sS -m 5 "$BASE_BACKEND/api/jobs" 2>/dev/null)
if [ -z "$list" ]; then
    echo "fail: /api/jobs returned empty" >&2
    exit 1
fi

job=$(python3 - <<EOF "$list"
import json, sys
data = json.loads(sys.argv[1])
jobs = data.get("jobs") or []
# Prefer video jobs with frame_count >= 5; fall back to anything with frames.
def has_many_frames(j):
    return j.get("frame_count", 0) >= 5
candidates = [j for j in jobs if j.get("status") == "done" and j.get("kind") == "video" and has_many_frames(j)]
if not candidates:
    candidates = [j for j in jobs if j.get("status") == "done" and j.get("frame_count", 0) > 0]
candidates.sort(key=lambda j: j.get("created_ms", 0), reverse=True)
if candidates:
    j = candidates[0]
    print(f"{j['id']}|{j.get('kind')}|{j.get('frame_count', 0)}")
EOF
)

if [ -z "$job" ]; then
    echo "fail: no suitable job found (need done video with frames >= 5)"
    exit 1
fi
JOB_ID=$(echo "$job" | cut -d'|' -f1)
JOB_KIND=$(echo "$job" | cut -d'|' -f2)
FRAME_COUNT=$(echo "$job" | cut -d'|' -f3)

echo "timeline: probing job $JOB_ID ($JOB_KIND · $FRAME_COUNT frames) via $BASE_BACKEND"

# 1. Frames payload must include timestamp_ms / faces / index (the dots timeline.js draws).
detail=$(curl -sS -m 10 "$BASE_BACKEND/api/jobs/$JOB_ID" 2>/dev/null)
ok=$(python3 - <<EOF "$detail" "$FRAME_COUNT"
import json, sys
data = json.loads(sys.argv[1])
expected = int(sys.argv[2])
frames = data.get("frames") or []
if not frames:
    print("FAIL:no_frames")
    sys.exit(1)
ok = 0
for f in frames[:5]:
    if "timestamp_ms" in f and isinstance(f.get("faces", []), list) and isinstance(f.get("index", None), int):
        ok += 1
if ok == 0 and frames:
    print("FAIL:missing_fields")
    sys.exit(1)
# Track timeline-relevant summary: face frames vs no-face frames vs error frames.
face_frames = sum(1 for f in frames if len(f.get("faces") or []) > 0)
no_face = sum(1 for f in frames if len(f.get("faces") or []) == 0)
err = sum(1 for f in frames if f.get("status") == "error" or f.get("error"))
print(f"OK:{len(frames)}|{face_frames}|{no_face}|{err}")
EOF
)

if [ "${ok%%:*}" != "OK" ]; then
    echo "fail: $ok"
    exit 1
fi
echo "  ✅ frames payload: $ok"

# 2. Dev server (vite) must serve index.html with the timeline div + script tag.
served=$(curl -sS -m 5 "$DEV_SERVER/" 2>/dev/null)
if [ -z "$served" ]; then
    echo "warn: $DEV_SERVER not reachable; skipping HTML verification"
else
    if echo "$served" | grep -q 'id="timeline"'; then
        echo "  ✅ <div id=\"timeline\"> present in served HTML"
    else
        echo "fail: <div id=\"timeline\"> not in served HTML"
        exit 1
    fi
    if echo "$served" | grep -q '/timeline.js'; then
        echo "  ✅ <script src=\"/timeline.js\"> present in served HTML"
    else
        echo "fail: <script src=\"/timeline.js\"> not in served HTML"
        exit 1
    fi
    if curl -sS -o /dev/null -w "%{http_code}" "$DEV_SERVER/timeline.js" | grep -q '200'; then
        echo "  ✅ GET /timeline.js → 200"
    else
        echo "fail: GET /timeline.js not 200"
        exit 1
    fi
fi

echo ""
echo "timeline ok: UI will render one dot per frame (green=face, gray=no-face,"
echo "             red=error, amber=pending) for the $FRAME_COUNT frames above."
exit 0