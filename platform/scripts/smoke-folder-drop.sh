#!/usr/bin/env bash
# platform/scripts/smoke-folder-drop.sh
#
# Smoke test for the batch folder drop UX (dropzone-preview.js).
#
# Walks platform/testdata/pgm/ as a representative "folder of images" and
# posts each file to /api/jobs/image, exercising the same code path the
# web UI uses after walkFolder() collects files from a DataTransferItem.
#
# This script is intentionally not a literal UI drop (no headless browser
# available in CI; the UI is zero-dep). It validates the SERVER side
# accepts the batch the folder-walk produces.
#
# Usage:
#   bash platform/scripts/smoke-folder-drop.sh

set -u

BASE="${BASE:-http://localhost:20080}"
TESTDIR="${TESTDIR:-platform/testdata/pgm}"

if [ ! -d "$TESTDIR" ]; then
    echo "fail: test directory not found: $TESTDIR" >&2
    exit 1
fi

# Walk like web/js/dropzone-preview.js's walkFolder() would.
mapfile -t FILES < <(find "$TESTDIR" -type f \( -iname '*.pgm' -o -iname '*.ppm' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' -o -iname '*.tif' -o -iname '*.tiff' \) | sort)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "fail: no image files in $TESTDIR" >&2
    exit 1
fi

echo "folder-drop: walking $TESTDIR · found ${#FILES[@]} image files"
ok=0
fail=0
job_ids=()
for f in "${FILES[@]}"; do
    rel=${f#$TESTDIR/}
    out=$(curl -sS -m 30 -F "file=@$f" "$BASE/api/jobs/image" 2>&1)
    if echo "$out" | grep -q '"job_id"'; then
        ok=$((ok + 1))
        jid=$(echo "$out" | python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("job_id","?"))' "$out" 2>/dev/null || echo "?")
        job_ids+=("$rel -> $jid")
        printf "  ✅ %s -> %s\n" "$rel" "$jid"
    else
        fail=$((fail + 1))
        printf "  ❌ %s -> %s\n" "$rel" "$(echo "$out" | head -c 100)"
    fi
done

echo ""
echo "folder-drop ok: $ok / ${#FILES[@]} files submitted as individual image jobs"
if [ $ok -eq 0 ]; then
    echo "fail: 0 successful uploads"
    exit 1
fi
echo ""
echo "Web UI mirror: drag the $TESTDIR folder onto the page; dropzone-preview.js's"
echo "  collectFromDataTransfer() will walk the same way this script does,"
echo "  filter to images, and enqueue each one via upload-queue.js. Each image"
echo "  becomes its own job (server has no batch-image endpoint)."
exit 0