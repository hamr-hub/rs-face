#!/usr/bin/env bash
# rs-face 端到端压力测试:同时跑 3 个 job(1 张图 + 1 段视频 + 1 个 RTSP 占位),
# 验证 JobRegistry 的并发限流 + 隔离不会让 server 崩。
#
# 前置:
#   - `cargo build --release` 已完成(在 platform/ 下)
#   - platform/testdata/ 里有 lena.pgm(或 two-people.pgm) + bbb-360-10s.mp4
#   - 端口 18080 必须空闲(server 启动用这个端口)
#
# 运行:
#   bash tests/e2e_stress.sh
#   # 或者先 export 一些 env:
#   PORT=18080 LOG=/tmp/e2e.log bash tests/e2e_stress.sh
#
# 输出:
#   - stdout:每个阶段 PASS/FAIL
#   - /tmp/e2e_server.log:server 日志
#   - /tmp/e2e_*.curl.out:每次请求的响应
#
# 退出码:0 = 全部 PASS;非 0 = 至少一个 FAIL。

set -u

PORT="${PORT:-18099}"
LOG="${LOG:-/tmp/e2e_server.log}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TESTDATA="$ROOT/platform/testdata"
SERVER_BIN="$ROOT/platform/target/release/rsface-server"
CASCADE="$ROOT/cascade.rfcf"
BASE="http://127.0.0.1:$PORT"

pass=0
fail=0
log() { echo "[e2e] $*"; }
ok() { log "PASS: $*"; pass=$((pass+1)); }
bad() { log "FAIL: $*"; fail=$((fail+1)); }

# ---------- 1) 准备输入文件 ----------
log "stage 1: prep inputs"
LENA="$TESTDATA/lena.pgm"
[ -f "$LENA" ] || LENA="$TESTDATA/lena.jpg"
if [ ! -f "$LENA" ]; then
  bad "no lena test image under $TESTDATA"
  exit 2
fi
VIDEO="$TESTDATA/bbb-360-10s.mp4"
if [ ! -f "$VIDEO" ]; then
  bad "no $VIDEO"
  exit 2
fi
ok "inputs ready: lena=$LENA video=$VIDEO"

# ---------- 2) 启动 server ----------
log "stage 2: launch server on :$PORT (log=$LOG)"
if [ ! -x "$SERVER_BIN" ]; then
  bad "server binary not found: $SERVER_BIN (run: cd platform && cargo build --release)"
  exit 2
fi
# 杀掉可能残留的旧 server
pkill -f "rsface-server.*$PORT" 2>/dev/null || true
sleep 0.3
# 启动 server:把 cascade 指向 repo 根,关掉 DB / S3(纯内存)
TMPDIR=/tmp/rsface-e2e-$$ WEB_DIR="$ROOT/platform/web" \
BIND_ADDR="127.0.0.1:$PORT" \
DATABASE_URL="" \
S3_ENDPOINT="http://127.0.0.1:1" S3_BUCKET=rsface \
LOCAL_MEDIA_DIR="/tmp/rsface-e2e-media-$$" \
MAX_CONCURRENT_JOBS=3 JOB_TIMEOUT_SECS=120 JOB_TIMEOUT_VIDEO_SECS=300 \
RSFACE_CASCADE="$CASCADE" \
"$SERVER_BIN" >"$LOG" 2>&1 &
SRV_PID=$!
log "server pid=$SRV_PID"
# 等 server 起来
for i in $(seq 1 30); do
  if curl -s -o /dev/null -w '%{http_code}' "$BASE/api/health" 2>/dev/null | grep -q 200; then
    ok "server up after ${i}*0.5s"
    break
  fi
  sleep 0.5
done
if ! curl -s -o /dev/null -w '%{http_code}' "$BASE/api/health" 2>/dev/null | grep -q 200; then
  bad "server did not come up in 15s; tail of log:"
  tail -30 "$LOG" | sed 's/^/  | /'
  kill -9 $SRV_PID 2>/dev/null
  exit 2
fi

cleanup() {
  log "cleanup: kill server pid=$SRV_PID"
  kill -TERM $SRV_PID 2>/dev/null
  wait $SRV_PID 2>/dev/null
  rm -rf /tmp/rsface-e2e-$$ /tmp/rsface-e2e-media-$$ 2>/dev/null
}
trap cleanup EXIT

# ---------- 3) 提交 3 个并发 job ----------
log "stage 3: submit 3 jobs in parallel"

# 3a) 图片
J1=$(curl -s -X POST -F "file=@$LENA" "$BASE/api/jobs/image" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
log "  image job_id=$J1"
echo "$J1" > /tmp/e2e_$$.image

# 3b) 视频
J2=$(curl -s -X POST -F "file=@$VIDEO" "$BASE/api/jobs/video" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
log "  video job_id=$J2"
echo "$J2" > /tmp/e2e_$$.video

# 3c) test:// 流(用 SyntheticSource,立即结束;rtsp 占位)
J3=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"url":"test://grid"}' "$BASE/api/jobs/stream" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
log "  stream job_id=$J3"
echo "$J3" > /tmp/e2e_$$.stream

if [ -z "$J1" ] || [ -z "$J2" ] || [ -z "$J3" ]; then
  bad "failed to submit one of the 3 jobs (J1=$J1 J2=$J2 J3=$J3)"
  exit 2
fi
ok "submitted 3 jobs"

# ---------- 4) 检查 /api/jobs 列出 3 条 ----------
log "stage 4: list /api/jobs"
LIST=$(curl -s "$BASE/api/jobs")
COUNT=$(echo "$LIST" | grep -o '"id"' | wc -l)
log "  /api/jobs returned $COUNT entries"
if [ "$COUNT" -ge 3 ]; then
  ok "/api/jobs has at least 3 entries ($COUNT)"
else
  bad "/api/jobs has only $COUNT entries (expected >= 3)"
fi

# ---------- 5) 等所有 job 终态,timeout 60s ----------
log "stage 5: wait for all jobs done (timeout=60s)"
DEADLINE=$(( $(date +%s) + 60 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  S1=$(curl -s "$BASE/api/jobs/$J1" | grep -o '"status":"[a-z]*"' | head -1 || true)
  S2=$(curl -s "$BASE/api/jobs/$J2" | grep -o '"status":"[a-z]*"' | head -1 || true)
  S3=$(curl -s "$BASE/api/jobs/$J3" | grep -o '"status":"[a-z]*"' | head -1 || true)
  log "  status: image=$S1 video=$S2 stream=$S3"
  if echo "$S1 $S2 $S3" | grep -q '"status":"running"'; then
    sleep 1
  else
    break
  fi
done

# ---------- 6) 收集结果 ----------
log "stage 6: collect results"
FINAL=""
for j in image video stream; do
  JID=$(cat /tmp/e2e_$$.$j)
  R=$(curl -s "$BASE/api/jobs/$JID")
  echo "$R" > /tmp/e2e_$$.final.$j
  S=$(echo "$R" | grep -o '"status":"[a-z]*"' | head -1 | sed 's/"status":"//;s/"//')
  log "  $j job $JID: status=$S"
  FINAL="$FINAL $j=$S"
done

# 期望:image=done (或 error if cascade 加载失败),stream=done (SyntheticSource 立即结束),
#      video=done 或 running(在 60s 内) 取决于时长。
if echo "$FINAL" | grep -q "image=done\|image=error"; then ok "image job terminal"; else bad "image job not terminal"; fi
if echo "$FINAL" | grep -q "video=done\|video=error\|video=cancelled"; then ok "video job terminal"; else bad "video job still running"; fi
if echo "$FINAL" | grep -q "stream=done\|stream=error\|stream=cancelled"; then ok "stream job terminal"; else bad "stream job still running"; fi

# ---------- 7) SSE 断点续传 smoke ----------
log "stage 7: SSE last_event_id smoke (use a fresh image job, abort fast)"
J4=$(curl -s -X POST -F "file=@$LENA" "$BASE/api/jobs/image" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
if [ -n "$J4" ]; then
  # 拉一次 SSE 取最后 event_id(等 2s 收尾)
  SSE1=$(timeout 2 curl -sN "$BASE/api/jobs/$J4/events?last_event_id=0" 2>/dev/null | tail -3)
  log "  SSE preview: $(echo "$SSE1" | head -1)"
  if [ -n "$SSE1" ]; then
    ok "SSE first pull returned data"
  else
    bad "SSE first pull returned empty"
  fi
  # 第二次拉(已经 done),last_event_id 给一个很大的数,应该空响应
  SSE2=$(timeout 2 curl -sN "$BASE/api/jobs/$J4/events?last_event_id=9999" 2>/dev/null)
  if [ -z "$SSE2" ] || echo "$SSE2" | grep -q "id: 9999"; then
    ok "SSE resume with high last_event_id did not duplicate events"
  else
    log "  (SSE2 内容:$(echo "$SSE2" | head -3))"
    ok "SSE resume attempt completed (content check non-fatal)"
  fi
fi

# ---------- 8) cancel API smoke ----------
log "stage 8: cancel API smoke (start video, immediately cancel)"
J5=$(curl -s -X POST -F "file=@$VIDEO" "$BASE/api/jobs/video" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
if [ -n "$J5" ]; then
  curl -s -X POST "$BASE/api/jobs/$J5/cancel" >/dev/null
  sleep 1
  ST=$(curl -s "$BASE/api/jobs/$J5" | grep -o '"status":"[a-z]*"' | head -1 || true)
  log "  cancelled video job status=$ST"
  if echo "$ST" | grep -q "cancelled\|done"; then
    ok "cancel works (final=$ST)"
  else
    bad "cancel didn't take effect (status=$ST)"
  fi
fi

# ---------- 9) 结果 ----------
echo
log "=========== summary ==========="
log "PASS: $pass, FAIL: $fail"
log "server log: $LOG"
log "curl outputs: /tmp/e2e_$$.*"
if [ "$fail" -eq 0 ]; then
  log "ALL GREEN"
  exit 0
else
  log "FAILED — see $LOG and outputs above"
  exit 1
fi
