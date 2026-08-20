#!/usr/bin/env bash
# 把 testdata/face-walking.mp4 循环切片为本地 HLS,平台用 http://<host>:18080/stream.m3u8
# 当作直播流输入。用法:
#   bash platform/testdata/scripts/serve_hls.sh start
#   bash platform/testdata/scripts/serve_hls.sh stop
#   bash platform/testdata/scripts/serve_hls.sh status
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TESTDATA="$ROOT/testdata"
HLS_DIR="$TESTDATA/hls-loop"
PORT=18080
PIDFILE="/tmp/rsface-hls.pid"

start() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    echo "HLS server already running (pid=$(cat $PIDFILE))"
    return
  fi
  mkdir -p "$HLS_DIR"
  # 1) ffmpeg 切片循环(后台)
  ffmpeg -hide_banner -loglevel error -stream_loop -1 -re -i "$TESTDATA/face-walking.mp4" \
    -c copy -f hls -hls_time 4 -hls_list_size 0 \
    -hls_flags delete_segments+append_list \
    -hls_segment_filename "$HLS_DIR/seg%03d.ts" \
    "$HLS_DIR/stream.m3u8" &
  FFMPEG_PID=$!
  echo $FFMPEG_PID > "$PIDFILE"
  sleep 1
  # 2) python http server
  (cd "$HLS_DIR" && python3 -m http.server $PORT --bind 0.0.0.0 >/tmp/rsface-hls.log 2>&1 &)
  echo $! > /tmp/rsface-hls-http.pid
  sleep 1
  echo "HLS server started"
  echo "  ffmpeg pid:  $FFMPEG_PID"
  echo "  http pid:    $(cat /tmp/rsface-hls-http.pid)"
  echo "  stream URL:  http://localhost:$PORT/stream.m3u8"
}

stop() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat $PIDFILE)" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  if [[ -f /tmp/rsface-hls-http.pid ]]; then
    kill "$(cat /tmp/rsface-hls-http.pid)" 2>/dev/null || true
    rm -f /tmp/rsface-hls-http.pid
  fi
  pkill -f "ffmpeg.*hls-loop" 2>/dev/null || true
  pkill -f "http.server $PORT" 2>/dev/null || true
  echo "HLS server stopped"
}

status() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    echo "running (ffmpeg pid=$(cat $PIDFILE))"
  else
    echo "stopped"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac
