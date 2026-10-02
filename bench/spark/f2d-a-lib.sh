#!/usr/bin/env bash
# Shared helpers for the F2d stage-A operator scripts (sourced by bench/spark/f2d-a-*.sh; not run directly).
# Production oj-serve is on hot standby: nothing here stops or starts it. While our test server holds the GPU the
# standby watcher is paused (~/.spark-standby.pause, operator commit fd28e8d) so a Tower failure cannot start oj-serve
# into the same GPU; the pause is removed by every exit path — unless it was already there (not ours).
REPO=${REPO:-$HOME/octojet-f2d}; OUT=${OUT:-$HOME/octojet-runs/f2d-a}; M=${M:-$HOME/tensorfold/octojet-manifests}
IMAGE=${IMAGE:-tensorfold:0.3.6.2}; NAME=${NAME:-oj-f2d}; PARALLEL=${PARALLEL:-2}; SHARED_GPU=${SHARED_GPU:-0}
READY_DEADLINE=${READY_DEADLINE:-900}
PAUSE=$HOME/.spark-standby.pause; PAUSED_BY_US=0; PID=; DRY=; FAIL=0
PREFIX=${PREFIX:-f2d-a}; HOLD_PAUSE=${HOLD_PAUSE:-0}   # file prefix; HOLD_PAUSE=1 keeps the pause between servers (f3)
mkdir -p "$OUT"; TL="$OUT/$PREFIX-timeline.txt"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$TL"; }
pause_on(){ if [ -e "$PAUSE" ] && [ "$PAUSED_BY_US" = 0 ]; then log "standby pause already present (not ours): left in place"
            elif [ "$PAUSED_BY_US" = 0 ]; then touch "$PAUSE"; PAUSED_BY_US=1; log "standby watcher paused ($PAUSE)"; fi; }
pause_off(){ [ "$PAUSED_BY_US" = 1 ] || return 0; rm -f "$PAUSE"; PAUSED_BY_US=0; log "standby watcher resumed ($PAUSE removed)"; }
cleanup(){ local rc=$?; docker stop -t 60 "$NAME" >/dev/null 2>&1 || true; [ -n "$PID" ] && wait "$PID" 2>/dev/null || true
           pause_off; [ "$rc" = 0 ] || log "exit $rc"; }
trap cleanup EXIT
tenants_clear(){ local t jobs   # fail closed: a failing query refuses the start
  t=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) || { log "$1: nvidia-smi query FAILED: $t"; return 1; }
  jobs=$(docker ps --filter label=llm-research.gpu-job=true --format '{{.Names}}' 2>&1) || { log "$1: docker ps FAILED: $jobs"; return 1; }
  if [ "$SHARED_GPU" = 1 ]; then log "$1: SHARED GPU (owner's choice; numbers labelled), tenants: ${t:-none}; research jobs: ${jobs:-none}"; return 0; fi
  [ -z "$jobs" ] || { log "$1: research GPU job(s) running, refusing to start (ask the operator for a window): $jobs"; return 1; }
  [ -z "$t" ] || { log "$1: another GPU process is present, refusing to start: $t"; return 1; }; }
port_free(){ local i; for i in $(seq 60); do ss -ltn | grep -q ':8080 ' || return 0; sleep 1; done; return 1; }
drop_cache(){ docker run --rm --privileged alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
MOUNTS=(-v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v /srv/ai:/srv/ai:ro
        -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache:/root/.triton")
ENVS=(-e PYTHONPATH=/octojet/engine/src -e PYTHONDONTWRITEBYTECODE=1)
SERVE=(-m tensorfold.cli serve /tf/flashnext-nvfp4-mixed --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8
       --parallel "$PARALLEL" --packed-cache /tf/octojet-cache/packed)
start(){  # $1 label: tenant check, standby pause, page-cache drop, the test server with production's flags, health wait (no warm-up request)
  [ -n "$DRY" ] && return 0
  tenants_clear "$1" || exit 1
  pause_on
  drop_cache || { log "$1: page cache drop FAILED; aborting"; exit 1; }; log "$1: page cache dropped"
  free -g | tee -a "$TL"; log "$1: server start"
  printf '%s\n' "${SERVE[@]}" > "$OUT/$PREFIX-$1-args.txt"
  docker run --rm --gpus all --ipc=host --network host --name "$NAME" "${MOUNTS[@]}" "${ENVS[@]}" -w /octojet/engine \
    --entrypoint python "$IMAGE" "${SERVE[@]}" > "$OUT/$PREFIX-$1-server.log" 2>&1 &
  PID=$!; local t0=$SECONDS
  until grep -q "serving f1 at" "$OUT/$PREFIX-$1-server.log" 2>/dev/null && curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$1: server exited before it was healthy (see $OUT/$PREFIX-$1-server.log)"; exit 1; }
    [ $((SECONDS - t0)) -lt "$READY_DEADLINE" ] || { log "$1: not healthy after ${READY_DEADLINE}s"; exit 1; }; sleep 1
  done
  log "$1: healthy after $((SECONDS - t0))s"
  grep -h "loaded in\|streams of\|startup estimate\|vision:" "$OUT/$PREFIX-$1-server.log" | tee -a "$TL" || true; }
stop(){ [ -n "$DRY" ] && return 0
  docker stop -t 60 "$NAME" >/dev/null 2>&1 || true; [ -n "$PID" ] && wait "$PID" 2>/dev/null || true; PID=; log "$1: stopped"
  port_free || { log "$1: port 8080 not released"; exit 1; }; [ "$HOLD_PAUSE" = 1 ] || pause_off; }
bench(){ python3 bench/prefill_bench.py http://127.0.0.1:8080 f1 "$@" $DRY; }   # DRY=--dry-run while planning: validates, sends nothing
