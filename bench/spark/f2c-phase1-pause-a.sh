#!/usr/bin/env bash
set -euo pipefail
REPO=$HOME/octojet-f2c; OUT=$HOME/octojet-runs/f2c-phase1; M=$HOME/tensorfold/octojet-manifests; P=$HOME/tensorfold/octojet-profiles
mkdir -p "$OUT"; cd "$REPO"; source "$P/nsys-flags.env"
TL="$OUT/f2c-phase1-timeline.txt"; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$TL"; }
NAME=oj-f2c; PID=; READY_DEADLINE=900; FAIL=0; DRY=; RESTORED=0; PAUSED=0; Q27_WAS_RUNNING=0
restore(){  # production first, on every exit path: called explicitly at the end of the normal path, by the EXIT trap otherwise
  local rc=$?; [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  [ "$PAUSED" = 1 ] || { log "exit $rc before the pause started: nothing was stopped, nothing to restore"; return 0; }
  docker stop "$NAME" >/dev/null 2>&1 || true; [ -n "$PID" ] && wait "$PID" 2>/dev/null || true
  log "RESTORE (exit $rc): starting oj-serve and the 27B"; local bad=0 st
  <your oj-serve start command> || { st=$?; bad=1; log "RESTORE: oj-serve start FAILED (rc $st)"; }
  <your health check> || { st=$?; bad=1; log "RESTORE: oj-serve health check FAILED (rc $st)"; }
  if [ "$Q27_WAS_RUNNING" = 1 ] && [ "$SHARED_GPU" != 1 ]; then docker start "$Q27" >/dev/null 2>&1 || { bad=1; log "RESTORE: 27B ($Q27) restart FAILED"; }; fi
  if [ "$bad" = 0 ]; then log "PRODUCTION RESTORED"; else log "RESTORE INCOMPLETE, production needs manual action"; exit 70; fi; }
trap restore EXIT
Q27=${Q27:-q27}; SHARED_GPU=${SHARED_GPU:-0}                 # the 27B neighbour's container; SHARED_GPU=1 = owner accepts other tenants (numbers labelled shared GPU)
tenants_clear(){ local t jobs   # fail closed: a failing query refuses the start, in shared-GPU mode too
  t=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) || { log "$1: nvidia-smi query FAILED: $t"; return 1; }
  jobs=$(docker ps --filter label=llm-research.gpu-job=true --format '{{.Names}}' 2>&1) || { log "$1: docker ps FAILED: $jobs"; return 1; }
  if [ "$SHARED_GPU" = 1 ]; then log "$1: SHARED GPU (owner's choice), tenants: ${t:-none}; research jobs: ${jobs:-none}"; return 0; fi
  [ -z "$jobs" ] || { log "$1: research GPU job(s) running, refusing to start (ask the operator to hold them): $jobs"; return 1; }
  [ -z "$t" ] || { log "$1: another GPU process is present, refusing to start: $t"; return 1; }; }
MOUNTS=(-v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v /srv/ai:/srv/ai:ro
        -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache:/root/.triton")
ENVS=(-e PYTHONPATH=/octojet/engine/src -e PYTHONDONTWRITEBYTECODE=1 -e OCTOJET_PREFILL_TIMING=1 -e OCTOJET_NSYS_CAPTURE=1
      -e OCTOJET_PREFILL_TIMING_OUT=/tf/octojet-profiles/timing-records.jsonl)
PARALLEL=${PARALLEL:-3}                                     # production's --parallel at pause time (3 since 2026-09-30 04:30 UTC; was 2 for one hour)
SERVE=(-m tensorfold.cli serve MODEL --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel "$PARALLEL" --packed-cache /tf/octojet-cache/packed)
port_free(){ for i in $(seq 60); do ss -ltn | grep -q ':8080 ' || return 0; sleep 1; done; return 1; }
drop_cache(){ docker run --rm --privileged alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
bench(){ python3 bench/prefill_bench.py http://127.0.0.1:8080 f1 "$@" $DRY; }     # DRY=--dry-run during plan_check: validates, sends nothing
warmup(){ bench --manifest "$M/warm140k.json" --draft on --arm clean:1 --label "$1-warmup" --out "$OUT/f2c-phase1-$1-warmup.jsonl" | tail -1 | tee -a "$TL"; }
start(){  # $1 label, $2 model dir, $3 plain|nsys, $4 nsys report name
  if [ -n "$DRY" ]; then warmup "$1"; return 0; fi
  tenants_clear "$1" || exit 1
  drop_cache || { log "$1: page cache drop FAILED; aborting"; exit 1; }; log "$1: page cache dropped"
  free -g | tee -a "$TL"; log "$1: process start"
  local args=("${SERVE[@]}"); args[3]="$2"                   # SERVE: -m tensorfold.cli serve MODEL ... (MODEL is index 3)
  printf '%s\n' "${args[@]}" > "$OUT/f2c-phase1-$1-args.txt"                                   # the resolved launch
  for f in octojet.json config.json; do [ -f "$HOME/tensorfold/${2#/tf/}/$f" ] && cp "$HOME/tensorfold/${2#/tf/}/$f" "$OUT/f2c-phase1-$1-$f" || true; done
  [ "$3" = nsys ] && rm -f "$P/$4.nsys-rep" "$P/$4.sqlite" "$P/$4-kern.csv" "$P/$4-ranges.json" || true   # never read a stale report
  if [ "$3" = nsys ]; then
    docker run --rm --gpus all --ipc=host --network host --cap-add SYS_ADMIN --name "$NAME" "${MOUNTS[@]}" "${ENVS[@]}" -w /octojet/engine \
      --entrypoint /usr/local/cuda/bin/nsys tensorfold:0.3.6.2 profile --trace=$TRACE,nvtx,osrt --sample=none --cpuctxsw=none $GPUM \
      --capture-range=cudaProfilerApi --capture-range-end=stop --wait=all -f true -o "/tf/octojet-profiles/$4" \
      python "${args[@]}" > "$OUT/f2c-phase1-$1-server.log" 2>&1 &
  else
    docker run --rm --gpus all --ipc=host --network host --name "$NAME" "${MOUNTS[@]}" "${ENVS[@]}" -w /octojet/engine \
      --entrypoint python tensorfold:0.3.6.2 "${args[@]}" > "$OUT/f2c-phase1-$1-server.log" 2>&1 &
  fi
  PID=$!; t0=$SECONDS
  until grep -q "serving f1 at" "$OUT/f2c-phase1-$1-server.log" && curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$1: server exited before it was healthy"; exit 1; }
    [ $((SECONDS - t0)) -lt $READY_DEADLINE ] || { log "$1: not healthy after ${READY_DEADLINE}s"; exit 1; }; sleep 1
  done
  log "$1: healthy after $((SECONDS - t0))s"; grep -h "startup estimate\|streams of" "$OUT/f2c-phase1-$1-server.log" | tee -a "$TL"
  log "$1: selector warm-up (140k)"; warmup "$1"
}
stop(){ [ -n "$DRY" ] && return 0
  docker exec "$NAME" grep -E 'VmHWM|VmRSS' /proc/1/status 2>/dev/null | tee -a "$TL" || log "$1: no RSS telemetry (container already gone?)"   # best effort
  docker stop -t 900 "$NAME" >/dev/null 2>&1 || true; wait "$PID" || true; PID=; log "$1: stopped"; port_free || { log "$1: port not released"; exit 1; }; }   # 15 min grace: an nsys server writes its report at exit
arms(){ local label=$1 manifest=$2 draft=$3; shift 3
  bench --manifest "$manifest" --draft "$draft" --label "$label" --out "$OUT/f2c-phase1-$label.jsonl" "$@" | tee -a "$TL"; }
report_ready(){  # $1 report name, $2 server label: wait (bounded, 15 min) until nsys has written the report — AFTER the server stopped
  [ -n "$DRY" ] && return 0                   # nsys prints "Generated:" then a line holding only the report path once the file is complete
  local i sz prev=-1 stable=0 logf="$OUT/f2c-phase1-$2-server.log"
  for i in $(seq 180); do
    if grep -Eq "^[[:space:]]*/.*/$1\.nsys-rep[[:space:]]*$" "$logf" 2>/dev/null && sz=$(stat -c %s "$P/$1.nsys-rep" 2>/dev/null) && [ "${sz:-0}" -gt 0 ]; then
      if [ "$sz" = "$prev" ]; then stable=$((stable + 1)); else stable=0; fi; prev=$sz
      [ "$stable" -ge 2 ] && { log "$1: report written ($sz bytes, nsys Generated line seen); nsys stats / export / reduction run after production is restored"; return 0; }
    else prev=-1; stable=0; fi                 # a failed poll breaks the run of stable polls: only consecutive successes count
    sleep 5
  done
  log "$1: report NOT ready after 15 min (Generated line: $(grep -Ec "^[[:space:]]*/.*/$1\.nsys-rep" "$logf" 2>/dev/null || echo 0), last size: $prev)"; return 1; }
reduce(){  # $1 report name: kernel-row validation (nsys stats auto-exports sqlite), sqlite export + range reduction — CPU-heavy, AFTER restore
  docker run --rm -v "$HOME/tensorfold:/tf" --entrypoint /usr/local/cuda/bin/nsys tensorfold:0.3.6.2 stats --report cuda_gpu_kern_sum --format csv "/tf/octojet-profiles/$1.nsys-rep" > "$P/$1-kern.csv" 2>"$P/$1-stats.err" || { log "$1: nsys stats failed"; return 1; }
  python3 - "$P/$1-kern.csv" <<'PY' || { log "$1: no kernel rows in the report"; return 1; }
import csv, sys
raw = open(sys.argv[1]).read().splitlines()                                          # nsys stats prints "Generating SQLite file ..." /
start = next((i for i, l in enumerate(raw) if l.startswith("Time (%)")), None)       # "Processing [...]" before the CSV header
if start is None:
    print(f"{sys.argv[1]}: no 'Time (%)' header line found; first lines: {raw[:3]}"); sys.exit(1)
lines = [l for l in raw[start:] if l.strip() and not l.startswith("#")]
rows = list(csv.DictReader(lines))
cols = list(rows[0].keys()) if rows else []
inst = next((k for k in cols if k and k.startswith("Instances")), None)
ok = bool(rows) and inst is not None and "Name" in cols and any(
    r[inst].strip().isdigit() and int(r[inst]) > 0 and r["Name"].strip() for r in rows)
print(f"{sys.argv[1]}: {len(rows)} kernel rows; columns {cols}")
sys.exit(0 if ok else 1)
PY
  docker run --rm -v "$HOME/tensorfold:/tf" --entrypoint /usr/local/cuda/bin/nsys tensorfold:0.3.6.2 export --type sqlite -f true -o "/tf/octojet-profiles/$1.sqlite" "/tf/octojet-profiles/$1.nsys-rep" || { log "$1: sqlite export failed"; return 1; }
  python3 bench/nsys_ranges.py "$P/$1.sqlite" --out "$P/$1-ranges.json" | tee -a "$TL" || { log "$1: reducer rejected the capture (no admission bounds / no ranges)"; return 1; }; }
plan(){  # the measured sequence; with DRY set only prefill_bench --dry-run runs (servers, stops and report checks are skipped)
  if [ "${ONLY:-}" != S2 ]; then                 # ONLY=S2 re-runs the nsys server alone (S1 already measured)
    start S1 /tf/flashnext-nvfp4-mixed plain -
    arms S1-32k  "$M/m32k.json"  on --arm clean:2 --arm timing:2
    arms S1-128k "$M/m128k.json" on --arm clean:2 --arm timing:2 --arm histogram:1
    arms S1-210k "$M/m210k.json" on --arm clean:2 --arm timing:2 --arm histogram:1
    stop S1
  fi
  start S2 /tf/flashnext-nvfp4-mixed nsys f2c-210k
  arms S2-210k "$M/m210k.json" on --arm profile:1 --timeout 3600 || { rc=$?; [ -z "$DRY" ] || { log "dry-run FAILED (rc $rc)"; exit "$rc"; }; log "S2: profile arm row invalid (runner rc $rc); the report may still hold the kernels"; FAIL=1; }
  stop S2                                          # nsys finalises the report when the server exits (capture-range-end=stop keeps the server alive until then)
  report_ready f2c-210k S2 || { log "S2: REPORT INVALID"; touch "$OUT/S2-report-invalid"; FAIL=1; }; }
DRY=--dry-run; plan; DRY=; log "dry-run OK: every planned request parses and no artifact collides"   # BEFORE production stops
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || cp "$REPO/COMMIT" "$OUT/commit.txt" 2>/dev/null || true
printf '%s\n' "${SERVE[@]}" > "$OUT/serve-args.txt"; docker image inspect tensorfold:0.3.6.2 --format '{{.Id}}' > "$OUT/image-id.txt"; cp "$P/nsys-flags.env" "$OUT/"
q27=$(docker ps -q -f "name=^${Q27}$" 2>&1) || { log "docker ps FAILED before the pause: $q27"; exit 1; }; [ -n "$q27" ] && Q27_WAS_RUNNING=1
PAUSED=1; log "PAUSE A START: stopping oj-serve and the 27B"; <your oj-serve stop command>; [ "$SHARED_GPU" = 1 ] || docker stop "$Q27" >/dev/null 2>&1 || true; port_free || { log "port 8080 still listening"; exit 1; }
plan
restore                                                     # production back BEFORE nsys stats / export / reduction
reduce f2c-210k || { log "S2: REDUCTION INVALID"; touch "$OUT/S2-report-invalid"; FAIL=1; }
cp "$M"/*.json "$OUT"/ ; cp "$P"/nsys-flags.env "$P"/*-ranges.json "$P"/*-kern.csv "$P"/timing-records.jsonl "$OUT"/ 2>/dev/null || true
[ "$FAIL" = 0 ] || log "PAUSE A: an nsys capture was INVALID (marker file in $OUT); the results task must not use it"
exit "$FAIL"
