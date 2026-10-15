#!/usr/bin/env bash
# F15 diagnostic: identical kernels run 10-45% slower in Octojet's server than in upstream TensorFold's
# (bench/spark/f12-profile.sh, both orders; GPU at full clock, no throttling). Vary one thing at a time and time a cold
# 32k and 128k prompt (client side, after a throwaway long prompt), all on Vontra's MLX 4-bit checkpoint, int8 KV, text:
#   ours        Octojet ($REPO), image $IMAGE (tensorfold:0.3.6.2), --parallel 3              (baseline)
#   ours-nvcr   Octojet, image $UP_IMAGE (nvcr.io/nvidia/pytorch:26.07-py3)                     (image / environment)
#   ours-p1     Octojet, --parallel 1                                                           (memory reserved up front)
#   ours-ck0    Octojet, --prefix-checkpoints 0                                                 (checkpoint snapshots)
#   ours-f8off  Octojet, OCTOJET_LIVE_PREFILL_ROWS=0 OCTOJET_FILL_STREAM=0                      (F8 features)
#   ours-ctx64k Octojet, --parallel 3 --context 65536                                         (small KV, three streams)
#   prod / prod-m0  production's own flags (mixed NVFP4 checkpoint, --parallel 4, --vision, packed cache) with the
#               F17 table margin (default 2 GiB) / OCTOJET_TABLE_MARGIN_GIB=0 (the old plan)
#   up-ours     upstream $UP_TAG ($UP_SRC) in image $IMAGE                                       (image, from the other side)
#   up          upstream $UP_TAG in image $UP_IMAGE                                              (baseline)
# Production oj-serve is STOPPED for the whole run (standing test approval; PROD_STOP), standby pause held and removed on
# every exit path. About 7 cases x ~5 min = 35-40 min. Usage: bash bench/spark/f15-diag.sh [--dry-run]; CASES=... picks.
set -euo pipefail
REPO=${REPO:-$HOME/octojet-main}; OUT=${OUT:-$HOME/octojet-runs/f15}; NAME=${NAME:-oj-f15}; PARALLEL=3
PREFIX=f15; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
UP_TAG=${UP_TAG:-v0.6.2}; UP_IMAGE=${UP_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}; UP_SRC=${UP_SRC:-$HOME/tensorfold/tf-upstream-$UP_TAG}
MLX=${MLX:-$HOME/tensorfold/flashnext-mlx-4bit-mtp}
CASES=${CASES:-ours ours-nvcr ours-p1 ours-ck0 ours-f8off up-ours up}   # F16: CASES="ours ours-p1 ours-ctx64k up"; F17: CASES="prod-m0 prod"
DRYRUN=0; [ "${1:-}" = "--dry-run" ] && DRYRUN=1
cd "$REPO"
[ -e "$MLX/config.json" ] || { echo "error: $MLX missing" >&2; exit 2; }
[ -d "$UP_SRC/tensorfold" ] || { echo "error: $UP_SRC has no tensorfold" >&2; exit 2; }
for m in m32k m128k; do [ -f "$M/$m.json" ] || { echo "error: $M/$m.json missing" >&2; exit 2; }; done
T=$OUT/prompts; mkdir -p "$T"
docker run --rm -v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v "$OUT:/out" -w /octojet --entrypoint python "$IMAGE" \
  bench/cmp_probe.py text --tokenizer /tf/flashnext-mlx-4bit-mtp/tokenizer.json /tf/octojet-manifests/m32k.json \
  /tf/octojet-manifests/m128k.json --out-dir /out/prompts | tee -a "$TL"
for c in $CASES; do
  for m in m32k m128k; do { printf "Diagnostic case %s, cold.\n" "$c"; cat "$T/$m.txt"; } > "$T/$m-$c.txt"; done
  { printf "Diagnostic case %s, throwaway.\n" "$c"; cat "$T/m32k.txt"; } > "$T/throwaway-$c.txt"
done
if [ "$DRYRUN" = 1 ]; then echo "dry-run OK: cases $CASES"; exit 0; fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

serve(){  # $1 case: start its server, wait for /v1/models
  local c=$1 image engine env=() args=(serve "$MLX" --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel 3)
  case "$c" in
    ours) image=$IMAGE; engine=ours;;
    ours-nvcr) image=$UP_IMAGE; engine=ours;;
    ours-p1) image=$IMAGE; engine=ours; args=(serve "$MLX" --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel 1);;
    ours-ck0) image=$IMAGE; engine=ours; args+=(--prefix-checkpoints 0);;
    ours-f8off) image=$IMAGE; engine=ours; env=(-e OCTOJET_LIVE_PREFILL_ROWS=0 -e OCTOJET_FILL_STREAM=0);;
    ours-ctx64k) image=$IMAGE; engine=ours; args+=(--context 65536);;
    prod|prod-m0) image=$IMAGE; engine=ours
      # /srv/ai read-only as in oj-serve: the mixed checkpoint's index points at a snapshot there
      args=(serve /tf/flashnext-nvfp4-mixed --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel 4
            --packed-cache /tf/octojet-cache/packed --vision)
      env=(-v "${DEPS:-$HOME/tensorfold/octojet-pydeps}:/deps:ro" -v /srv/ai:/srv/ai:ro); [ "$c" = prod-m0 ] && env+=(-e OCTOJET_TABLE_MARGIN_GIB=0);;
    up-ours) image=$IMAGE; engine=up;;
    up) image=$UP_IMAGE; engine=up;;
    *) log "unknown case $c"; return 1;;
  esac
  tenants_clear "$c" || exit 1
  drop_cache || { log "$c: page cache drop FAILED"; exit 1; }
  local tag=${image//[^a-zA-Z0-9]/_}
  mkdir -p "$HOME/tensorfold/torch-ext-$engine-$tag" "$HOME/tensorfold/triton-cache-$engine-$tag"
  local src=(-v "$REPO:/octojet:ro" -e PYTHONPATH=/octojet/engine/src:/deps)
  [ "$engine" = up ] && src=(-v "$UP_SRC:/up:ro" -e PYTHONPATH=/up)
  docker run --rm --gpus all --ipc=host --network host --name "$NAME" "${src[@]}" -v "$HOME/tensorfold:/tf" -v "$MLX:$MLX:ro" \
    -v "$HOME/tensorfold/torch-ext-$engine-$tag:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache-$engine-$tag:/root/.triton" \
    -e PYTHONDONTWRITEBYTECODE=1 ${env[@]+"${env[@]}"} --entrypoint python "$image" -m tensorfold.cli "${args[@]}" \
    > "$OUT/$c-server.log" 2>&1 &
  PID=$!; local t0=$SECONDS
  until curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$c: server exited before it was healthy (see $OUT/$c-server.log)"; return 1; }
    [ $((SECONDS - t0)) -lt "$READY_DEADLINE" ] || { log "$c: not healthy after ${READY_DEADLINE}s"; return 1; }; sleep 2
  done
  log "$c: healthy after $((SECONDS - t0))s ($image, $engine)"
  grep -h -i "startup estimate\|streams\|warm-up\|loaded in" "$OUT/$c-server.log" | head -4 | tee -a "$TL" || true; }

ttft(){ python3 bench/cmp_probe.py ttft http://127.0.0.1:8080 f1 --prompt "$1" --label "$2" --out "$OUT/f15-ttft.jsonl" 2>&1 \
  | tee -a "$TL" || { log "$2: FAILED"; FAIL=1; }; }

pause_on
log "WINDOW: stopping production oj-serve ($PROD_STOP)"
bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
port_free || { log "WINDOW: port 8080 still listening"; exit 1; }
for c in $CASES; do
  if serve "$c"; then
    ttft "$T/throwaway-$c.txt" "throwaway-$c"
    free -m > "$OUT/$c-free.txt"; grep -E "^(Cached|MemFree|MemAvailable)" /proc/meminfo >> "$OUT/$c-free.txt"
    for m in m32k m128k; do
      vmstat 1 > "$OUT/$c-$m-vmstat.txt" & VM=$!          # disk reads (bi) and page-cache churn during the prompt
      ttft "$T/$m-$c.txt" "$c-$m"
      kill "$VM" 2>/dev/null || true; wait "$VM" 2>/dev/null || true
    done
  else FAIL=1; fi
  stop "$c"
done
python3 - "$OUT/f15-ttft.jsonl" $CASES <<'PY' | tee "$OUT/f15-summary.txt" | tee -a "$TL"
import json, sys
rows = {r["label"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
import os
out = os.path.dirname(sys.argv[1])
def disk_mb(c, m):
    """MB read from block devices during the prompt: vmstat's bi (KiB/s, one line a second) summed."""
    p = os.path.join(out, f"{c}-{m}-vmstat.txt")
    if not os.path.exists(p):
        return None
    total = 0
    for line in open(p):
        f = line.split()
        if len(f) > 9 and f[0].isdigit():
            total += int(f[8])
    return round(total / 1024)
def cached_gib(c):
    p = os.path.join(out, f"{c}-free.txt")
    for line in open(p) if os.path.exists(p) else []:
        if line.startswith("Cached:"):
            return round(int(line.split()[1]) / 2**20, 1)
print("case | cold 32k TTFT | cold 128k TTFT | disk read during 128k (MB) | page cache after start (GiB)")
for c in sys.argv[2:]:
    g = lambda m: (rows.get(f"{c}-{m}") or {}).get("ttft_s")
    print(f"{c} | {g('m32k')} | {g('m128k')} | {disk_mb(c, 'm128k')} | {cached_gib(c)}")
PY
log "f15 diagnostic done: rc $FAIL (summary $OUT/f15-summary.txt)"
exit "$FAIL"
