#!/usr/bin/env bash
# F2d stage A — TIMING replay (owner-approved window, GPU exclusive; ~35 min). Four fresh servers (streamed x 2 repetitions,
# non-streamed with max_tokens 64 x 2), each: cold + four identical 71,444-token calls; every start is preceded by a
# page-cache drop and logged with its UTC time; then the gate (bench/f2d_gate.py). Then two informational servers: the
# multi-burst sequence (P burst -> variant -> P+2k burst -> variant -> P+4k) and the retry probe (P, then the variant and
# its identical retry at once, then P+2k: which kept entry survived?). Production oj-serve is on standby: nothing here
# stops or starts it; the standby watcher is paused while our server holds the GPU.
#   PARALLEL=2 bash bench/spark/f2d-a-replay.sh --caller-timeout 30
set -euo pipefail
source "$(dirname "$0")/f2d-a-lib.sh"
T=; while [ $# -gt 0 ]; do case "$1" in --caller-timeout) T=${2:-}; shift 2;; *) echo "usage: $0 --caller-timeout SECONDS" >&2; exit 2;; esac; done
[[ "$T" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v t="$T" 'BEGIN{exit !(t > 0)}' \
  || { echo "error: --caller-timeout must be a finite positive number of seconds (got '${T:-}')" >&2; exit 2; }
[ "$SHARED_GPU" != 1 ] || { echo "error: the stage-A gate is measured GPU-exclusive (spec section 8); SHARED_GPU=1 is not accepted here" >&2; exit 2; }
cd "$REPO"
for m in cP cPp cP2 cP2p cP3; do [ -f "$M/$m.json" ] || { echo "error: $M/$m.json missing; run bench/spark/f2d-a-manifests.sh first" >&2; exit 2; }; done
render(){ sed "s#__M__#$M#g" "bench/spark/$1" > "$OUT/$1"; }     # the JSON carries no variables: substitute the host manifest dir
render f2d-a-replay.json; render f2d-a-replay-64.json; render f2d-a-bursts.json
replay(){ local label=$1 file=$2; shift 2                          # an error row is logged here and fails the gate later
  bench --replay "$OUT/$file" --label "$label" --server-label "$label" --stage A "$@" --out "$OUT/f2d-a-$label.jsonl" | tee -a "$TL" \
    || log "$label: the runner reported an error row (rc ${PIPESTATUS[0]})"; }
probe(){ local label=$1; shift                                     # one single-manifest request on the current server: a miss is logged, not gated
  bench "$@" --label "$label" --server-label "$label" --stage A --rep 1 --out "$OUT/f2d-a-$label.jsonl" | tee -a "$TL" \
    || log "$label: a probe missed its expectation or errored (rc ${PIPESTATUS[0]}; informational)"; }
plan(){ local rep
  for rep in 1 2; do start "stream-$rep";   replay "stream-$rep"   f2d-a-replay.json    --rep "$rep";             stop "stream-$rep"; done
  for rep in 1 2; do start "nostream-$rep"; replay "nostream-$rep" f2d-a-replay-64.json --rep "$rep" --no-stream; stop "nostream-$rep"; done
  start bursts; replay bursts f2d-a-bursts.json --rep 1; stop bursts
  start retry
  probe retry --manifest "$M/cP.json"  --arm clean:1 --draft on                                                  # P cold: slot A keeps P
  probe retry --manifest "$M/cPp.json" --arm clean:1 --draft on --concurrent 2 --expect-reuse any --expect-cached any   # the variant and its identical retry at once: cold + busy miss
  if [ "$PARALLEL" -le 2 ]; then RETRY=(--expect-reuse none --expect-cached 0); else RETRY=(--expect-reuse extend --expect-cached 71444); fi   # two slots: the retry evicts P; a third slot keeps it
  probe retry --manifest "$M/cP2.json" --arm clean:1 --draft on "${RETRY[@]}"                                                 # P + 2k after the retry storm
  stop retry; }
DRY=--dry-run; plan; DRY=; log "dry-run OK: every planned request parses and no artifact collides"
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || cp "$REPO/COMMIT" "$OUT/commit.txt" 2>/dev/null || true
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
python3 - "$OUT/f2d-a-meta.json" "$T" "$PARALLEL" "$(cat "$OUT/commit.txt" 2>/dev/null || true)" "$IMAGE_ID" "${SERVE[*]}" <<'PY'
import json, sys
out, T, par, commit, image, serve = sys.argv[1:]
json.dump({"caller_timeout_s": float(T), "parallel": int(par), "shared_gpu": False, "commit": commit.strip(), "image_id": image,
           "server_args": serve, "runner_http_timeout_s": 1800.0}, open(out, "w"), indent=1)
PY
plan
python3 bench/f2d_gate.py "$OUT"/f2d-a-stream-1.jsonl "$OUT"/f2d-a-stream-2.jsonl "$OUT"/f2d-a-nostream-1.jsonl "$OUT"/f2d-a-nostream-2.jsonl \
  --caller-timeout "$T" --prompt-tokens 71444 --out "$OUT/f2d-a-gate.json" | tee -a "$TL" || FAIL=${PIPESTATUS[0]}
cp "$M"/cP*.json "$OUT"/
log "stage A replay done: gate rc $FAIL (see $OUT/f2d-a-gate.json; bursts and retry rows are informational)"
exit "$FAIL"
