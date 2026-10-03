#!/usr/bin/env bash
# F8 step 1: where a live reply's pause goes while a long prompt fills, and what smaller prompt chunks would buy.
# For each case in CASES (chunk rows:live chunk rows; default 2048:0 2048:1024 2048:512, 2,048 is production's with
# --vision; live 0 = the old fixed chunks) a test server with
# production's flags (+ --vision) and OCTOJET_ROUND_LOG on:
#   gap    a stream decodes while a cold m128k prompt is sent: every inter-token gap of the stream, the prompt's TTFT
#   alone  a different cold 128k prompt with nothing else running: its TTFT (the cost of the chunk size)
#   rounds per round: seconds in the prompt chunk and in the decode step, with the prompt's position (rounds-<rows>.jsonl)
# Production oj-serve is STOPPED for the whole run (standing test approval; PROD_STOP), standby pause held, restored on
# every exit path by the operator's watcher. About 25-30 min.
# Usage: bash bench/spark/f8-measure.sh [--dry-run]
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f8}; OUT=${OUT:-$HOME/octojet-runs/f8}; NAME=${NAME:-oj-f8}; PARALLEL=3
PREFIX=f8; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
DEPS=${DEPS:-$HOME/tensorfold/octojet-pydeps}
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
CASES=${CASES:-2048:0 2048:1024 2048:512}   # chunk rows : live chunk rows (0 = off, the old behaviour)
MOUNTS+=(-v "$DEPS:/deps:ro" -v "$OUT:/out")
SERVE+=(--vision)
DRYRUN=0; [ "${1:-}" = "--dry-run" ] && DRYRUN=1
cd "$REPO"
[ -f "$M/m128k.json" ] || { echo "error: $M/m128k.json missing" >&2; exit 2; }
[ -d "$DEPS" ] || { echo "error: $DEPS missing" >&2; exit 2; }
T=$OUT/prompts; mkdir -p "$T"
docker run --rm -v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v "$OUT:/out" -w /octojet --entrypoint python "$IMAGE" \
  bench/cmp_probe.py text --tokenizer /tf/flashnext-nvfp4-mixed/tokenizer.json /tf/octojet-manifests/m128k.json \
  --out-dir /out/prompts | tee -a "$TL"
for r in $CASES; do
  { printf "Chunk-size run %s, a different prompt.\n" "$r"; cat "$T/m128k.txt"; } > "$T/m128k-alone-$r.txt"
  { printf "Chunk-size run %s, the gap prompt.\n" "$r"; cat "$T/m128k.txt"; } > "$T/m128k-gap-$r.txt"
done
printf "Say hello in five words." > "$T/warm.txt"
if [ "$DRYRUN" = 1 ]; then echo "dry-run OK: cases $CASES, prompts in $T"; exit 0; fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
pause_on
log "WINDOW: stopping production oj-serve ($PROD_STOP)"
bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
port_free || { log "WINDOW: port 8080 still listening"; exit 1; }
probe(){ python3 bench/cmp_probe.py "$@" 2>&1 | tee -a "$TL" || { log "probe $1 FAILED"; FAIL=1; }; }
for r in $CASES; do
  rm -f "$OUT/rounds-$r.jsonl"
  ENVS=(-e PYTHONPATH=/octojet/engine/src:/deps -e PYTHONDONTWRITEBYTECODE=1 -e TENSORFOLD_PREFILL_ROWS=${r%%:*}
        -e OCTOJET_LIVE_PREFILL_ROWS=${r##*:} -e OCTOJET_ROUND_LOG=/out/rounds-$r.jsonl)
  start "case-${r/:/-}"
  probe ttft http://127.0.0.1:8080 f1 --prompt "$T/warm.txt" --label warm --out "$OUT/warm-$r.jsonl"
  probe gap http://127.0.0.1:8080 f1 --prompt "$T/m128k-gap-$r.txt" --out "$OUT/gap-$r.json"
  probe ttft http://127.0.0.1:8080 f1 --prompt "$T/m128k-alone-$r.txt" --label "alone-$r" --out "$OUT/alone.jsonl"
  stop "case-${r/:/-}"
done
python3 - "$OUT" $CASES <<'PY' | tee "$OUT/f8-summary.txt" | tee -a "$TL"
import json, os, statistics, sys
out, rows = sys.argv[1], sys.argv[2:]
alone = {r["label"]: r for r in (json.loads(l) for l in open(os.path.join(out, "alone.jsonl")))} if os.path.exists(os.path.join(out, "alone.jsonl")) else {}
print("chunk rows:live rows | max gap | median gap | 128k TTFT with a live stream | 128k TTFT alone | chunk s (p50 / p90 / max, last quarter) | decode s p50")
for r in rows:
    g = json.load(open(os.path.join(out, f"gap-{r}.json"))) if os.path.exists(os.path.join(out, f"gap-{r}.json")) else {}
    rl = [json.loads(l) for l in open(os.path.join(out, f"rounds-{r}.jsonl"))] if os.path.exists(os.path.join(out, f"rounds-{r}.jsonl")) else []
    fill = [x for x in rl if x["filling"] and x["fill_s"] > 0 and x["filling"][0][0] > 100000]
    late = [x["fill_s"] for x in fill if x["filling"][0][1] >= 0.75 * x["filling"][0][0]]
    dec = [x["decode_s"] for x in rl if x["live"] > 0]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None
    print(f"{r} | {g.get('a_max_gap_s')} | {g.get('a_median_gap_s')} | {g.get('b_ttft_s')} | "
          f"{(alone.get(f'alone-{r}') or {}).get('ttft_s')} | {q(late, .5)} / {q(late, .9)} / {max(late) if late else None} | "
          f"{statistics.median(dec) if dec else None}")
PY
log "f8 measure done: rc $FAIL (summary $OUT/f8-summary.txt, rounds $OUT/rounds-*.jsonl)"
exit "$FAIL"
