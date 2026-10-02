#!/usr/bin/env bash
# F4 (prompts inside rounds, identical-prompt copies, F2d stage-B checkpoints): the operator's combined test window.
#   1) container tests (beside production): the F4 CUDA tests (lanes, twin copy, checkpoints) on the 4-layer NVFP4 cut
#      and the synthetic model, the F2d/F2c/F3 CUDA suites they touch, and the CPU tests of the new code.
#   Then production oj-serve is STOPPED (owner's standing approval; PROD_STOP) with the standby watcher paused for the
#   whole window; the pause is removed on every exit path and the operator's watcher restarts oj-serve.
#   2) a test server with production's flags + --vision + the default --prefix-checkpoints: its startup lines (estimate,
#      window, "N prefix checkpoints each") go to f4-startup.txt.
#   3) live (a): the largest token gap of a decoding stream while m128k is admitted (f4_live.py gap).
#   4) live (b): two identical m32k calls sent together: B's reuse / reuse_copy / cached, B's first token after A's.
#   5) live (c): P (cP.json, 71,444 tokens) cold, then P' (cPp.json, shares 69,013): reuse, cached, TTFT.
#   6) live (e): bench/vision_probe.py --quick (red square, isolation).
#   7) live (d): bench_openai --expect-equal on this server, then on a second server with --prefix-checkpoints 0 (same
#      build, same flags otherwise); every token_sha must be equal (f4-text-exactness.json).
# Usage: bash bench/spark/f4-test.sh [--dry-run] [--steps 1,2,3,4,5,6,7]   (2 is implied by 3-7; about 35 min in all)
#   --dry-run checks inputs and renders every request; starts nothing and stops nothing. Knobs: PROD_STOP (default
#   "docker stop -t 60 oj-serve"), M, DEPS, IMAGE, TENSORFOLD_PREFILL_ROWS, CHECKPOINTS (passes --prefix-checkpoints N
#   to the first server; default: the engine's), SHARED_GPU=1.
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f4}; OUT=${OUT:-$HOME/octojet-runs/f4}; NAME=${NAME:-oj-f4}; PARALLEL=${PARALLEL:-3}
PREFIX=f4; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
DEPS=${DEPS:-$HOME/tensorfold/octojet-pydeps}
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
MOUNTS+=(-v "$DEPS:/deps:ro")
ENVS=(-e PYTHONPATH=/octojet/engine/src:/deps -e PYTHONDONTWRITEBYTECODE=1)
for v in TENSORFOLD_PREFILL_ROWS TENSORFOLD_VISION_WORKSPACE_MIB; do [ -z "${!v:-}" ] || ENVS+=(-e "$v=${!v}"); done
BASE_SERVE=("${SERVE[@]}" --vision)
MAIN_SERVE=("${BASE_SERVE[@]}"); [ -z "${CHECKPOINTS:-}" ] || MAIN_SERVE+=(--prefix-checkpoints "$CHECKPOINTS")
OFF_SERVE=("${BASE_SERVE[@]}" --prefix-checkpoints 0)
STEPS=1,2,3,4,5,6,7; DRYRUN=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRYRUN=1; shift;;
  --steps) STEPS=${2:-}; shift 2;;
  *) echo "usage: $0 [--dry-run] [--steps 1,2,3,4,5,6,7]" >&2; exit 2;; esac; done
[[ "$STEPS" =~ ^[1-7](,[1-7])*$ ]] || { echo "error: --steps takes a comma list of 1-7 (got '$STEPS')" >&2; exit 2; }
has(){ [[ ",$STEPS," == *",$1,"* ]]; }
cd "$REPO"
need=(); has 3 && need+=(m128k); has 4 && need+=(m32k); has 5 && need+=(cP cPp)
for m in ${need[@]+"${need[@]}"}; do [ -f "$M/$m.json" ] || { echo "error: $M/$m.json missing" >&2; exit 2; }; done
[ -d "$DEPS" ] || { echo "error: $DEPS missing (transformers and av for the container)" >&2; exit 2; }

live(){ local how=(--out "$3"); [ -z "$DRY" ] || how=(--dry-run)
  python3 bench/f4_live.py http://127.0.0.1:8080 f1 "$1" --manifest "$2" "${how[@]}" 2>&1 | tee -a "$TL" \
    || { log "live $1: FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }; }
openai(){ [ -n "$DRY" ] && return 0
  python3 engine/tools/bench_openai.py http://127.0.0.1:8080 f1 --expect-equal --label "$1" --output "$OUT/f4-bench-openai-$1.json" | tee -a "$TL" \
    || { log "$1: bench_openai --expect-equal FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }; }
probe(){ local dry=(); [ -n "$DRY" ] && dry=(--dry-run) || dry=(--out /out/f4-probe)
  docker run --rm --network host -v "$REPO:/octojet:ro" -v "$OUT:/out" -v "$DEPS:/deps:ro" -e PYTHONPATH=/deps \
    -e PYTHONDONTWRITEBYTECODE=1 -w /octojet --entrypoint python "$IMAGE" \
    bench/vision_probe.py http://127.0.0.1:8080 f1 --media /out --quick "${dry[@]}" 2>&1 | tee -a "$TL" \
    || { log "vision probe: an expectation failed or a request errored (rc ${PIPESTATUS[0]}; see $OUT/f4-probe.md)"; FAIL=1; }; }
window(){
  if has 2 || has 3 || has 4 || has 5 || has 6 || has 7; then
    SERVE=("${MAIN_SERVE[@]}"); start main
    [ -n "$DRY" ] || grep -h "startup estimate\|loaded in\|streams of\|vision:" "$OUT/f4-main-server.log" > "$OUT/f4-startup.txt" || true
    if has 3; then live gap "$M/m128k.json" "$OUT/f4-gap.json"; fi
    if has 4; then live twins "$M/m32k.json" "$OUT/f4-twins.json"; fi
    if has 5; then
      bench --manifest "$M/cP.json" --arm clean:1 --draft on --max-tokens 64 --expect-reuse none --label P-cold \
        --server-label main --out "$OUT/f4-checkpoint.jsonl" | tee -a "$TL" || { log "checkpoint: P cold FAILED"; FAIL=1; }
      bench --manifest "$M/cPp.json" --arm clean:1 --draft on --max-tokens 64 --expect-reuse checkpoint --expect-cached any \
        --label Pp-checkpoint --server-label main --out "$OUT/f4-checkpoint.jsonl" | tee -a "$TL" \
        || { log "checkpoint: P' did not resume from a checkpoint"; FAIL=1; }
    fi
    if has 6; then probe; fi
    if has 7; then openai main; fi
    stop main
  fi
  if has 7; then
    SERVE=("${OFF_SERVE[@]}"); start off
    [ -n "$DRY" ] || grep -h "startup estimate\|streams of" "$OUT/f4-off-server.log" >> "$OUT/f4-startup.txt" || true
    openai off
    stop off
  fi; }

DRY=--dry-run; window; DRY=; log "dry-run OK: inputs present, every planned request renders"
[ "$DRYRUN" = 1 ] && exit 0
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || cp "$REPO/COMMIT" "$OUT/commit.txt" 2>/dev/null || true

if has 1; then
  SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
  CPU="tests/test_cuda_prompts_inside_rounds.py tests/test_prefix_checkpoints.py tests/test_prefix_reuse.py tests/test_prefix_reuse_multi.py tests/test_cuda_stream_slots.py tests/test_prefill_timing_cuts.py"
  log "tests: container tests (OCTOJET_NVFP4_FLASHNEXT_LAYERS=4), beside production"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" -v "$SCRATCH:/scratch" "${ENVS[@]}" -e TMPDIR=/scratch -e OCTOJET_CACHE_DIR=/scratch/default \
    -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 -w /octojet/engine --entrypoint bash "$IMAGE" -c "
      set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
      python -m pytest $CPU -q -p no:cacheprovider || rc=1
      python -m pytest tests/cuda/test_flashnext_multi.py tests/cuda/test_flashnext_vision.py tests/cuda/test_flashnext_forward.py -q -p no:cacheprovider --basetemp=/scratch/pytest1 || rc=1
      python -m pytest tests/cuda/test_qwen4_exp_nvfp4.py -q -p no:cacheprovider --basetemp=/scratch/pytest2 || rc=1
      python -m pytest tests/cuda/test_prefix_reuse_flashnext.py tests/cuda/test_prefill_timing_flashnext.py -q -p no:cacheprovider --basetemp=/scratch/pytest3 || rc=1
      echo \"F4 TESTS rc=\$rc\"
      exit \$rc" 2>&1 | tee "$OUT/f4-tests.txt" || { log "tests: container tests FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
  docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true' || true
  rmdir "$SCRATCH" 2>/dev/null || true
  log "tests: done (FAIL=$FAIL so far)"
fi

if has 2 || has 3 || has 4 || has 5 || has 6 || has 7; then
  pause_on
  log "WINDOW: stopping production oj-serve ($PROD_STOP); the standby watcher restarts it when the pause is removed"
  bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
  port_free || { log "WINDOW: port 8080 still listening after stopping production"; exit 1; }
  window
  python3 - "$OUT" "$M" <<'PY' | tee "$OUT/f4-summary.txt" | tee -a "$TL" || FAIL=1
import json, os, sys
out, m = sys.argv[1:3]
ok = True
def load(name):
    p = os.path.join(out, name)
    return json.load(open(p)) if os.path.exists(p) else None
def rows(name):
    p = os.path.join(out, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
g = load("f4-gap.json")
if g:
    print(f"(a) lanes: decoding stream's largest gap while m128k ({g.get('b_prompt_tokens')} tokens) was admitted: "
          f"{g.get('a_max_gap_s')} s ({g.get('a_tokens_during_b')} tokens during its {g.get('b_ttft_s')} s TTFT); "
          f"before F4 it was the whole prefill (~68 s)")
    ok &= bool(g.get("a_max_gap_s") is not None and g["a_max_gap_s"] < 10 and not g.get("error"))
t = load("f4-twins.json")
if t:
    b = t.get("b") or {}
    print(f"(b) twins: B reuse {t.get('b_reuse')}, reuse_copy {t.get('b_reuse_copy')}, cached {t.get('b_cached')} of "
          f"{t.get('prompt_tokens')}; B's first token {t.get('b_after_a_s')} s after A's (gate < 1 s); B TTFT {b.get('ttft_s')} s")
    ok &= bool(t.get("b_reuse") == "exact" and t.get("b_cached") == t.get("prompt_tokens")
               and t.get("b_after_a_s") is not None and t["b_after_a_s"] < 1.0)
r = rows("f4-checkpoint.jsonl")
if r:
    def ids(n):
        return json.load(open(os.path.join(m, n)))["ids"]
    p, pp = ids("cP.json"), ids("cPp.json")
    d = next((i for i, (x, y) in enumerate(zip(p, pp)) if x != y), min(len(p), len(pp)))
    rows_ = int(os.environ.get("TENSORFOLD_PREFILL_ROWS") or 2048)
    for row in r:
        print(f"(c) {row.get('label')}: {row.get('prompt_tokens')} tokens, reuse {row.get('reuse')}, cached "
              f"{row.get('cached')}, TTFT {row.get('ttft_s')} s")
    last = r[-1]
    print(f"(c) P' shares {d} tokens with P; the last {rows_}-row chunk end at or before it is {d // rows_ * rows_} "
          f"(the cached the default N = 4 placement should give when that end is a checkpoint)")
    ok &= last.get("reuse") == "checkpoint"
print("F4 live checks: " + ("PASS" if ok else "CHECK"))
sys.exit(0 if ok else 1)
PY
  if has 7; then
    python3 - "$OUT/f4-bench-openai-main.json" "$OUT/f4-bench-openai-off.json" "$OUT/f4-text-exactness.json" <<'PY' | tee -a "$TL" || FAIL=1
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
key = lambda r: (r["prompt"], r["temperature"])
va, vb = {key(r): r["token_sha_all"] for r in a}, {key(r): r["token_sha_all"] for r in b}
rows = [{"prompt": k[0], "temperature": k[1], "equal": va.get(k) == vb.get(k)} for k in sorted(set(va) | set(vb))]
ok = bool(rows) and all(r["equal"] for r in rows)
json.dump({"checkpoints_vs_off_equal": ok, "rows": rows}, open(sys.argv[3], "w"), indent=1)
print(f"(d) text exactness, default --prefix-checkpoints vs 0: {'IDENTICAL' if ok else 'DIFFERENT'} "
      f"({sum(r['equal'] for r in rows)}/{len(rows)} prompt x temperature)")
sys.exit(0 if ok else 1)
PY
  fi
fi
log "F4 window done: rc $FAIL (tests $OUT/f4-tests.txt, startup $OUT/f4-startup.txt, summary $OUT/f4-summary.txt, exactness $OUT/f4-text-exactness.json, probe $OUT/f4-probe.md)"
exit "$FAIL"
