#!/usr/bin/env bash
# F3 (Flash Next image/video input, tiled QSA select, TENSORFOLD_PREFILL_ROWS) — the operator's combined test window.
#   1) container tests (beside production): the new CUDA tests (vision, tiled select, prompt chunk rows), the F2d/F2c
#      CUDA suites, and the host tests that need Pillow/transformers/PyAV; on the 4-layer NVFP4 cut.
#   Then production oj-serve is STOPPED (owner's standing approval; PROD_STOP) with the standby watcher paused for the
#   whole window; the pause is removed on every exit path and the operator's watcher restarts oj-serve.
#   2) a test server with production's flags plus --vision: its startup lines (load time, streams, cache slots, the
#      tower and workspace it reserves) go to the timeline and f3-startup.txt.
#   3) text exactness: bench_openai --expect-equal on the vision server and on a second server WITHOUT --vision (same
#      build); every token_sha must be equal (f3-text-exactness.json).
#   4) prefill timing on both servers: m32k, m128k, m210k, two cold repetitions each (the manifest, then a copy whose
#      last token differs: an identical second call would be an F2d exact hit and measure nothing).
#      F2c baselines (--parallel 2, 2,048-row chunks): 15.22 s, 68.53 s, 227.01 s.
#   5) bench/vision_probe.py on the vision server (red square, robots, towel acceptance, video, isolation, an image
#      request while text decodes): replies verbatim in f3-probe.md / .jsonl.
# Usage: bash bench/spark/f3-test.sh [--dry-run] [--steps 1,2,3,4,5]
#   --dry-run checks inputs, renders every request (prefill_bench --dry-run, vision_probe --dry-run in the image), starts
#   nothing and stops nothing. Knobs: PROD_STOP (default "docker stop -t 60 oj-serve"), MEDIA, DEPS, M, IMAGE,
#   TENSORFOLD_PREFILL_ROWS / TENSORFOLD_VISION_WORKSPACE_MIB (passed into the servers when set), SHARED_GPU=1.
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f3}; OUT=${OUT:-$HOME/octojet-runs/f3}; NAME=${NAME:-oj-f3}; PARALLEL=${PARALLEL:-3}
PREFIX=f3; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
MEDIA=${MEDIA:-$HOME/octojet-test-media}; DEPS=${DEPS:-$HOME/tensorfold/octojet-pydeps}
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
MOUNTS+=(-v "$DEPS:/deps:ro")
ENVS=(-e PYTHONPATH=/octojet/engine/src:/deps -e PYTHONDONTWRITEBYTECODE=1)
for v in TENSORFOLD_PREFILL_ROWS TENSORFOLD_VISION_WORKSPACE_MIB TENSORFOLD_MAX_IMAGES TENSORFOLD_IMAGE_TOKENS TENSORFOLD_VIDEO_TOKENS; do
  [ -z "${!v:-}" ] || ENVS+=(-e "$v=${!v}"); done
TEXT_SERVE=("${SERVE[@]}"); VISION_SERVE=("${SERVE[@]}" --vision)
STEPS=1,2,3,4,5; DRYRUN=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRYRUN=1; shift;;
  --steps) STEPS=${2:-}; shift 2;;
  *) echo "usage: $0 [--dry-run] [--steps 1,2,3,4,5]" >&2; exit 2;; esac; done
[[ "$STEPS" =~ ^[1-5](,[1-5])*$ ]] || { echo "error: --steps takes a comma list of 1-5 (got '$STEPS')" >&2; exit 2; }
has(){ [[ ",$STEPS," == *",$1,"* ]]; }
cd "$REPO"
SIZES=(32k 128k 210k)
if has 4; then for s in "${SIZES[@]}"; do [ -f "$M/m$s.json" ] || { echo "error: $M/m$s.json missing" >&2; exit 2; }; done; fi
if has 5; then for f in robots-sim-observation.png towel-crumpled.jpg towel-flat.jpg towel-crumpled-1280.jpg towel-flat-1280.jpg; do
  [ -f "$MEDIA/$f" ] || { echo "error: $MEDIA/$f missing" >&2; exit 2; }; done; fi
[ -d "$DEPS" ] || { echo "error: $DEPS missing (transformers 5.17.0 and av 19.0.0 for the container)" >&2; exit 2; }

variants(){ local s; for s in "${SIZES[@]}"; do      # the same length, last token changed: cold again after the manifest
  python3 - "$M/m$s.json" "$OUT/m$s-b.json" <<'PY'
import hashlib, json, sys
m = json.load(open(sys.argv[1])); ids = list(m["ids"])
ids[-1] = next(t for t in reversed(ids[:-1]) if t != ids[-1])
m.update(ids=ids, sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), derived_from=m.get("sha256"))
m.pop("prompt_sha", None); json.dump(m, open(sys.argv[2], "w"))
PY
  done; }
prefill(){ local label=$1 s; for s in "${SIZES[@]}"; do
  bench --manifest "$M/m$s.json" --arm clean:1 --draft on --label "$label-$s-a" --server-label "$label" --out "$OUT/f3-prefill-$label.jsonl" | tee -a "$TL" \
    || { log "$label: prefill $s (a) reported an error or a reuse (rc ${PIPESTATUS[0]})"; FAIL=1; }
  bench --manifest "$OUT/m$s-b.json" --arm clean:1 --draft on --label "$label-$s-b" --server-label "$label" --out "$OUT/f3-prefill-$label.jsonl" | tee -a "$TL" \
    || { log "$label: prefill $s (b) reported an error or a reuse (rc ${PIPESTATUS[0]})"; FAIL=1; }
  done; }
openai(){ [ -n "$DRY" ] && return 0
  python3 engine/tools/bench_openai.py http://127.0.0.1:8080 f1 --expect-equal --label "$1" --output "$OUT/f3-bench-openai-$1.json" | tee -a "$TL" \
    || { log "$1: bench_openai --expect-equal FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }; }
probe(){ local dry=(); [ -n "$DRY" ] && dry=(--dry-run) || dry=(--out /out/f3-probe)
  docker run --rm --network host -v "$REPO:/octojet:ro" -v "$MEDIA:/media:ro" -v "$OUT:/out" -v "$DEPS:/deps:ro" \
    -e PYTHONPATH=/deps -e PYTHONDONTWRITEBYTECODE=1 -w /octojet --entrypoint python "$IMAGE" \
    bench/vision_probe.py http://127.0.0.1:8080 f1 --media /media "${dry[@]}" 2>&1 | tee -a "$TL" \
    || { log "vision probe: an expectation failed or a request errored (rc ${PIPESTATUS[0]}; see $OUT/f3-probe.md)"; FAIL=1; }; }
window(){                                       # 2-5: the vision server, then the text server
  if has 2 || has 3 || has 4 || has 5; then
    SERVE=("${VISION_SERVE[@]}"); start vision
    [ -n "$DRY" ] || grep -h "startup estimate\|loaded in\|streams of\|vision:" "$OUT/f3-vision-server.log" > "$OUT/f3-startup.txt" || true
    if has 3; then openai vision; fi
    if has 4; then prefill vision; fi
    if has 5; then probe; fi
    stop vision
  fi
  if has 3 || has 4; then
    SERVE=("${TEXT_SERVE[@]}"); start text
    [ -n "$DRY" ] || grep -h "startup estimate\|loaded in\|streams of" "$OUT/f3-text-server.log" >> "$OUT/f3-startup.txt" || true
    if has 3; then openai text; fi
    if has 4; then prefill text; fi
    stop text
  fi; }

if has 4; then variants; fi
DRY=--dry-run; window; DRY=; log "dry-run OK: inputs present, every planned request renders"
[ "$DRYRUN" = 1 ] && exit 0
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || cp "$REPO/COMMIT" "$OUT/commit.txt" 2>/dev/null || true

if has 1; then
  SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
  CPU="tests/test_vision_images.py tests/test_vision_server.py tests/test_vision_cuda.py tests/test_vision_flashnext.py tests/test_vision_rotary.py tests/test_flash_capacity_dispatch.py tests/test_prefill_timing_cuts.py /octojet/bench/test_vision_probe.py"
  log "tests: container tests (OCTOJET_NVFP4_FLASHNEXT_LAYERS=4), beside production"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" -v "$SCRATCH:/scratch" "${ENVS[@]}" -e TMPDIR=/scratch -e OCTOJET_CACHE_DIR=/scratch/default \
    -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 -w /octojet/engine --entrypoint bash "$IMAGE" -c "
      set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
      python -m pytest $CPU -q -p no:cacheprovider || rc=1
      python -m pytest tests/cuda/test_flashnext_vision.py -q -p no:cacheprovider --basetemp=/scratch/pytest1 || rc=1
      python -m pytest tests/cuda/test_flashnext_kernels.py -k 'register_width or qsa_selection' -q -p no:cacheprovider --basetemp=/scratch/pytest2 || rc=1
      python -m pytest tests/cuda/test_prefix_reuse_flashnext.py -k 'not chunk_sizes' tests/cuda/test_flashnext_multi.py tests/cuda/test_prefill_timing_flashnext.py -q -p no:cacheprovider --basetemp=/scratch/pytest3 || rc=1
      echo \"F3 TESTS rc=\$rc\"
      if python -m pytest tests/cuda/test_prefix_reuse_flashnext.py -k chunk_sizes -q -p no:cacheprovider --basetemp=/scratch/pytest4; then echo 'CHUNK-SIZE TEST (informational) rc=0'; else echo 'CHUNK-SIZE TEST (informational) rc=1'; fi
      exit \$rc" 2>&1 | tee "$OUT/f3-tests.txt" || { log "tests: container tests FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
  docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true' || true
  rmdir "$SCRATCH" 2>/dev/null || true
  log "tests: done (FAIL=$FAIL so far)"
fi

if has 2 || has 3 || has 4 || has 5; then
  pause_on
  log "WINDOW: stopping production oj-serve ($PROD_STOP); the standby watcher restarts it when the pause is removed"
  bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
  port_free || { log "WINDOW: port 8080 still listening after stopping production"; exit 1; }
  window
  if has 3; then
    python3 - "$OUT/f3-bench-openai-vision.json" "$OUT/f3-bench-openai-text.json" "$OUT/f3-text-exactness.json" <<'PY' | tee -a "$TL" || FAIL=1
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
key = lambda r: (r["prompt"], r["temperature"])
va, vb = {key(r): r["token_sha_all"] for r in a}, {key(r): r["token_sha_all"] for r in b}
rows = [{"prompt": k[0], "temperature": k[1], "equal": va.get(k) == vb.get(k)} for k in sorted(set(va) | set(vb))]
ok = bool(rows) and all(r["equal"] for r in rows)
json.dump({"vision_vs_text_equal": ok, "rows": rows}, open(sys.argv[3], "w"), indent=1)
print(f"text exactness, --vision vs without: {'IDENTICAL' if ok else 'DIFFERENT'} ({sum(r['equal'] for r in rows)}/{len(rows)} prompt x temperature)")
sys.exit(0 if ok else 1)
PY
  fi
  if has 4; then
    python3 - "$OUT/f3-prefill-vision.jsonl" "$OUT/f3-prefill-text.jsonl" <<'PY' | tee "$OUT/f3-prefill-summary.txt" | tee -a "$TL" || true
import json, sys
base = {"32k": 15.22, "128k": 68.53, "210k": 227.01}
for path in sys.argv[1:]:
    try:
        rows = [json.loads(l) for l in open(path)]
    except OSError:
        continue
    for size, ref in base.items():
        t = [r["ttft_s"] for r in rows if str(r.get("label", "")).split("-")[1:2] == [size] and r.get("ttft_s")]
        if t:
            print(f"{rows[0].get('server_label')}: {size} cold TTFT {', '.join(f'{x:.2f}' for x in t)} s (F2c baseline {ref:.2f} s at --parallel 2)")
PY
  fi
fi
log "F3 window done: rc $FAIL (tests $OUT/f3-tests.txt, startup $OUT/f3-startup.txt, exactness $OUT/f3-text-exactness.json, prefill $OUT/f3-prefill-summary.txt, probe $OUT/f3-probe.md)"
exit "$FAIL"
