#!/usr/bin/env bash
# F6 / A3: build and verify the self-contained public checkpoint (spec 2026-09-29-f2-release-design.md section 4).
#   1) build (beside production, no GPU): disk check (NEED_GB, default 120 GB free beside OUT), then
#      tools/build_release_checkpoint.py --mlx MLX --nvfp4 NVFP4 --out OUT --hash in the $IMAGE container, run as the
#      operator's uid so the output is the operator's. The build report lands beside OUT (OUT.build-report.json).
#   2) check (beside production, CPU): check_mixed_checkpoint.py OUT --reference-mlx MLX --reference-nvfp4 NVFP4
#      --layers all --source-bytes -> f6-check.json (routers, 48/48 routed and shared layers agree, every tensor's bytes
#      equal its source's, dropped names absent).
#   3) cut-model tests (beside production, GPU): tests/cuda/test_qwen4_exp_nvfp4.py and test_prefix_reuse_flashnext.py
#      with OCTOJET_NVFP4_FLASHNEXT=OUT (4 layers), plus the CPU tests of the release code.
#   Then production oj-serve is STOPPED (owner's standing approval; PROD_STOP) with the standby watcher paused for the
#   window; every exit path removes the pause and the operator's watcher restarts oj-serve.
#   4) window: OUT served with production's flags, text only (--kv-dtype int8 --parallel 3, its own --packed-cache
#      PACKED so production's cache is untouched): startup estimate and window (production: 3 x 261,872 at ~88.35 GiB
#      with --vision; text only expect ~1 GiB less), bench_openai --expect-equal (drafted == serial), concurrent_equal
#      --prompts 2, acc_eval (GSM8K 250) + run_humaneval.sh (HumanEval 164). Gates: exactness equal; GSM8K >= 245-3 and
#      HumanEval >= 155-4 (production mixed dir; the release's shared experts come from HF main bf16, not fp8hybrid);
#      window >= 261,872 a stream; estimate within 1.5 GiB of 87.35. -> f6-summary.txt
# Usage: bash bench/spark/release-build.sh [--dry-run] [--steps 1,2,3,4]
#   --dry-run validates inputs (sources, disk, steps) in the container; builds, stops and starts nothing.
# Knobs (host paths): OUT (the checkpoint, default $HOME/tensorfold/octojet-release), MLX (default
#   $HOME/tensorfold/flashnext-mlx-4bit-mtp, i.e. /tf/flashnext-mlx-4bit-mtp), NVFP4 (default $HOME/tensorfold/up-radix,
#   RadixArk HF main), MLX_REVISION (default "unknown"), NVFP4_REVISION (default 7b719225242aacd3dbd3f9407468c2ee9a9d2594),
#   RUNS (logs, default $HOME/octojet-runs/f6), PACKED (default $HOME/tensorfold/octojet-cache/packed-release; "off"
#   packs at every start), CACHE_GB (free space the packed cache needs, default 75), NEED_GB, PROD_STOP, IMAGE, REPO,
#   SHARED_GPU=1, GSM_BASE/HE_BASE (245/155), GSM_TOL/HE_TOL (3/4).
set -euo pipefail
RELEASE=${OUT:-$HOME/tensorfold/octojet-release}
OUT=${RUNS:-$HOME/octojet-runs/f6}            # f2d-a-lib.sh's OUT is the run directory (logs, receipts)
REPO=${REPO:-$HOME/octojet-f6}; NAME=${NAME:-oj-f6}; PARALLEL=${PARALLEL:-3}
PREFIX=f6; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
MLX=${MLX:-$HOME/tensorfold/flashnext-mlx-4bit-mtp}; NVFP4=${NVFP4:-$HOME/tensorfold/up-radix}
MLX_REVISION=${MLX_REVISION:-unknown}; NVFP4_REVISION=${NVFP4_REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
PACKED=${PACKED:-$HOME/tensorfold/octojet-cache/packed-release}; CACHE_GB=${CACHE_GB:-75}; NEED_GB=${NEED_GB:-120}
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
GSM_BASE=${GSM_BASE:-245}; HE_BASE=${HE_BASE:-155}; GSM_TOL=${GSM_TOL:-3}; HE_TOL=${HE_TOL:-4}
STEPS=1,2,3,4; DRYRUN=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRYRUN=1; shift;;
  --steps) STEPS=${2:-}; shift 2;;
  *) echo "usage: $0 [--dry-run] [--steps 1,2,3,4]" >&2; exit 2;; esac; done
[[ "$STEPS" =~ ^[1-4](,[1-4])*$ ]] || { echo "error: --steps takes a comma list of 1-4 (got '$STEPS')" >&2; exit 2; }
has(){ [[ ",$STEPS," == *",$1,"* ]]; }
cd "$REPO"

# host path -> container path: under $HOME/tensorfold it is /tf/...; elsewhere the same path, mounted as is
cpath(){ case "$1" in "$HOME/tensorfold") echo /tf;; "$HOME/tensorfold"/*) echo "/tf/${1#"$HOME/tensorfold/"}";; *) echo "$1";; esac; }
extra_mount(){ case "$1" in "$HOME/tensorfold"|"$HOME/tensorfold"/*) ;; *) MOUNTS+=(-v "$1:$1$2");; esac; }
RELEASE_PARENT=$(dirname "$RELEASE")
extra_mount "$MLX" :ro; extra_mount "$NVFP4" :ro; extra_mount "$RELEASE_PARENT" ""
[ "$PACKED" = off ] || extra_mount "$PACKED" ""
C_MLX=$(cpath "$MLX"); C_NVFP4=$(cpath "$NVFP4"); C_RELEASE=$(cpath "$RELEASE")
C_PACKED=off; [ "$PACKED" = off ] || C_PACKED=$(cpath "$PACKED")

# ---- inputs ------------------------------------------------------------------------------------------------------------
for d in "$MLX" "$NVFP4"; do
  [ -f "$d/model.safetensors.index.json" ] || { echo "error: $d/model.safetensors.index.json missing" >&2; exit 2; }; done
[ -f "$MLX/config.json" ] || { echo "error: $MLX/config.json missing" >&2; exit 2; }
[[ "$NVFP4_REVISION" =~ ^([0-9a-f]{40}|unknown)$ ]] || { echo "error: NVFP4_REVISION must be a 40-hex commit or 'unknown'" >&2; exit 2; }
[[ "$MLX_REVISION" =~ ^([0-9a-f]{40}|unknown)$ ]] || { echo "error: MLX_REVISION must be a 40-hex commit or 'unknown'" >&2; exit 2; }
free_gb(){ local p=$1; while [ ! -d "$p" ]; do p=$(dirname "$p"); done; df -Pk "$p" | awk 'NR==2 {print int($4 / 1000000)}'; }
if has 1; then
  if [ -e "$RELEASE" ] && [ -n "$(ls -A "$RELEASE" 2>/dev/null)" ]; then
    echo "error: $RELEASE exists and is not empty (remove it, set OUT, or skip step 1 to verify an existing build)" >&2; exit 2; fi
  g=$(free_gb "$RELEASE_PARENT")
  [ "$g" -ge "$NEED_GB" ] || { echo "error: $g GB free beside $RELEASE, the build needs ~$NEED_GB GB" >&2; exit 2; }
  echo "disk: $g GB free beside $RELEASE (need ~$NEED_GB GB)"
elif has 2 || has 3 || has 4; then
  [ -f "$RELEASE/octojet.json" ] || { echo "error: $RELEASE/octojet.json missing (build it with step 1)" >&2; exit 2; }
fi
if has 4 && [ "$PACKED" != off ] && [ -z "$(ls -A "$PACKED" 2>/dev/null)" ]; then
  g=$(free_gb "$PACKED"); need=$CACHE_GB; has 1 && need=$((CACHE_GB + NEED_GB))
  [ "$g" -ge "$need" ] || { echo "error: $g GB free for the packed cache $PACKED; need ~$need GB (or PACKED=off)" >&2; exit 2; }
fi
# the sources as the container sees them: every shard each index names is readable there (links into /srv/ai included)
preflight(){ docker run --rm "${MOUNTS[@]}" "${ENVS[@]}" -w /octojet/engine --entrypoint python "$IMAGE" - "$C_MLX" "$C_NVFP4" <<'PY'
import json, os, sys
for root in sys.argv[1:]:
    shards = sorted(set(json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"].values()))
    bad = [s for s in shards if not os.access(os.path.join(root, s), os.R_OK)]
    print(f"{root}: {len(shards)} shards" + (f", UNREADABLE {bad[:5]}" if bad else ""))
    if bad:
        sys.exit(1)
PY
}
log "inputs: MLX $MLX ($C_MLX, revision $MLX_REVISION), NVFP4 $NVFP4 ($C_NVFP4, revision $NVFP4_REVISION), OUT $RELEASE ($C_RELEASE), steps $STEPS"
if has 1 || has 2; then preflight 2>&1 | tee -a "$TL" || { log "inputs: a source shard is unreadable in the container"; exit 2; }; fi
SERVE=(-m tensorfold.cli serve "$C_RELEASE" --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel "$PARALLEL"
       --packed-cache "$C_PACKED")
if [ "$DRYRUN" = 1 ]; then log "dry-run OK: inputs valid; serve args: ${SERVE[*]}"; exit 0; fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

# ---- 1) build ----------------------------------------------------------------------------------------------------------
if has 1; then
  log "build: $RELEASE (beside production)"; t0=$SECONDS
  docker run --rm --user "$(id -u):$(id -g)" "${MOUNTS[@]}" "${ENVS[@]}" -e HOME=/tmp -w /octojet/engine --entrypoint python "$IMAGE" \
    tools/build_release_checkpoint.py --mlx "$C_MLX" --nvfp4 "$C_NVFP4" --out "$C_RELEASE" --hash \
    --mlx-revision "$MLX_REVISION" --nvfp4-revision "$NVFP4_REVISION" 2>&1 | tee "$OUT/f6-build.txt" \
    || { log "build: FAILED (rc ${PIPESTATUS[0]}; see $OUT/f6-build.txt)"; exit 1; }
  cp "$RELEASE.build-report.json" "$OUT/" 2>/dev/null || true
  log "build: done in $((SECONDS - t0)) s; $(du -sh "$RELEASE" | cut -f1) in $RELEASE"
fi

# ---- 2) check ----------------------------------------------------------------------------------------------------------
if has 2; then
  log "check: check_mixed_checkpoint --layers all --source-bytes (beside production)"; t0=$SECONDS
  docker run --rm "${MOUNTS[@]}" "${ENVS[@]}" -w /octojet/engine --entrypoint python "$IMAGE" \
    tools/check_mixed_checkpoint.py "$C_RELEASE" --reference-mlx "$C_MLX" --reference-nvfp4 "$C_NVFP4" \
    --layers all --source-bytes > "$OUT/f6-check.json" 2> "$OUT/f6-check.err" \
    && log "check: ok in $((SECONDS - t0)) s" || { log "check: FAILED (rc $?; $OUT/f6-check.json, $OUT/f6-check.err)"; FAIL=1; }
fi

# ---- 3) cut-model tests ------------------------------------------------------------------------------------------------
if has 3; then
  SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
  CPU="tests/test_release_checkpoint.py tests/test_check_mixed_checkpoint.py tests/test_hub_mixed.py tests/test_nvfp4_loader.py"
  log "tests: 4-layer cut of $RELEASE (OCTOJET_NVFP4_FLASHNEXT_LAYERS=4), beside production"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" -v "$SCRATCH:/scratch" "${ENVS[@]}" -e TMPDIR=/scratch -e OCTOJET_CACHE_DIR=/scratch/default \
    -e OCTOJET_NVFP4_FLASHNEXT="$C_RELEASE" -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 -w /octojet/engine --entrypoint bash "$IMAGE" -c "
      set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
      python -m pytest $CPU -q -p no:cacheprovider || rc=1
      python -m pytest tests/cuda/test_qwen4_exp_nvfp4.py -q -p no:cacheprovider --basetemp=/scratch/pytest1 || rc=1
      python -m pytest tests/cuda/test_prefix_reuse_flashnext.py -q -p no:cacheprovider --basetemp=/scratch/pytest2 || rc=1
      echo \"F6 TESTS rc=\$rc\"
      exit \$rc" 2>&1 | tee "$OUT/f6-tests.txt" || { log "tests: FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
  docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true' || true
  rmdir "$SCRATCH" 2>/dev/null || true
  log "tests: done (FAIL=$FAIL so far)"
fi

# ---- 4) window ---------------------------------------------------------------------------------------------------------
if has 4; then
  [ "$PACKED" = off ] || mkdir -p "$PACKED"
  pause_on
  log "WINDOW: stopping production oj-serve ($PROD_STOP); the standby watcher restarts it when the pause is removed"
  bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
  port_free || { log "WINDOW: port 8080 still listening after stopping production"; exit 1; }
  start release
  grep -h "startup estimate\|loaded in\|streams of\|packed" "$OUT/f6-release-server.log" > "$OUT/f6-startup.txt" || true
  log "exactness: bench_openai --expect-equal"
  python3 engine/tools/bench_openai.py http://127.0.0.1:8080 f1 --expect-equal --label release --output "$OUT/f6-bench-openai.json" | tee -a "$TL" \
    || { log "exactness: bench_openai --expect-equal FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; echo fail > "$OUT/f6-bench-openai.failed"; }
  log "exactness: concurrent_equal --prompts 2"
  python3 bench/concurrent_equal.py http://127.0.0.1:8080 f1 --prompts 2 --out "$OUT/f6-concurrent-equal.json" | tee -a "$TL" \
    || { log "exactness: concurrent_equal FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; echo fail > "$OUT/f6-concurrent-equal.failed"; }
  log "accuracy: GSM8K 250 + HumanEval 164 (greedy, thinking off)"
  python3 bench/acc_eval.py http://127.0.0.1:8080 f1 "$OUT/f6-acc" > "$OUT/f6-acc-summary.json" 2>> "$TL" \
    || { log "accuracy: acc_eval FAILED"; FAIL=1; }
  bash bench/run_humaneval.sh "$OUT/f6-acc.jsonl" > /dev/null 2>> "$TL" || { log "accuracy: run_humaneval FAILED"; FAIL=1; }
  stop release
  pause_off
fi

# ---- summary -----------------------------------------------------------------------------------------------------------
python3 - "$OUT" "$RELEASE" "$GSM_BASE" "$GSM_TOL" "$HE_BASE" "$HE_TOL" "$STEPS" <<'PY' | tee "$OUT/f6-summary.txt" | tee -a "$TL" || FAIL=1
import json, os, re, sys
out, release = sys.argv[1:3]
gsm_base, gsm_tol, he_base, he_tol = (int(x) for x in sys.argv[3:7])
steps = sys.argv[7].split(",")
ok = True
def load(name):
    p = os.path.join(out, name)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None
print(f"F6 release checkpoint: {release}")
b = load(os.path.basename(release) + ".build-report.json")
if b:
    e = b["experts"]
    print(f"(1) build: total {b['total_size'] / 1e9:.1f} GB ({b['total_size'] / 2**30:.1f} GiB) from the indices, "
          f"{b['out_file_bytes'] / 1e9:.1f} GB on disk; base {b['base']['total_size'] / 2**30:.2f} GiB in "
          f"{len(b['base']['shards'])} shards ({b['base']['dropped_tensors']} tensors dropped); experts "
          f"{e['total_size'] / 2**30:.2f} GiB, {e['copied_whole']} shards copied whole, {e['rewritten']} rewritten, "
          f"{e['wasted_bytes'] / 2**20:.1f} MiB unneeded kept")
elif "1" in steps:
    print("(1) build: no build report"); ok = False
if "2" in steps:
    c = load("f6-check.json")
    if c:
        sb = c.get("source_bytes", {})
        print(f"(2) check: ok={c.get('ok')}; routers_equal {c.get('routers_equal')}; routed layers agree "
              f"{c.get('routed_layers_agree')}, shared {c.get('shared_layers_agree')}; scales "
              f"{c.get('scale_value_min')}..{c.get('scale_value_max')}; bytes: base {sb.get('base', {}).get('checked')} "
              f"checked / {sb.get('base', {}).get('mismatched')} differ, experts {sb.get('experts', {}).get('checked')} / "
              f"{sb.get('experts', {}).get('mismatched')}, dropped present {sb.get('dropped', {}).get('present')}, "
              f"kept missing {sb.get('kept', {}).get('missing')}")
        ok &= bool(c.get("ok"))
    else:
        print("(2) check: no f6-check.json"); ok = False
if "3" in steps:
    t = open(os.path.join(out, "f6-tests.txt")).read() if os.path.exists(os.path.join(out, "f6-tests.txt")) else ""
    m = re.search(r"F6 TESTS rc=(\d+)", t)
    print(f"(3) cut-model tests: rc {m.group(1) if m else 'missing'}")
    ok &= bool(m and m.group(1) == "0")
if "4" in steps:
    s = open(os.path.join(out, "f6-startup.txt")).read() if os.path.exists(os.path.join(out, "f6-startup.txt")) else ""
    est = re.search(r"startup estimate ([\d.]+) GiB", s)
    win = re.search(r"(\d+) streams of (\d+)", s)
    est_v = float(est.group(1)) if est else None
    win_v = int(win.group(2)) if win else None
    print(f"(4) startup: estimate {est_v} GiB (production 88.35 with --vision; text only expect ~87.35), "
          f"{win.group(1) + ' x ' + win.group(2) if win else 'window missing'} (production 3 x 261,872)")
    ok &= est_v is not None and abs(est_v - 87.35) <= 1.5 and win_v is not None and win_v >= 261872
    eq = not os.path.exists(os.path.join(out, "f6-bench-openai.failed"))
    ce = not os.path.exists(os.path.join(out, "f6-concurrent-equal.failed"))
    print(f"(4) exactness: bench_openai --expect-equal {'EQUAL' if eq else 'FAILED'}; concurrent_equal {'EQUAL' if ce else 'FAILED'}")
    ok &= eq and ce
    acc = load("f6-acc-summary.json")
    he = None
    p = os.path.join(out, "f6-acc.jsonl.humaneval.txt")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("{"):
                he = json.loads(line)
    gsm = round(acc["gsm8k_acc"] * acc["gsm8k_n"]) if acc else None
    hep = he["humaneval_pass"] if he else None
    print(f"(4) accuracy: GSM8K {gsm}/{acc['gsm8k_n'] if acc else '?'} (production {gsm_base}/250, gate >= {gsm_base - gsm_tol}); "
          f"HumanEval {hep}/{he['humaneval_n'] if he else '?'} (production {he_base}/164, gate >= {he_base - he_tol})")
    ok &= gsm is not None and gsm >= gsm_base - gsm_tol and hep is not None and hep >= he_base - he_tol
print("F6 release checks: " + ("PASS" if ok else "CHECK"))
sys.exit(0 if ok else 1)
PY
log "F6 done: rc $FAIL (summary $OUT/f6-summary.txt; build report $RELEASE.build-report.json; check $OUT/f6-check.json)"
exit "$FAIL"
