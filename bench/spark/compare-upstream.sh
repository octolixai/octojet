#!/usr/bin/env bash
# Octojet vs the fastest published setup (owner's request, 2026-10-01): upstream TensorFold 0.6.0 (latest release) on
# the two NVFP4 Flash Next checkpoints it names for Sparks, against Octojet main on the mixed checkpoint.
# Same box, same prompts, GPU exclusive (no research job), page cache dropped before each start, text only (upstream
# refuses images on NVFP4 checkpoints), int8 KV, --parallel 3 for every server.
#
# Configs (run in order; CONFIGS="ours up-lil up-radix" by default; a config whose checkpoint is missing is skipped):
#   ours      Octojet ($REPO, image $IMAGE), /tf/flashnext-nvfp4-mixed, packed cache, default prefix checkpoints (4)
#   up-lil    TensorFold 0.6.0 (installed into $UP_SRC), $UP_LIL   = local-inference-lab/Qwen3.8-Flash-Next-NVFP4
#   up-radix  TensorFold 0.6.0,                         $UP_RADIX = RadixArk/Qwen3.8-Flash-Next-NVFP4 (HF main)
# Per config (about 60-75 min each; ACC=0 skips accuracy and saves ~30 min each):
#   startup  window per stream, startup estimate, load time            -> <cfg>-server.log, <cfg>-startup.txt
#   warm-up  one short request (JIT/graph compile) before any timing
#   ttft     cold m32k / m128k / m210k, one stream, 64 tokens (client side, text prompts) -> <cfg>-ttft.jsonl
#   decode   bench_openai, 64 tokens x 10 reps, temperature 1.0 and 0  -> <cfg>-decode.json
#   agent    agent_bench: 80k conversation + 3 turns of +5k tokens     -> <cfg>-agent.jsonl
#   variant  cP (71,444) cold, then cPp (shares 69,013)                -> <cfg>-ttft.jsonl
#   resend   cP again, identical                                       -> <cfg>-ttft.jsonl
#   lanes    a decoding stream's largest gap while m128k is admitted   -> <cfg>-gap.json
#   acc      GSM8K 250 + HumanEval 164 (greedy, thinking off)          -> <cfg>-acc.jsonl (+ run_humaneval.sh)
# Then compare-summary.txt (one table, all configs).
# MODE=short: warm-up, cold m32k then a second, different cold 32k prompt, the variant trio and accuracy only (about 40 min a config).
#
# Usage: bash bench/spark/compare-upstream.sh --prepare   # once: install TensorFold 0.6.0 into $UP_SRC (pip --target)
#        bash bench/spark/compare-upstream.sh [--dry-run]
# Production oj-serve is STOPPED for the whole run (owner's standing test approval; PROD_STOP) with the standby
# watcher paused; every exit path removes the pause and the operator's watcher restarts oj-serve.
set -euo pipefail
REPO=${REPO:-$HOME/octojet-cmp}; OUT=${OUT:-$HOME/octojet-runs/compare}; NAME=${NAME:-oj-cmp}; PARALLEL=3
PREFIX=cmp; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
UP_TAG=${UP_TAG:-v0.6.0}; UP_IMAGE=${UP_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}; UP_SRC=${UP_SRC:-$HOME/tensorfold/tf-upstream-$UP_TAG}
UP_LIL=${UP_LIL:-}; UP_RADIX=${UP_RADIX:-}; CONFIGS=${CONFIGS:-ours up-lil up-radix}; ACC=${ACC:-1}; MODE=${MODE:-full}
DRYRUN=0; PREPARE=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRYRUN=1; shift;; --prepare) PREPARE=1; shift;;
  *) echo "usage: $0 [--prepare | --dry-run]" >&2; exit 2;; esac; done

if [ "$PREPARE" = 1 ]; then
  mkdir -p "$UP_SRC"
  docker run --rm -v "$UP_SRC:/up" --entrypoint bash "$UP_IMAGE" -c \
    "pip install -q --no-deps --target /up 'git+https://github.com/ashhart/TensorFold.git@$UP_TAG' && python -c 'import sys; sys.path.insert(0, \"/up\"); import tensorfold; print(\"tensorfold\", tensorfold.__version__)'"
  echo "installed into $UP_SRC; if serving fails on a missing module, add it with pip install --target $UP_SRC <module>"
  exit 0
fi

cd "$REPO"
for m in m32k m128k m210k cP cPp; do [ -f "$M/$m.json" ] || { echo "error: $M/$m.json missing" >&2; exit 2; }; done
ckpt_of(){ case "$1" in ours) echo /tf/flashnext-nvfp4-mixed;; up-lil) echo "$UP_LIL";; up-radix) echo "$UP_RADIX";; esac; }
RUN=()
for c in $CONFIGS; do
  case "$c" in ours|up-lil|up-radix) ;; *) echo "error: unknown config $c" >&2; exit 2;; esac
  p=$(ckpt_of "$c")
  if [ "$c" != ours ] && { [ -z "$p" ] || [ ! -e "$p/config.json" ]; }; then echo "skip $c: checkpoint not set or missing ($p)"; continue; fi
  if [ "$c" != ours ] && [ ! -d "$UP_SRC/tensorfold" ]; then echo "error: $UP_SRC has no tensorfold (run --prepare)" >&2; exit 2; fi
  RUN+=("$c")
done
[ ${#RUN[@]} -gt 0 ] || { echo "error: nothing to run" >&2; exit 2; }
echo "configs: ${RUN[*]}"

serve(){  # $1 config: start the server with that config's engine and checkpoint, wait for /v1/models
  local c=$1 p; p=$(ckpt_of "$c")
  tenants_clear "$c" || exit 1
  drop_cache || { log "$c: page cache drop FAILED"; exit 1; }; free -g | tee -a "$TL"
  local args=(serve "$p" --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel "$PARALLEL")
  if [ "$c" = ours ]; then
    args+=(--packed-cache /tf/octojet-cache/packed)
    docker run --rm --gpus all --ipc=host --network host --name "$NAME" "${MOUNTS[@]}" "${ENVS[@]}" -w /octojet/engine \
      --entrypoint python "$IMAGE" -m tensorfold.cli "${args[@]}" > "$OUT/$c-server.log" 2>&1 &
  else
    mkdir -p "$HOME/tensorfold/torch-ext-up" "$HOME/tensorfold/triton-cache-up"
    docker run --rm --gpus all --ipc=host --network host --name "$NAME" -v "$UP_SRC:/up:ro" -v "$HOME/tensorfold:/tf" \
      -v /srv/ai:/srv/ai:ro -v "$p:$p:ro" -v "$HOME/tensorfold/torch-ext-up:/root/.cache/torch_extensions" \
      -v "$HOME/tensorfold/triton-cache-up:/root/.triton" -e PYTHONPATH=/up -e PYTHONDONTWRITEBYTECODE=1 \
      --entrypoint python "$UP_IMAGE" -m tensorfold.cli "${args[@]}" > "$OUT/$c-server.log" 2>&1 &
  fi
  PID=$!; printf '%s\n' "$c" "$p" "${args[@]}" > "$OUT/$c-args.txt"; local t0=$SECONDS
  until curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$c: server exited before it was healthy (see $OUT/$c-server.log)"; return 1; }
    [ $((SECONDS - t0)) -lt "$READY_DEADLINE" ] || { log "$c: not healthy after ${READY_DEADLINE}s"; return 1; }; sleep 1
  done
  log "$c: healthy after $((SECONDS - t0))s"
  grep -h -i "startup estimate\|loaded in\|streams of\|window\|packed tables" "$OUT/$c-server.log" | tee "$OUT/$c-startup.txt" | tee -a "$TL" || true
  echo "healthy_after_s $((SECONDS - t0))" >> "$OUT/$c-startup.txt"; }

step(){ local what=$1; shift; log "$C: $what"; "$@" 2>&1 | tee -a "$TL" || { log "$C: $what FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }; }
T=$OUT/prompts   # text versions of the manifests (upstream's completions take text only)
probe(){ python3 bench/cmp_probe.py "$@"; }
ttft(){ probe ttft http://127.0.0.1:8080 f1 --prompt "$T/$1.txt" --label "$2" --out "$OUT/$C-ttft.jsonl"; }

measure_short(){   # MODE=short: the open items only (32k twice, the variant trio, accuracy)
  step warm-up probe ttft http://127.0.0.1:8080 f1 --prompt "$T/warm.txt" --label warm-up --out "$OUT/$C-warmup.jsonl"
  step "cold m32k" ttft m32k cold-m32k
  step "cold m32k, second prompt" ttft m32k-b cold-m32k-b   # unique first line: cold again, after the first long prompt
  step "variant: cP cold" ttft cP P-cold
  step "variant: cPp" ttft cPp Pp-variant
  step "resend: cP" ttft cP P-resend
  step accuracy bash -c "python3 bench/acc_eval.py http://127.0.0.1:8080 f1 '$OUT/$C-acc' > '$OUT/$C-acc-summary.json'"
  step humaneval bash bench/run_humaneval.sh "$OUT/$C-acc.jsonl"; }

measure(){   # client-side timing, identical for every engine
  if [ "$MODE" = short ]; then measure_short; return; fi
  step warm-up probe ttft http://127.0.0.1:8080 f1 --prompt "$T/warm.txt" --label warm-up --out "$OUT/$C-warmup.jsonl"
  for m in m32k m128k m210k; do step "cold $m" ttft "$m" "cold-$m"; done
  step "variant: cP cold" ttft cP P-cold
  step "variant: cPp" ttft cPp Pp-variant
  step "resend: cP" ttft cP P-resend
  step decode python3 engine/tools/bench_openai.py http://127.0.0.1:8080 f1 --tokens 64 --reps 10 --allow-missing-ids --label "$C" --output "$OUT/$C-decode.json"
  step agent bash -c "python3 bench/agent_bench.py http://127.0.0.1:8080 f1 --base-tokens 80000 --new 5000 --turns 3 --label $C > '$OUT/$C-agent.jsonl'"
  step lanes probe gap http://127.0.0.1:8080 f1 --prompt "$T/m128k.txt" --out "$OUT/$C-gap.json"
  if [ "$ACC" = 1 ]; then
    step accuracy bash -c "python3 bench/acc_eval.py http://127.0.0.1:8080 f1 '$OUT/$C-acc' > '$OUT/$C-acc-summary.json'"
    step humaneval bash bench/run_humaneval.sh "$OUT/$C-acc.jsonl"
  fi; }

mkdir -p "$T"
log "prompts: manifests -> text (container, the mixed checkpoint's tokenizer)"
docker run --rm -v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v "$OUT:/out" -w /octojet --entrypoint python "$IMAGE" \
  bench/cmp_probe.py text --tokenizer /tf/flashnext-nvfp4-mixed/tokenizer.json \
  /tf/octojet-manifests/m32k.json /tf/octojet-manifests/m128k.json /tf/octojet-manifests/m210k.json \
  /tf/octojet-manifests/cP.json /tf/octojet-manifests/cPp.json --out-dir /out/prompts | tee -a "$TL"
printf "Say hello in five words." > "$T/warm.txt"
{ printf "Second run, a different prompt.\n"; cat "$T/m32k.txt"; } > "$T/m32k-b.txt"
for m in m32k m128k m210k cP cPp; do [ -s "$T/$m.txt" ] || { echo "error: $T/$m.txt not written" >&2; exit 2; }; done
if [ "$DRYRUN" = 1 ]; then echo "dry-run OK: configs ${RUN[*]}, prompts in $T"; exit 0; fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true; echo "$UP_TAG" > "$OUT/upstream-tag.txt"
pause_on
log "WINDOW: stopping production oj-serve ($PROD_STOP)"
bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
port_free || { log "WINDOW: port 8080 still listening"; exit 1; }
for C in "${RUN[@]}"; do
  if serve "$C"; then measure; else FAIL=1; fi
  stop "$C"
done
python3 bench/compare_summary.py "$OUT" "${RUN[@]}" | tee "$OUT/compare-summary.txt" | tee -a "$TL" || FAIL=1
log "compare done: rc $FAIL (summary $OUT/compare-summary.txt)"
exit "$FAIL"
