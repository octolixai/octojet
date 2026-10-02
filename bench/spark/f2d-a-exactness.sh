#!/usr/bin/env bash
# F2d stage A — CORRECTNESS (beside research, ~30 min). 1) The container tests on the 4-layer NVFP4 cut (its PLE layer
# is index 1 and its attention layer index 3, so PLE, QSA and pool construction are active; the chunk-size test is
# stage-B evidence and runs separately). 2) Acceptance receipts against a test server with production's flags: two
# identical 71k calls at once (cold + busy miss), an identical third call (exact hit), bench_openai --expect-equal
# (drafted == serial; the repeated prompt makes later repetitions exact hits) and concurrent_equal.
set -euo pipefail
source "$(dirname "$0")/f2d-a-lib.sh"
cd "$REPO"
[ -f "$M/cP.json" ] || { echo "error: $M/cP.json missing; run bench/spark/f2d-a-manifests.sh first" >&2; exit 2; }
SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
clean_scratch(){ docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true'; rmdir "$SCRATCH" 2>/dev/null || true; }
trap 'rc=$?; set +e; clean_scratch; (exit $rc); cleanup' EXIT   # root-owned scratch files are removed from inside a container; set +e so a failed run still reaches cleanup (pause off, container stopped, exit code logged)
# bench/spark/test_f2d_a_scripts.py runs on the host (it forks bash; the container's pids limit breaks it)
CPU="tests/test_prefix_reuse.py tests/test_prefix_reuse_multi.py tests/test_prefix_reuse_engine.py tests/test_prefix_reuse_server.py tests/test_cuda_stream_slots.py tests/test_cuda_tool_choice.py /octojet/bench/test_prefill_prompt.py /octojet/bench/test_prefill_bench.py /octojet/bench/test_f2d_gate.py"
log "exactness: container tests (OCTOJET_NVFP4_FLASHNEXT_LAYERS=4)"
docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" -v "$SCRATCH:/scratch" "${ENVS[@]}" -e TMPDIR=/scratch -e OCTOJET_CACHE_DIR=/scratch/default \
  -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 -w /octojet/engine --entrypoint bash "$IMAGE" -c "
    set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
    python -m pytest $CPU -q -p no:cacheprovider || rc=1
    python -m pytest tests/cuda/test_prefix_reuse_flashnext.py -k 'not chunk_sizes' tests/cuda/test_flashnext_multi.py -q -p no:cacheprovider --basetemp=/scratch/pytest || rc=1
    echo \"STAGE-A TESTS rc=\$rc\"
    if python -m pytest tests/cuda/test_prefix_reuse_flashnext.py -k chunk_sizes -q -p no:cacheprovider --basetemp=/scratch/pytest2; then echo 'CHUNK-SIZE TEST (stage-B evidence) rc=0'; else echo 'CHUNK-SIZE TEST (stage-B evidence) rc=1'; fi
    exit \$rc" 2>&1 | tee "$OUT/f2d-a-cuda-tests.txt" || { log "exactness: container tests FAILED (rc ${PIPESTATUS[0]})"; exit 1; }
log "exactness: container tests passed"
start acceptance
A=(--label acceptance --server-label acceptance --stage A --rep 1 --out "$OUT/f2d-a-acceptance.jsonl")
log "acceptance: two identical 71k calls at once (cold + busy miss, recorded), then an identical third call (must be an exact hit)"
bench --manifest "$M/cP.json" --arm clean:1 --draft on --concurrent 2 --expect-reuse any --expect-cached any "${A[@]}" | tee -a "$TL" \
  || log "acceptance: the busy probe reported an error row (rc ${PIPESTATUS[0]})"
bench --manifest "$M/cP.json" --arm clean:1 --draft on --expect-reuse exact --expect-cached 71444 "${A[@]}" | tee -a "$TL" \
  || { log "acceptance: the identical third call was NOT an exact hit (rc ${PIPESTATUS[0]})"; FAIL=1; }
log "acceptance: bench_openai --expect-equal"
python3 engine/tools/bench_openai.py http://127.0.0.1:8080 f1 --expect-equal --output "$OUT/f2d-a-bench-openai.json" | tee -a "$TL" \
  || { log "acceptance: bench_openai FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
log "acceptance: concurrent_equal"
python3 bench/concurrent_equal.py http://127.0.0.1:8080 f1 --prompts 2 --out "$OUT/f2d-a-concurrent-equal.json" | tee -a "$TL" \
  || { log "acceptance: concurrent_equal FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
stop acceptance
cp "$M/cP.json" "$OUT/"
log "exactness done: rc $FAIL"
exit "$FAIL"
