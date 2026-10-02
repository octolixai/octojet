#!/usr/bin/env bash
set -euo pipefail
OUT=$HOME/octojet-runs/f2c-phase1; SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$OUT" "$SCRATCH"
clean_scratch(){ docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh tensorfold:0.3.6.2 -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true'; rmdir "$SCRATCH" 2>/dev/null || true; }
trap clean_scratch EXIT                    # the container writes root-owned files: remove them from inside a container, never fail the run over it
docker run --rm --gpus all --ipc=host -v "$HOME/octojet-f2c:/octojet:ro" -v "$HOME/tensorfold:/tf:ro" -v /srv/ai:/srv/ai:ro -v "$SCRATCH:/scratch" \
  -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache:/root/.triton" \
  -e PYTHONPATH=/octojet/engine/src -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/scratch -e OCTOJET_CACHE_DIR=/scratch/default \
  -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=2 -w /octojet/engine --entrypoint bash tensorfold:0.3.6.2 -c '
    set -euo pipefail; pip install -q "pytest>=8,<10"
    python -m pytest tests/test_prefill_timing.py tests/test_prefill_timing_cuts.py tests/test_prefill_timing_scheduler.py tests/test_prefill_timing_server.py /octojet/bench/test_prefill_prompt.py /octojet/bench/test_prefill_bench.py /octojet/bench/test_nsys_ranges.py -q -p no:cacheprovider
    python -m pytest tests/cuda/test_prefill_timing_flashnext.py tests/cuda/test_qwen4_exp_nvfp4.py -q -p no:cacheprovider --basetemp=/scratch/pytest' 2>&1 | tee "$OUT/f2c-phase1-cuda-tests.txt"
