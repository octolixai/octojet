#!/usr/bin/env bash
set -euo pipefail
OUT=$HOME/octojet-runs/f2c-phase1; P=$HOME/tensorfold/octojet-profiles; mkdir -p "$OUT" "$P"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee "$OUT/f2c-phase1-tenants.txt"   # the pause scripts rely on this query
docker ps --filter label=llm-research.gpu-job=true --format '{{.Names}} {{.Status}}' | tee -a "$OUT/f2c-phase1-tenants.txt"   # research GPU jobs (operator's llm-research-spark-*)
docker run --rm --gpus all --ipc=host --cap-add SYS_ADMIN -v "$HOME/octojet-f2c:/octojet:ro" -v "$HOME/tensorfold:/tf" \
  -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache:/root/.triton" \
  -e PYTHONPATH=/octojet/engine/src -w /tmp --entrypoint bash tensorfold:0.3.6.2 -c '
    set -euo pipefail; NSYS=/usr/local/cuda/bin/nsys
    $NSYS --version | tee /tf/octojet-profiles/nsys-version.txt
    $NSYS profile --help > /tf/octojet-profiles/nsys-profile-help.txt
    $NSYS stats --help-reports > /tf/octojet-profiles/nsys-stats-reports.txt 2>&1 || true
    TRACE=cuda-sw; grep -q "cuda-sw" /tf/octojet-profiles/nsys-profile-help.txt || TRACE=cuda
    GPUM="--gpu-metrics-devices=0 --gpu-metrics-frequency=10000"; grep -q "gpu-metrics-devices" /tf/octojet-profiles/nsys-profile-help.txt || GPUM=""
    rm -f /tf/octojet-profiles/preflight.nsys-rep /tf/octojet-profiles/preflight.sqlite /tf/octojet-profiles/preflight-kern.csv /tf/octojet-profiles/preflight-ranges.json
    cat > /tmp/preflight.py <<PY
import threading, torch
import sys
nv = torch.cuda.nvtx; result = {}
def work():          # the shapes the captures produce: an admission range, block ranges, kernels, a memset, a memcpy
    try:
        x = torch.ones(1024, 1024, device="cuda"); z = torch.empty(1 << 20, device="cuda"); torch.cuda.synchronize()
        rc1 = torch.cuda.cudart().cudaProfilerStart()
        nv.range_push("admission")
        nv.range_push("main:router"); y = x @ x; z.zero_(); nv.range_pop()
        nv.range_push("main:expert_up"); y = y @ x; h = y[:1].cpu(); nv.range_pop()
        torch.cuda.synchronize(); nv.range_pop()
        rc2 = torch.cuda.cudart().cudaProfilerStop()
        result["rc"] = (int(rc1), int(rc2)); print("profiler rc", int(rc1), int(rc2), float(h.sum()), flush=True)
    except Exception as exc:
        result["error"] = repr(exc); print("preflight error:", repr(exc), flush=True)
t = threading.Thread(target=work); t.start(); t.join()
sys.exit(0 if result.get("rc") == (0, 0) else 1)     # a non-zero profiler rc or a worker exception fails the capture
PY
    try_capture(){  # $1 = GPU-metrics flags or "": a capture counts only if nsys exits 0, the report exists AND the program printed its rc line
      rm -f /tf/octojet-profiles/preflight.nsys-rep /tf/octojet-profiles/preflight-capture.log
      $NSYS profile --trace=$TRACE,nvtx,osrt --sample=none --cpuctxsw=none $1 --capture-range=cudaProfilerApi \
        --capture-range-end=stop --wait=all -f true -o /tf/octojet-profiles/preflight python /tmp/preflight.py 2>&1 | tee /tf/octojet-profiles/preflight-capture.log
      [ "${PIPESTATUS[0]}" = 0 ] || { echo "capture: nsys exit ${PIPESTATUS[0]}"; return 1; }
      [ -s /tf/octojet-profiles/preflight.nsys-rep ] || { echo "capture: no report generated"; return 1; }
      grep -q "profiler rc 0 0" /tf/octojet-profiles/preflight-capture.log || { echo "capture: program did not print profiler rc 0 0 (OOM or crash under the profiler?)"; return 1; }; }
    if [ -n "$GPUM" ] && ! try_capture "$GPUM"; then echo "GPU metrics sampling FAILED on this GPU (see above); retrying without it"; GPUM=""; fi
    [ -n "$GPUM" ] || try_capture "" || { echo "PREFLIGHT FAILED: capture without GPU metrics failed too"; exit 1; }
    $NSYS stats --report cuda_gpu_kern_sum --format csv /tf/octojet-profiles/preflight.nsys-rep > /tf/octojet-profiles/preflight-kern.csv
    $NSYS export --type sqlite -f true -o /tf/octojet-profiles/preflight.sqlite /tf/octojet-profiles/preflight.nsys-rep
    python3 /octojet/bench/nsys_ranges.py /tf/octojet-profiles/preflight.sqlite --out /tf/octojet-profiles/preflight-ranges.json
    python3 - <<PY
import json; r = json.load(open("/tf/octojet-profiles/preflight-ranges.json"))
assert r["capture_bounds"] == "admission", r["capture_bounds"]
for name in ("main:router", "main:expert_up"):
    assert r["ranges"][name]["busy_ms"] > 0 and r["ranges"][name]["launches"] >= 1, (name, r["ranges"].get(name))
print("preflight ranges ok:", {k: v["launches"] for k, v in r["ranges"].items()}, "counters:", r["counters"])
PY
    echo "TRACE=$TRACE" > /tf/octojet-profiles/nsys-flags.env; echo "GPUM=\"$GPUM\"" >> /tf/octojet-profiles/nsys-flags.env
    wc -l /tf/octojet-profiles/preflight-kern.csv; cat /tf/octojet-profiles/nsys-flags.env' 2>&1 | tee "$OUT/f2c-phase1-preflight.txt"
