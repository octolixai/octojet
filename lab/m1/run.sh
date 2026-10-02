#!/usr/bin/env bash
# M1 operator entry point. Run inside tensorfold:0.3.6.2 on the GB10, from the repo root:
#   docker run --rm --gpus all --ipc=host -v $HOME/octojet:/octojet -w /octojet tensorfold:0.3.6.2 \
#     bash lab/m1/run.sh results/m1-$(date +%F)
# COMPILE_ONLY=1 builds everything and stops before touching the GPU (no --gpus needed).
set -uo pipefail
out=${1:?usage: run.sh OUTDIR}
mkdir -p "$out"
out=$(cd "$out" && pwd)
touch "$out/m1-probe.jsonl" "$out/m1-fp4.jsonl" "$out/m1-tf.jsonl"  # compare.py reads all three even if a step fails
stamp() { echo "$(date -u +%FT%TZ) $*" >> "$out/m1-timeline.txt"; }  # UTC, to line up with router logs
cd "$(dirname "$0")"
# Arch-specific target: block-scaled FP4 mma exists only on compute_121a. With CUDA 13.3, `-arch=sm_121a` still
# emitted compute_121 PTX (phase A, 2026-09-28) and ptxas rejected the mma; the explicit -gencode form works.
arch=${OJ_ARCH:--gencode arch=compute_121a,code=sm_121a}
build=$(mktemp -d)
{
  nvcc --version | tail -2
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  git -c safe.directory="*" -C ../.. rev-parse --short HEAD
  echo "arch flags: $arch; ARCH in env: ${ARCH:-<unset>}"
} > "$out/m1-compile.txt" 2>&1

# Compile matrix: each optional feature must build *and link* on its own. Conditional graphs may need the device
# runtime (cudaGraphSetConditional); try without it first, then with -rdc=true -lcudadevrt.
flags=""; link=""
for v in FP4 FP8 COND; do
  if nvcc -O3 -std=c++17 $arch -DOJ_$v -o "$build/p_$v" probe.cu >> "$out/m1-compile.txt" 2>&1; then
    echo "OJ_$v: builds" >> "$out/m1-compile.txt"; flags="$flags -DOJ_$v"
  elif [[ $v == COND ]] && nvcc -O3 -std=c++17 $arch -DOJ_$v -rdc=true -o "$build/p_$v" probe.cu -lcudadevrt \
      >> "$out/m1-compile.txt" 2>&1; then
    echo "OJ_$v: builds with -rdc=true -lcudadevrt" >> "$out/m1-compile.txt"; flags="$flags -DOJ_$v"
    link="-rdc=true -lcudadevrt"
  else
    echo "OJ_$v: FAILS to build" >> "$out/m1-compile.txt"
  fi
done
if [[ ${COMPILE_ONLY:-0} == 1 ]]; then
  nvcc -O3 -std=c++17 $arch $flags ${link:+-rdc=true} -o "$build/probe" probe.cu ${link:+-lcudadevrt} \
    >> "$out/m1-compile.txt" 2>&1 && echo "probe: combined build ok" >> "$out/m1-compile.txt" \
    || echo "probe: combined build FAILS" >> "$out/m1-compile.txt"
  nvcc -O3 -std=c++17 $arch -Xptxas -v -o "$build/fp4_bench" fp4_bench.cu >> "$out/m1-compile.txt" 2>&1 \
    && echo "fp4_bench: build ok" >> "$out/m1-compile.txt" || echo "fp4_bench: build FAILS" >> "$out/m1-compile.txt"
  python -c "import tensorfold.cuda.experts" >> "$out/m1-compile.txt" 2>&1 \
    && echo "tensorfold experts: import ok" >> "$out/m1-compile.txt" || echo "tensorfold experts: import FAILS" >> "$out/m1-compile.txt"
  rm -rf "$build"; cat "$out/m1-compile.txt"; exit 0
fi
if nvcc -O3 -std=c++17 $arch $flags ${link:+-rdc=true} -o "$build/probe" probe.cu ${link:+-lcudadevrt} \
    >> "$out/m1-compile.txt" 2>&1; then
  stamp "probe start"
  "$build/probe" | tee "$out/m1-probe.jsonl"
else
  echo "probe: combined build FAILS (flags:$flags)" >> "$out/m1-compile.txt"
  echo '{"check":"probe_build","ok":false}' | tee "$out/m1-probe.jsonl"
fi
# Every step leaves a {"check":...,"ok":false} record when it fails, so compare.py never judges on silence.
if [[ $flags != *OJ_FP4* ]]; then
  echo '{"check":"fp4_compile","ok":false}' | tee "$out/m1-fp4.jsonl"
elif ! nvcc -O3 -std=c++17 $arch -Xptxas -v -o "$build/fp4_bench" fp4_bench.cu >> "$out/m1-compile.txt" 2>&1; then
  echo '{"check":"fp4_bench_build","ok":false}' | tee "$out/m1-fp4.jsonl"
else
  stamp "fp4_bench start"
  "$build/fp4_bench" --check | tee "$out/m1-fp4.jsonl"
  [[ ${PIPESTATUS[0]} == 0 ]] || echo '{"check":"fp4_bench_run","ok":false}' | tee -a "$out/m1-fp4.jsonl"
fi
stamp "tf_baseline start"
python tf_baseline.py | tee "$out/m1-tf.jsonl"
[[ ${PIPESTATUS[0]} == 0 ]] || echo '{"check":"tf_baseline_run","ok":false}' | tee -a "$out/m1-tf.jsonl"
# Repeat FP4 timing after TensorFold: if it differs much from the first pass, something else shared the GPU.
if [[ -x $build/fp4_bench ]]; then
  stamp "fp4_bench repeat start"
  "$build/fp4_bench" --rows 1,8,512,2048,8192 > "$out/m1-fp4-repeat.jsonl" 2>/dev/null
fi
stamp "done"
python compare.py "$out/m1-probe.jsonl" "$out/m1-fp4.jsonl" "$out/m1-tf.jsonl" \
  | tee "$out/m1-summary.md"
rm -rf "$build"
