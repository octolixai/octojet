// M1 probe: toolkit facts, peak mma.sync throughput per format, block-scale fragment layout, conditional graphs.
// Build: nvcc -O3 -std=c++17 -arch=sm_121a [-DOJ_FP4] [-DOJ_FP8] [-DOJ_COND] -o probe probe.cu
#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "ref.h"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      printf("{\"check\":\"error\",\"ok\":false,\"where\":\"%s\",\"msg\":\"%s\"}\n", #x, cudaGetErrorString(e_)); \
      return 1;                                                                            \
    }                                                                                      \
  } while (0)

// ---- MMA wrappers
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#ifdef OJ_FP8
__device__ __forceinline__ void mma_fp8(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

#ifdef OJ_FP4
#include "fp4mma.cuh"
#endif

// ---- Peak throughput: 8 independent accumulator chains per warp, operands in registers.
constexpr int CHAINS = 8, ITERS = 4096;

template <int FMT>
__global__ void peak_kernel(float* sink, uint32_t seed) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = seed * (threadIdx.x + 1) * (i + 3) & 0x33333333u;  // small finite values
  for (int i = 0; i < 2; ++i) b[i] = seed * (threadIdx.x + 7) * (i + 5) & 0x33333333u;
  [[maybe_unused]] const uint32_t sc = 0x38383838u;  // UE4M3 1.0 in every byte (fp4 only)
  float d[CHAINS][4] = {};
  for (int it = 0; it < ITERS; ++it)
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
      if (FMT == 0) mma_bf16(d[c], a, b);
#ifdef OJ_FP8
      if (FMT == 1) mma_fp8(d[c], a, b);
#endif
#ifdef OJ_FP4
      if (FMT == 2) mma_fp4<0, 0>(d[c], a, b, sc, sc);
#endif
    }
  float s = 0;
  for (int c = 0; c < CHAINS; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  if (s == 12345.f) sink[0] = s;  // keep the work alive
}

template <int FMT>
int run_peak(const char* name, double flops_per_mma, int sms) {
  float* sink;
  CK(cudaMalloc(&sink, 4));
  const int blocks = sms * 8, threads = 128;
  peak_kernel<FMT><<<blocks, threads>>>(sink, 3);  // warm-up
  CK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  std::vector<float> ms(5);
  for (auto& m : ms) {
    cudaEventRecord(t0);
    peak_kernel<FMT><<<blocks, threads>>>(sink, 3);
    cudaEventRecord(t1);
    CK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&m, t0, t1);
  }
  std::sort(ms.begin(), ms.end());
  const double mmas = double(blocks) * (threads / 32) * ITERS * CHAINS;
  printf("{\"check\":\"peak\",\"format\":\"%s\",\"tflops\":%.1f,\"ms\":%.3f}\n", name,
         mmas * flops_per_mma / (ms[2] * 1e-3) / 1e12, ms[2]);
  cudaFree(sink);
  return 0;
}

// ---- Conditional graph: a device-side while loop whose body increments a counter until it reaches `target`.
#ifdef OJ_COND
__global__ void cond_body(int* counter, int target, cudaGraphConditionalHandle h) {
  const int c = ++counter[0];
  cudaGraphSetConditional(h, c < target ? 1 : 0);
}

int run_cond() {
  const int target = 1000;
  int* counter;
  CK(cudaMalloc(&counter, 4));
  CK(cudaMemset(counter, 0, 4));
  cudaGraph_t graph;
  CK(cudaGraphCreate(&graph, 0));
  cudaGraphConditionalHandle h;
  CK(cudaGraphConditionalHandleCreate(&h, graph, 1, cudaGraphCondAssignDefault));
  cudaGraphNodeParams cp = {};
  cp.type = cudaGraphNodeTypeConditional;
  cp.conditional.handle = h;
  cp.conditional.type = cudaGraphCondTypeWhile;
  cp.conditional.size = 1;
  cudaGraphNode_t cnode;
#if CUDART_VERSION >= 13000
  CK(cudaGraphAddNode(&cnode, graph, nullptr, nullptr, 0, &cp));  // CUDA 13 adds edge data
#else
  CK(cudaGraphAddNode(&cnode, graph, nullptr, 0, &cp));
#endif
  cudaGraph_t body = cp.conditional.phGraph_out[0];
  cudaKernelNodeParams kp = {};
  void* args[] = {&counter, (void*)&target, &h};
  kp.func = (void*)cond_body;
  kp.gridDim = dim3(1); kp.blockDim = dim3(1);
  kp.kernelParams = args;
  cudaGraphNode_t knode;
  CK(cudaGraphAddKernelNode(&knode, body, nullptr, 0, &kp));
  cudaGraphExec_t exec;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  CK(cudaGraphLaunch(exec, 0));
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1));
  float ms;
  cudaEventElapsedTime(&ms, t0, t1);
  int got = 0;
  CK(cudaMemcpy(&got, counter, 4, cudaMemcpyDeviceToHost));
  printf("{\"check\":\"cond_graph\",\"ok\":%s,\"iters\":%d,\"us_per_iter\":%.2f}\n", got == target ? "true" : "false",
         got, ms * 1000.f / got);
  cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaFree(counter);
  return 0;
}
#endif

int main() {
  int dev = 0, rt = 0, drv = 0;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  printf("{\"check\":\"toolkit\",\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"runtime\":%d,\"driver\":%d,"
         "\"fp4\":%s,\"fp8\":%s,\"cond\":%s}\n",
         p.name, p.major, p.minor, p.multiProcessorCount, rt, drv,
#ifdef OJ_FP4
         "true",
#else
         "false",
#endif
#ifdef OJ_FP8
         "true",
#else
         "false",
#endif
#ifdef OJ_COND
         "true"
#else
         "false"
#endif
  );
  if (run_peak<0>("bf16", 2.0 * 16 * 8 * 16, p.multiProcessorCount)) return 1;
#ifdef OJ_FP8
  if (run_peak<1>("fp8", 2.0 * 16 * 8 * 32, p.multiProcessorCount)) return 1;
#endif
#ifdef OJ_FP4
  if (run_peak<2>("fp4bs", 2.0 * 16 * 8 * 64, p.multiProcessorCount)) return 1;
  print_scale_map(discover_scale_map());
#endif
#ifdef OJ_COND
  if (run_cond()) return 1;
#endif
  return 0;
}
