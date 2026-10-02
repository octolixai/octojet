// M1: W4A4 NVFP4 grouped expert pipeline at Flash Next shapes. Correctness vs CPU, row invariance, timing.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -o fp4_bench fp4_bench.cu
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "fp4mma.cuh"
#include "ref.h"

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      printf("{\"check\":\"cuda\",\"ok\":false,\"where\":\"%s:%d\"}\n", __FILE__, __LINE__);         \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

constexpr int TILE = 128;         // pairs an item holds: eight m16 sub-tiles, so each expert's weights are read
constexpr int SUBS = TILE / 16;   // once per 128 pairs; sub-tiles with no pairs are skipped
constexpr float G_ACT = 1.0f;     // static activation global scale (synthetic activations are O(1))
constexpr float G_W = 1.0f;       // weight global scale
constexpr int GATE_LD = 73;       // odd row stride (floats): the epilogue's per-row reads are bank-conflict free

// Activation tiles: item i owns SUBS sub-tiles, each K/64 k-steps of A_KSTEP_U32 words.
__host__ __device__ inline size_t act_off(int item, int sub, int K) {
  return (size_t(item) * SUBS + sub) * (K / 64) * A_KSTEP_U32;
}

// A warp per row: lane l quantizes k-steps l, l+32, ... of the gathered bf16 row and writes them into its tile.
// Rows past `cnt` in a used sub-tile are written as zeros; sub-tiles with no pairs are left alone (never read).
__global__ void __launch_bounds__(128) quant_gather(const __nv_bfloat16* __restrict__ x, int slots, int K,
                                                    const int* __restrict__ items, int n_items,
                                                    const int* __restrict__ members, uint32_t* __restrict__ tiles) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int item = blockIdx.x / (TILE / 4), r = (blockIdx.x % (TILE / 4)) * 4 + warp;
  if (item >= n_items) return;
  const int first = items[3 * item + 1], cnt = items[3 * item + 2];
  if ((r & ~15) >= cnt) return;
  const __nv_bfloat16* src = r < cnt ? x + size_t(members[first + r] / slots) * K : nullptr;
  uint32_t* tile = tiles + act_off(item, r >> 4, K);
  for (int ks = lane; ks < K / 64; ks += 32) {
    uint8_t codes[64], sc[4];
    float v[16];
    for (int j = 0; j < 4; ++j) {
      if (src) {  // 16 bf16 = two 16-byte loads (rows are K*2 bytes with K % 64 == 0, so aligned)
        const uint4* p = reinterpret_cast<const uint4*>(src + ks * 64 + j * 16);
        const uint4 q[2] = {p[0], p[1]};
        const uint32_t* w = reinterpret_cast<const uint32_t*>(q);
        for (int i = 0; i < 8; ++i) {
          v[2 * i] = __uint_as_float(w[i] << 16);            // bf16 -> fp32: the high half of an fp32
          v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        }
      } else {
        for (int i = 0; i < 16; ++i) v[i] = 0.f;
      }
      quantize_block16(v, G_ACT, codes + 16 * j, sc + j);
    }
    pack_act_row(codes, sc, 64, r & 15, tile + size_t(ks) * A_KSTEP_U32);
  }
}

// One warp's share of a grouped GEMM: all used sub-tiles of `item` times this warp's 16 packed columns
// (n8 tiles nt0, nt0+1). The weight fragments of a k-step are loaded once and reused by every sub-tile.
// Fixed K order and no cross-row arithmetic: each row's bits are independent of the other rows (row-invariant).
template <int SA, int SB>
__device__ __forceinline__ void accumulate(float (&acc)[SUBS][2][4], const uint32_t* __restrict__ tiles, int item,
                                           int K, const uint32_t* __restrict__ We, int nt0, int cnt, int lane,
                                           int a_pat) {
  const int KS = K / 64, subs = (cnt + 15) >> 4;
  const int ra = sfa_row(lane, SA, a_pat), cb = sfb_col(lane, SB);
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t b[2][2], sb[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint32_t* tile = We + (size_t(nt0 + j) * KS + ks) * W_TILE_U32;
      const uint2 v = reinterpret_cast<const uint2*>(tile)[lane];
      b[j][0] = v.x; b[j][1] = v.y;
      sb[j] = cb >= 0 ? tile[64 + cb] : 0u;
    }
#pragma unroll
    for (int s = 0; s < SUBS; ++s) {
      if (s < subs) {
        const uint32_t* step = tiles + act_off(item, s, K) + size_t(ks) * A_KSTEP_U32;
        const uint4 v = reinterpret_cast<const uint4*>(step)[lane];
        const uint32_t a[4] = {v.x, v.y, v.z, v.w};
        const uint32_t sa = ra >= 0 ? step[128 + ra] : 0u;
        mma_fp4<SA, SB>(acc[s][0], a, b[0], sa, sb[0]);
        mma_fp4<SA, SB>(acc[s][1], a, b[1], sa, sb[1]);
      }
    }
  }
}

// Up GEMM with SwiGLU and quantization in the epilogue. CTA = (item, 64 intermediate columns) = 128 packed
// weight columns (gate_up_row order): warps 0-3 compute gate, warps 4-7 the matching up columns. The result is the
// down GEMM's input tile for k-step `slab`, so the fp32 intermediate never leaves the chip. WRITE_H (checks only)
// also stores gate and up in natural column order.
template <int SA, int SB, bool WRITE_H>
__global__ void __launch_bounds__(256, 2) gemm_up_swiglu(const uint32_t* __restrict__ tiles,
                                                      const uint32_t* __restrict__ W, int D, int I,
                                                      const int* __restrict__ items, int n_items,
                                                      uint32_t* __restrict__ out_tiles, float* __restrict__ h,
                                                      int a_pat) {
  __shared__ float act_s[TILE][GATE_LD];
  const int slabs = I / 64, item = blockIdx.x / slabs, slab = blockIdx.x - item * slabs;
  if (item >= n_items) return;
  const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int subs = (cnt + 15) >> 4, lc0 = (warp & 3) * 16;
  const uint32_t* We = W + size_t(e) * (2 * I / 8) * (D / 64) * W_TILE_U32;
  float acc[SUBS][2][4] = {};
  accumulate<SA, SB>(acc, tiles, item, D, We, (slab * 128 + warp * 16) / 8, cnt, lane, a_pat);
  const float gs = G_ACT * G_W;
  if (warp < 4) {  // gate
#pragma unroll
    for (int s = 0; s < SUBS; ++s)
      if (s < subs)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int r = s * 16 + g + 8 * hh, col = lc0 + j * 8 + 2 * t;
            act_s[r][col] = acc[s][j][2 * hh] * gs;
            act_s[r][col + 1] = acc[s][j][2 * hh + 1] * gs;
          }
  }
  __syncthreads();
  if (warp >= 4) {  // up: same (row, column) positions as the matching gate warp
#pragma unroll
    for (int s = 0; s < SUBS; ++s)
      if (s < subs)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              const int r = s * 16 + g + 8 * hh, col = lc0 + j * 8 + 2 * t + q;
              const float gv = act_s[r][col], uv = acc[s][j][2 * hh + q] * gs;
              if (WRITE_H && r < cnt) {
                float* row = h + size_t(first + r) * 2 * I;
                row[slab * 64 + col] = gv;
                row[I + slab * 64 + col] = uv;
              }
              act_s[r][col] = gv / (1.f + __expf(-gv)) * uv;
            }
  }
  __syncthreads();
  const int r = threadIdx.x;  // thread r quantizes row r: its 64 values are k-step `slab` of the down input
  if (r < TILE && (r & ~15) < cnt) {
    uint8_t codes[64], sc[4];
    for (int j = 0; j < 4; ++j) quantize_block16(&act_s[r][16 * j], G_ACT, codes + 16 * j, sc + j);
    pack_act_row(codes, sc, 64, r & 15, out_tiles + act_off(item, r >> 4, I) + size_t(slab) * A_KSTEP_U32);
  }
}

// Down GEMM: CTA = (item, 64 output columns), warp w covers 16 of them; bf16 output per pair (member position).
template <int SA, int SB>
__global__ void __launch_bounds__(128, 4) gemm_down(const uint32_t* __restrict__ tiles, const uint32_t* __restrict__ W,
                                                 int N, int K, const int* __restrict__ items, int n_items,
                                                 __nv_bfloat16* __restrict__ out, int a_pat) {
  const int slabs = N / 64, item = blockIdx.x / slabs, slab = blockIdx.x - item * slabs;
  if (item >= n_items) return;
  const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nt0 = (slab * 64 + warp * 16) / 8, subs = (cnt + 15) >> 4;
  const uint32_t* We = W + size_t(e) * (N / 8) * (K / 64) * W_TILE_U32;
  float acc[SUBS][2][4] = {};
  accumulate<SA, SB>(acc, tiles, item, K, We, nt0, cnt, lane, a_pat);
  const float gs = G_ACT * G_W;
#pragma unroll
  for (int s = 0; s < SUBS; ++s)
    if (s < subs)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = s * 16 + g + 8 * hh;
          if (r >= cnt) continue;
          const size_t o = size_t(first + r) * N + (nt0 + j) * 8 + 2 * t;
          reinterpret_cast<__nv_bfloat162*>(out)[o / 2] =
              __floats2bfloat162_rn(acc[s][j][2 * hh] * gs, acc[s][j][2 * hh + 1] * gs);
        }
}

// The selectors must be immediates, so each (a_sel, b_sel) pair is its own instantiation.
#define OJ_SCALE_SWITCH(m, CALL)            \
  switch ((m).a_sel * 4 + (m).b_sel) {      \
    case 0: CALL(0, 0); break;              \
    case 1: CALL(0, 1); break;              \
    case 2: CALL(0, 2); break;              \
    case 3: CALL(0, 3); break;              \
    case 4: CALL(1, 0); break;              \
    case 5: CALL(1, 1); break;              \
    case 6: CALL(1, 2); break;              \
    case 7: CALL(1, 3); break;              \
    default: fprintf(stderr, "no kernel for scale map a_sel %d b_sel %d\n", (m).a_sel, (m).b_sel); abort(); \
  }

static void launch_up(const ScaleMap& m, bool write_h, int n_items, const uint32_t* tiles, const uint32_t* W, int D,
                      int I, const int* items, uint32_t* out_tiles, float* h) {
  const int grid = n_items * (I / 64);
#define OJ_UP(sa, sb)                                                                                       \
  if (write_h) gemm_up_swiglu<sa, sb, true><<<grid, 256>>>(tiles, W, D, I, items, n_items, out_tiles, h, m.a_pat); \
  else gemm_up_swiglu<sa, sb, false><<<grid, 256>>>(tiles, W, D, I, items, n_items, out_tiles, h, m.a_pat)
  OJ_SCALE_SWITCH(m, OJ_UP)
#undef OJ_UP
}

static void launch_down(const ScaleMap& m, int n_items, const uint32_t* tiles, const uint32_t* W, int D, int I,
                        const int* items, __nv_bfloat16* out) {
  const int grid = n_items * (D / 64);
#define OJ_DOWN(sa, sb) gemm_down<sa, sb><<<grid, 128>>>(tiles, W, D, I, items, n_items, out, m.a_pat)
  OJ_SCALE_SWITCH(m, OJ_DOWN)
#undef OJ_DOWN
}

// ---- Host side
struct Weights {
  std::vector<uint32_t> up, dn;  // packed, all experts (host copy for the CPU reference)
  uint32_t *d_up = nullptr, *d_dn = nullptr;
  size_t upw = 0, dnw = 0;       // words per expert
};

static Weights make_weights(int E, int D, int I, uint32_t seed) {
  Weights w;
  std::mt19937 rng(seed);
  w.upw = size_t(2 * I / 8) * (D / 64) * W_TILE_U32;
  w.dnw = size_t(D / 8) * (I / 64) * W_TILE_U32;
  w.up.resize(E * w.upw);
  w.dn.resize(E * w.dnw);
  for (int e = 0; e < E; ++e) {
    random_packed_weight(rng, 2 * I, D, 0.05f, &w.up[e * w.upw]);
    random_packed_weight(rng, D, I, 0.05f, &w.dn[e * w.dnw]);
  }
  CK(cudaMalloc(&w.d_up, w.up.size() * 4));
  CK(cudaMalloc(&w.d_dn, w.dn.size() * 4));
  CK(cudaMemcpy(w.d_up, w.up.data(), w.up.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(w.d_dn, w.dn.data(), w.dn.size() * 4, cudaMemcpyHostToDevice));
  return w;
}

struct Cell {
  int rows = 0, n_items = 0;
  Plan plan;
  int *d_items = nullptr, *d_members = nullptr;
  __nv_bfloat16* d_x = nullptr;
  uint32_t *d_t1 = nullptr, *d_t2 = nullptr;
  float* d_h = nullptr;
  __nv_bfloat16* d_y = nullptr;
  std::vector<__nv_bfloat16> x;
};

static Cell make_cell(int rows, const std::vector<int>& picks, const std::vector<__nv_bfloat16>& x, int D, int I,
                      int E, int topk, bool need_h = false) {
  Cell c;
  c.rows = rows;
  c.x = x;
  c.plan = make_plan(picks, rows, topk, E, TILE);
  c.n_items = int(c.plan.items.size() / 3);
  const int pairs = rows * topk;
  CK(cudaMalloc(&c.d_items, c.plan.items.size() * 4));
  CK(cudaMalloc(&c.d_members, c.plan.members.size() * 4));
  CK(cudaMemcpy(c.d_items, c.plan.items.data(), c.plan.items.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(c.d_members, c.plan.members.data(), c.plan.members.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&c.d_x, c.x.size() * 2));
  CK(cudaMemcpy(c.d_x, c.x.data(), c.x.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&c.d_t1, act_off(c.n_items, 0, D) * 4));
  CK(cudaMalloc(&c.d_t2, act_off(c.n_items, 0, I) * 4));
  if (need_h) CK(cudaMalloc(&c.d_h, size_t(pairs) * 2 * I * 4));  // check runs only
  CK(cudaMalloc(&c.d_y, size_t(pairs) * D * 2));
  return c;
}

static std::vector<__nv_bfloat16> random_x(int rows, int D, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__nv_bfloat16> x(size_t(rows) * D);
  for (auto& v : x) v = __float2bfloat16(nd(rng));
  return x;
}

static Cell make_random_cell(int rows, int D, int I, int E, int topk, uint32_t seed, bool need_h = false) {
  return make_cell(rows, random_picks(rows, topk, E, seed), random_x(rows, D, seed + 1), D, I, E, topk, need_h);
}

static void free_cell(Cell& c) {
  cudaFree(c.d_items); cudaFree(c.d_members); cudaFree(c.d_x); cudaFree(c.d_t1); cudaFree(c.d_t2);
  cudaFree(c.d_h); cudaFree(c.d_y);
}

// The three kernels; ev (optional, 4 events) marks the boundaries for the per-kernel breakdown. The check runs
// set write_h, which also stores the up GEMM's fp32 gate/up output for the CPU reference.
static void run_pipeline(const Cell& c, const Weights& w, const ScaleMap& m, int D, int I, int topk,
                         cudaEvent_t* ev = nullptr, bool write_h = false) {
  if (ev) cudaEventRecord(ev[0]);
  quant_gather<<<c.n_items * (TILE / 4), 128>>>(c.d_x, topk, D, c.d_items, c.n_items, c.d_members, c.d_t1);
  if (ev) cudaEventRecord(ev[1]);
  launch_up(m, write_h, c.n_items, c.d_t1, w.d_up, D, I, c.d_items, c.d_t2, c.d_h);
  if (ev) cudaEventRecord(ev[2]);
  launch_down(m, c.n_items, c.d_t2, w.d_dn, D, I, c.d_items, c.d_y);
  if (ev) cudaEventRecord(ev[3]);
}

// Dot of dequantized activations with row n of a packed [N, K] weight; mag = sum |products|.
static double ref_dot(const std::vector<float>& a, const uint32_t* packed, int K, int n, double* mag) {
  double s = 0, m = 0;
  for (int k = 0; k < K; ++k) {
    const double p = double(a[k]) * dequant(weight_code(packed, K, n, k), weight_scale(packed, K, n, k / 16), G_W);
    s += p;
    m += std::fabs(p);
  }
  *mag = m;
  return s;
}

static void quant_dequant_row(const float* v, int K, std::vector<float>& out) {
  out.resize(K);
  for (int b = 0; b < K / 16; ++b) {
    uint8_t q[16], s;
    quantize_block16(v + 16 * b, G_ACT, q, &s);
    for (int i = 0; i < 16; ++i) out[16 * b + i] = dequant(q[i], s, G_ACT);
  }
}

// CPU reference of both GEMMs on sampled pairs and columns. Up: from the device's own quantized input (the
// quantizer is shared code). Down: from the device's own h, through SwiGLU + quantize on the CPU. Every pair of
// every item is sampled at small R; at large R every (pairs/3000)th member, which still covers all 128 tile rows. The up
// weights are packed in gate_up_row order, so natural column n lives at packed column gate_up_col(n).
static bool check_gemms(const Cell& c, const Weights& w, int D, int I, int topk, double* rel_up, double* rel_dn) {
  const int pairs = c.rows * topk, Nu = 2 * I;
  std::vector<float> h(size_t(pairs) * Nu);
  std::vector<__nv_bfloat16> y(size_t(pairs) * D);
  CK(cudaMemcpy(h.data(), c.d_h, h.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y.data(), c.d_y, y.size() * 2, cudaMemcpyDeviceToHost));
  *rel_up = *rel_dn = 0;
  const int stride = std::max(1, pairs / 3000);  // ~3,000 sampled pairs per cell keeps the CPU side to seconds
  std::vector<float> xin(D), a, act(I), ad;
  for (int it = 0; it < c.n_items; ++it) {
    const int e = c.plan.items[3 * it], first = c.plan.items[3 * it + 1], cnt = c.plan.items[3 * it + 2];
    for (int r = 0; r < cnt; ++r) {
      const int m = first + r;
      if (m % stride) continue;
      const int tok = c.plan.members[m] / topk;
      for (int k = 0; k < D; ++k) xin[k] = __bfloat162float(c.x[size_t(tok) * D + k]);
      quant_dequant_row(xin.data(), D, a);
      for (int n = 0; n < Nu; n += 37) {
        double mag;
        const double want = ref_dot(a, &w.up[e * w.upw], D, gate_up_col(n, I), &mag);
        *rel_up = std::max(*rel_up, std::fabs(h[size_t(m) * Nu + n] - want) / (mag + 1e-6));
      }
      for (int i = 0; i < I; ++i) {
        const float gte = h[size_t(m) * Nu + i], up = h[size_t(m) * Nu + I + i];
        act[i] = gte / (1.f + std::exp(-gte)) * up;
      }
      quant_dequant_row(act.data(), I, ad);
      for (int n = 0; n < D; n += 37) {
        double mag;
        const double want = ref_dot(ad, &w.dn[e * w.dnw], I, n, &mag);
        *rel_dn = std::max(*rel_dn, std::fabs(__bfloat162float(y[size_t(m) * D + n]) - want) / (mag + 1e-6));
      }
    }
  }
  // Up: exact products, tensor-core fp32-ish accumulation. Down adds bf16 output rounding (2^-9) and rare
  // quantization-boundary flips from __expf vs expf. A layout mistake gives errors of order 1.
  return *rel_up < 1e-3 && *rel_dn < 3e-2;
}

constexpr int INVARIANCE_WINDOWS[] = {1, 2, 3, 7, 9, 16, 17, 31, 32, 33, 64, 65, 80, 96, 100, 127, 128, 129, 200};

// Row invariance: one token (same values, same experts) placed last in windows of W rows that all pick its
// experts, so it sits at tile position (W-1) % 128: every sub-tile, and past 128 the second item of each expert.
// Its outputs must equal the solo (W=1) run bit for bit. The other rows are fresh random tokens per window.
static bool check_row_invariance(const Weights& w, const ScaleMap& m, int D, int I, int E, int topk, int* bad_w) {
  const std::vector<int> ref_picks = random_picks(1, topk, E, 777);
  const std::vector<__nv_bfloat16> ref_x = random_x(1, D, 778);
  std::vector<__nv_bfloat16> solo;
  for (int W : INVARIANCE_WINDOWS) {
    std::vector<__nv_bfloat16> x = random_x(W, D, 1000 + W);
    std::copy(ref_x.begin(), ref_x.end(), x.begin() + size_t(W - 1) * D);
    Cell c = make_cell(W, invariance_picks(W, ref_picks), x, D, I, E, topk);
    run_pipeline(c, w, m, D, I, topk);
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> y(size_t(W) * topk * D), mine(size_t(topk) * D);
    CK(cudaMemcpy(y.data(), c.d_y, y.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < c.plan.members.size(); ++k) {
      const int pair = c.plan.members[k];
      if (pair / topk == W - 1) memcpy(&mine[size_t(pair % topk) * D], &y[k * D], size_t(D) * 2);
    }
    free_cell(c);
    if (solo.empty()) solo = mine;
    else if (memcmp(solo.data(), mine.data(), solo.size() * 2) != 0) { *bad_w = W; return false; }
  }
  return true;
}

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
  return v;
}

static float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  int D = 2560, I = 640, E = 512, topk = 10;
  bool check = false;
  std::vector<int> rows = {1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--hidden") D = atoi(argv[++i]);
    else if (a == "--inter") I = atoi(argv[++i]);
    else if (a == "--experts") E = atoi(argv[++i]);
    else if (a == "--topk") topk = atoi(argv[++i]);
    else if (a == "--rows") rows = parse_list(argv[++i]);
    else if (a == "--check") check = true;
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (D % 64 || I % 64) { fprintf(stderr, "hidden and inter must be multiples of 64\n"); return 2; }

  // Scale-lane mapping first: timing doesn't depend on it, correctness does.
  const ScaleMap m = discover_scale_map();
  print_scale_map(m);

  fprintf(stderr, "generating %d experts...\n", E);
  Weights w = make_weights(E, D, I, 42);
  if (check) {
    for (int r : {1, 3, 64, 2048, 8192}) {  // 8192: items hold ~160 pairs, so all 8 sub-tiles are exercised
      Cell c = make_random_cell(r, D, I, E, topk, 1234, /*need_h=*/true);
      run_pipeline(c, w, m, D, I, topk, nullptr, /*write_h=*/true);
      CK(cudaDeviceSynchronize());
      double up, dn;
      const bool ok = check_gemms(c, w, D, I, topk, &up, &dn);
      printf("{\"check\":\"gemm\",\"rows\":%d,\"ok\":%s,\"max_rel_up\":%.3g,\"max_rel_down\":%.3g}\n", r,
             ok ? "true" : "false", up, dn);
      fflush(stdout);
      free_cell(c);
    }
    int bad_w = 0;
    const bool inv = check_row_invariance(w, m, D, I, E, topk, &bad_w);
    printf("{\"check\":\"row_invariance\",\"ok\":%s,\"windows\":\"%d..%d\",\"first_bad_window\":%d}\n",
           inv ? "true" : "false", INVARIANCE_WINDOWS[0], *std::max_element(std::begin(INVARIANCE_WINDOWS),
           std::end(INVARIANCE_WINDOWS)), bad_w);
    fflush(stdout);
  }
  cudaEvent_t t0, t1, ev[4];
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  for (auto& e : ev) cudaEventCreate(&e);
  for (int r : rows) {
    Cell c = make_random_cell(r, D, I, E, topk, 1234);
    for (int i = 0; i < 5; ++i) run_pipeline(c, w, m, D, I, topk);
    CK(cudaDeviceSynchronize());
    std::vector<float> ms(20);
    for (auto& t : ms) {
      cudaEventRecord(t0);
      run_pipeline(c, w, m, D, I, topk);
      cudaEventRecord(t1);
      CK(cudaEventSynchronize(t1));
      cudaEventElapsedTime(&t, t0, t1);
    }
    std::vector<float> part[3];
    for (int i = 0; i < 20; ++i) {  // separate runs so the extra events don't touch the totals above
      run_pipeline(c, w, m, D, I, topk, ev);
      CK(cudaEventSynchronize(ev[3]));
      for (int k = 0; k < 3; ++k) {
        float t;
        cudaEventElapsedTime(&t, ev[k], ev[k + 1]);
        part[k].push_back(t);
      }
    }
    std::sort(ms.begin(), ms.end());
    const double flops = double(r) * topk * 2.0 * (2.0 * I * D + double(D) * I);
    printf("{\"bench\":\"fp4\",\"rows\":%d,\"ms\":%.4f,\"ms_min\":%.4f,\"ms_max\":%.4f,\"flops\":%.6g}\n", r, ms[10],
           ms[0], ms[19], flops);
    // SwiGLU is fused into the up GEMM now; the field stays for the table's shape.
    printf("{\"bench\":\"fp4_parts\",\"rows\":%d,\"quant\":%.4f,\"up\":%.4f,\"swiglu\":0,\"down\":%.4f}\n", r,
           median(part[0]), median(part[1]), median(part[2]));
    fflush(stdout);
    free_cell(c);
  }
  return 0;
}
