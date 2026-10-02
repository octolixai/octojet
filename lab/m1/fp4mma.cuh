// Block-scaled FP4 MMA (NVFP4: E2M1 x E2M1, UE4M3 scale per 16) and runtime discovery of its scale-lane mapping.
// Shared by probe.cu and fp4_bench.cu so both use the same instruction and the same discovery.
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include <cmath>
#include <random>
#include <vector>

#include "ref.h"

// SA / SB are the mma's thread-id selectors for the A and B scale operands; they must be immediates.
template <int SA, int SB>
__device__ __forceinline__ void mma_fp4(float* d, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, %12}, %11, {0, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "n"(SA), "n"(SB));
}

// The mapping in use: A scale selector + lane pairing, B scale selector. found = every discovery step matched.
struct ScaleMap {
  int a_sel = 0, a_pat = 0, b_sel = 0;
  bool regs_ok = false, a_found = false, b_found = false, combined_ok = false;
  bool found() const { return regs_ok && a_found && b_found && combined_ok; }
};

// One 16x64 by 64x8 product with explicit per-lane scale words.
template <int SA, int SB>
__global__ void fp4_layout_kernel(const uint32_t* atile, const uint32_t* wtile, const uint32_t* lane_sa,
                                  const uint32_t* lane_sb, float* out) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = atile[4 * lane + i];
  b[0] = wtile[2 * lane];
  b[1] = wtile[2 * lane + 1];
  float d[4] = {0, 0, 0, 0};
  mma_fp4<SA, SB>(d, a, b, lane_sa[lane], lane_sb[lane]);
  out[g * 8 + 2 * t] = d[0];
  out[g * 8 + 2 * t + 1] = d[1];
  out[(g + 8) * 8 + 2 * t] = d[2];
  out[(g + 8) * 8 + 2 * t + 1] = d[3];
}

namespace fp4layout {

struct Case {
  uint8_t ac[16][64], wc[8][64];
  uint8_t as[16][4], ws[8][4];
};

inline void want(const Case& c, float* out) {
  for (int r = 0; r < 16; ++r)
    for (int n = 0; n < 8; ++n) {
      double s = 0;
      for (int k = 0; k < 64; ++k)
        s += double(dequant(c.ac[r][k], c.as[r][k / 16], 1.f)) * dequant(c.wc[n][k], c.ws[n][k / 16], 1.f);
      out[r * 8 + n] = float(s);
    }
}

inline int mismatches(const float* got, const float* ref) {
  int bad = 0;
  for (int i = 0; i < 128; ++i)
    if (!(std::fabs(got[i] - ref[i]) <= 1e-3f * (1.f + std::fabs(ref[i])))) ++bad;
  return bad;
}

// Runs the product with the given selectors and lane words; returns the number of wrong outputs (-1 on CUDA error).
inline int run(const Case& c, int sa_sel, int sb_sel, const uint32_t* lsa, const uint32_t* lsb) {
  std::vector<uint32_t> atile(A_KSTEP_U32, 0), wtile(W_TILE_U32, 0);
  for (int r = 0; r < 16; ++r) pack_act_row(c.ac[r], c.as[r], 64, r, atile.data());
  pack_weight(&c.wc[0][0], &c.ws[0][0], 8, 64, wtile.data());
  uint32_t *da, *dw, *dsa, *dsb;
  float* dout;
  if (cudaMalloc(&da, A_KSTEP_U32 * 4) || cudaMalloc(&dw, W_TILE_U32 * 4) || cudaMalloc(&dsa, 128) ||
      cudaMalloc(&dsb, 128) || cudaMalloc(&dout, 512))
    return -1;
  cudaMemcpy(da, atile.data(), A_KSTEP_U32 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, wtile.data(), W_TILE_U32 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsa, lsa, 128, cudaMemcpyHostToDevice);
  cudaMemcpy(dsb, lsb, 128, cudaMemcpyHostToDevice);
  switch (sa_sel * 4 + sb_sel) {
    case 0: fp4_layout_kernel<0, 0><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 1: fp4_layout_kernel<0, 1><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 2: fp4_layout_kernel<0, 2><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 3: fp4_layout_kernel<0, 3><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 4: fp4_layout_kernel<1, 0><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 5: fp4_layout_kernel<1, 1><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 6: fp4_layout_kernel<1, 2><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    case 7: fp4_layout_kernel<1, 3><<<1, 32>>>(da, dw, dsa, dsb, dout); break;
    default: return -1;
  }
  float got[128], ref[128];
  const bool ok = cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(got, dout, 512, cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(da); cudaFree(dw); cudaFree(dsa); cudaFree(dsb); cudaFree(dout);
  if (!ok) return -1;
  want(c, ref);
  return mismatches(got, ref);
}

inline uint32_t sw(const uint8_t* s) { return pack_scales4(s); }

}  // namespace fp4layout

// Discover the scale-lane mapping in three steps, each changing one thing:
//   1. all scales 1.0 in every lane: checks the A/B/C register layout alone;
//   2. distinct A scales per (row, block), B scales 1.0 everywhere: tries every A candidate;
//   3. A scales 1.0 everywhere, distinct B scales per (col, block): tries every B selector;
// then the two winners together. Scales are powers of two from 0.125 to 8, so any mix-up changes the product.
inline ScaleMap discover_scale_map() {
  using namespace fp4layout;
  ScaleMap m;
  Case c;
  std::mt19937 rng(11);
  for (auto& row : c.ac) for (auto& v : row) v = rng() & 15;
  for (auto& row : c.wc) for (auto& v : row) v = rng() & 15;
  const uint8_t one = 0x38, pw[7] = {0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50};
  uint8_t as_d[16][4], ws_d[8][4];
  for (int r = 0; r < 16; ++r) for (int j = 0; j < 4; ++j) as_d[r][j] = pw[(r + 2 * j) % 7];
  for (int n = 0; n < 8; ++n) for (int j = 0; j < 4; ++j) ws_d[n][j] = pw[(3 * n + j) % 7];
  uint32_t ones[32], lsa[32], lsb[32];
  for (auto& v : ones) v = 0x38383838u;

  for (auto& r : c.as) for (auto& v : r) v = one;  // step 1
  for (auto& r : c.ws) for (auto& v : r) v = one;
  m.regs_ok = run(c, 0, 0, ones, ones) == 0;

  memcpy(c.as, as_d, sizeof as_d);  // step 2
  for (int k = 0; k < N_SFA_CANDIDATES && !m.a_found; ++k) {
    const SfaCandidate cand = SFA_CANDIDATES[k];
    for (int lane = 0; lane < 32; ++lane) {
      const int r = sfa_row(lane, cand.sel, cand.pat);
      lsa[lane] = r >= 0 ? sw(c.as[r]) : 0u;
    }
    if (run(c, cand.sel, 0, lsa, ones) == 0) { m.a_found = true; m.a_sel = cand.sel; m.a_pat = cand.pat; }
  }

  for (auto& r : c.as) for (auto& v : r) v = one;  // step 3
  memcpy(c.ws, ws_d, sizeof ws_d);
  for (int sel = 0; sel < N_SFB_CANDIDATES && !m.b_found; ++sel) {
    for (int lane = 0; lane < 32; ++lane) {
      const int n = sfb_col(lane, sel);
      lsb[lane] = n >= 0 ? sw(c.ws[n]) : 0u;
    }
    if (run(c, 0, sel, ones, lsb) == 0) { m.b_found = true; m.b_sel = sel; }
  }

  if (m.a_found && m.b_found) {  // both together
    memcpy(c.as, as_d, sizeof as_d);
    for (int lane = 0; lane < 32; ++lane) {
      const int r = sfa_row(lane, m.a_sel, m.a_pat), n = sfb_col(lane, m.b_sel);
      lsa[lane] = r >= 0 ? sw(c.as[r]) : 0u;
      lsb[lane] = n >= 0 ? sw(c.ws[n]) : 0u;
    }
    m.combined_ok = run(c, m.a_sel, m.b_sel, lsa, lsb) == 0;
  }
  return m;
}

inline void print_scale_map(const ScaleMap& m) {
  printf("{\"check\":\"layout\",\"ok\":%s,\"regs_ok\":%s,\"a_found\":%s,\"b_found\":%s,\"combined_ok\":%s,"
         "\"a_sel\":%d,\"a_pat\":%d,\"b_sel\":%d}\n",
         m.found() ? "true" : "false", m.regs_ok ? "true" : "false", m.a_found ? "true" : "false",
         m.b_found ? "true" : "false", m.combined_ok ? "true" : "false", m.a_sel, m.a_pat, m.b_sel);
  fflush(stdout);
}
