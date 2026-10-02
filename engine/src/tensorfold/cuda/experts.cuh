// Shared by the grouped-expert kernels (experts.cu: decode form and plan; experts_prefill.cu: prefill form).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int NTW = 4;                       // n8 tiles a warp: 32 output columns
constexpr int COLS = 8 * NTW;
constexpr uint32_t LOW = 0x000F000Fu;
constexpr uint32_t K128 = 0x43004300u;       // the bf16 pair (128, 128)

// Weight formats: F 0 = MLX affine 4-bit (int4 code; bf16 scale and bias a column a group of GS inputs);
// F 1 = NVFP4 (E2M1 code; one UE4M3 scale a column a block of 16 inputs; an fp32 scale a matrix, applied to the
// accumulator). Both store a group's codes as one uint4 a lane: lane gq*4+t holds column nt*8+gq's inputs
// [8t, 8t+8) of the group in word nt. F 1 stores the scales as 16 words a group: word gq*2 + sub (sub = the
// 16-input half the lane's inputs fall in, t >> 1), byte nt.
template <int GS, int F = 0>
struct Geo {
  static_assert(F == 0 || GS == 32, "NVFP4 blocks are 32 inputs (two scale blocks of 16)");
  static constexpr int KS = GS / 16;             // mma k-steps a group
  static constexpr int WV = NTW * GS / 128;      // weight uint4 a lane a group
  static constexpr int XV = GS / 32;             // input uint4 a row a lane a group
  static constexpr int SBV = F ? 1 : NTW / 2;    // F 0: scale and bias uint4 a quad a group; F 1: scale uint4 a quad
  static constexpr int BLOCK = 32 * WV + 4 * SBV;
};

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// nibbles at bits sh and sh + 16 -> the exact bf16 pair (q0, q1): (128 + q) - 128
__device__ __forceinline__ uint32_t nib2(uint32_t w, int sh) {
  uint32_t v = ((w >> sh) & LOW) | K128;
  const uint32_t k = K128;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// Two E2M1 codes at bits [3:0] and [19:16] of (w >> sh) -> the exact bf16 pair. Magnitudes 0 .5 1 1.5 2 3 4 6 are
// bf16 0000 3F00 3F80 3FC0 4000 4040 4080 40C0: the low and high bytes come from two 8-entry byte tables (prmt),
// the sign bit moves from bit 3 to bit 15 of each half.
constexpr uint32_t E2M1_LO_A = 0xC0800000u, E2M1_LO_B = 0xC0804000u;   // low bytes of codes 0-3 and 4-7
constexpr uint32_t E2M1_HI_A = 0x3F3F3F00u, E2M1_HI_B = 0x40404040u;   // high bytes
__device__ __forceinline__ uint32_t e2m1x2(uint32_t w, int sh) {
  const uint32_t v = (w >> sh) & LOW;
  const uint32_t m0 = v & 7u, m1 = (v >> 16) & 7u;
  const uint32_t lo = __byte_perm(E2M1_LO_A, E2M1_LO_B, m0 | (m1 << 8)) & 0x00FF00FFu;
  const uint32_t hi = __byte_perm(E2M1_HI_A, E2M1_HI_B, (m0 << 4) | (m1 << 12)) & 0xFF00FF00u;
  return lo | hi | ((v & 0x00080008u) << 12);
}

// UE4M3 (bias 7, no sign) -> bf16 bits, exact: e > 0 gives exponent e + 120 and mantissa m << 4; e == 0 is m * 2^-9.
__device__ __forceinline__ uint32_t ue4m3_bf16(uint32_t b) {
  const uint32_t e = (b >> 3) & 15u, m = b & 7u;
  if (e) return ((e + 120u) << 7) | (m << 4);
  return m ? (__float_as_uint(__uint2float_rn(m) * 0.001953125f) >> 16) : 0u;
}

// The bf16 pair (s, s) of byte j of a lane's scale word.
__device__ __forceinline__ uint32_t scale_pair(uint32_t sw, int j) {
  const uint32_t s = ue4m3_bf16((sw >> (8 * j)) & 255u);
  return s | (s << 16);
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float lo_f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_f(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// SwiGLU as the families define it: bf16(bf16(silu(g)) * u) on bf16 g and u, clipped first when limit > 0
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

__device__ __forceinline__ float relu2(float a) {
  const float u = fmaxf(bf(a), 0.f);
  return u * u;
}

// EPI 0: fp32 out (down); 1: bf16 relu(bf16(acc))^2; 2: bf16 SwiGLU(matrix 0, matrix 1); 3: bf16 out (prefill down)
template <int EPI, int M, int RT>
__device__ __forceinline__ void epilogue(const float (&acc)[M][RT][NTW][4], int r, void* out, int N, int col0, int pr0,
                                         int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][r][j][2 * h], a1 = acc[0][r][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0, o1;
        if constexpr (EPI == 1) {
          o0 = relu2(a0);
          o1 = relu2(a1);
        } else if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][r][j][2 * h], limit);
          o1 = swiglu(a1, acc[M - 1][r][j][2 * h + 1], limit);
        } else {
          o0 = a0;
          o1 = a1;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

}  // namespace
