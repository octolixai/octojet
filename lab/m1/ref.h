// Octojet M1 reference code: NVFP4 codecs, block quantizer, fragment packers, routing plan, CPU GEMM.
// Compiled by nvcc (host + device) and by clang++ for the CPU tests, so both run the same arithmetic.
#pragma once

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#ifdef __CUDACC__
#define OJ_HD __host__ __device__
#else
#define OJ_HD
#endif

// ---- E2M1: sign bit 3, exponent bits 2..1 (bias 1), mantissa bit 0. Values 0, .5, 1, 1.5, 2, 3, 4, 6.
OJ_HD inline float e2m1_decode(uint8_t code) {
  const int mag = code & 7;
  float v = (mag < 2) ? 0.5f * mag : float(1 << ((mag >> 1) - 1)) * (1.f + 0.5f * (mag & 1));
  return (code & 8) ? -v : v;
}

// Round to nearest, ties to the even code (the code with mantissa bit 0), saturate at 6. Branchless: each
// comparison adds one code step; `>` vs `>=` at a midpoint picks the even neighbour. NaN compares false -> 0.
OJ_HD inline uint8_t e2m1_encode(float x) {
  const uint8_t sign = std::signbit(x) && !(x != x) ? 8 : 0;
  const float a = std::fabs(x);
  const int c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
  return uint8_t(sign | c);
}

// ---- UE4M3: unsigned, exponent bits 6..3 (bias 7), mantissa bits 2..0; 0x7F is NaN (never produced).
OJ_HD inline float ue4m3_decode(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  if (e == 0) return std::ldexp(float(m), -9);          // subnormal: m * 2^-9
  return std::ldexp(1.f + m / 8.f, e - 7);
}

// Round to nearest even over the finite codes 0x00..0x7E; saturate at 448. Closed form: subnormals are m * 2^-9;
// normals take the binade from frexp and round the 3-bit mantissa with nearbyint (ties to even, the default mode);
// a mantissa that rounds up to 8 carries into the next binade, which is the same code arithmetic.
OJ_HD inline uint8_t ue4m3_encode(float x) {
  if (!(x > 0.f)) return 0;
  if (x >= 448.f) return 0x7E;
  if (x < 0.015625f) return uint8_t(std::nearbyint(x * 512.f));  // 8 -> code 0x08, the smallest normal
  int e2;
  const float f = std::frexp(x, &e2);                             // x = f * 2^e2, f in [0.5, 1)
  const int m = int(std::nearbyint((f * 2.f - 1.f) * 8.f));       // 0..8
  const int code = ((e2 - 1 + 7) << 3) + m;
  return uint8_t(code > 0x7E ? 0x7E : code);
}

// ---- NVFP4 block of 16 (modelopt semantics): s = ue4m3(amax / 6 / g), q = e2m1(x / (s * g)).
OJ_HD inline void quantize_block16(const float* x, float global_scale, uint8_t* codes16, uint8_t* scale) {
  float amax = 0.f;
  for (int i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(x[i]));
  const uint8_t s = ue4m3_encode(amax / 6.f / global_scale);
  *scale = s;
  const float denom = ue4m3_decode(s) * global_scale;
  for (int i = 0; i < 16; ++i) codes16[i] = denom > 0.f ? e2m1_encode(x[i] / denom) : uint8_t(0);
}

OJ_HD inline float dequant(uint8_t code, uint8_t scale, float global_scale) {
  return e2m1_decode(code) * ue4m3_decode(scale) * global_scale;
}

// ---- Fragment packers for mma.m16n8k64 kind::mxf4nvf4 scale_vec::4X (layout documented in the M1 plan, Task 2).

constexpr int W_TILE_U32 = 72;    // 32 lanes x (b0, b1) + 8 scale words
constexpr int A_KSTEP_U32 = 144;  // 32 lanes x (a0..a3) + 16 scale words

// Which row / col a lane's scale word serves. The PTX ISA's exact lane assignment for scale_vec::4X is not pinned
// down here, so it is a runtime choice: `sel` is the mma's thread-id selector (an immediate), `pat` how the two
// A-scale lanes of a quad pair up ({2sel, 2sel+1} or {sel, sel+2}). probe.cu / fp4_bench.cu discover the right one.
struct SfaCandidate { int sel, pat; };
constexpr SfaCandidate SFA_CANDIDATES[] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
constexpr int N_SFA_CANDIDATES = 4;
constexpr int N_SFB_CANDIDATES = 4;  // sfb selector 0..3

OJ_HD inline int sfa_row(int lane, int sel, int pat) {
  const int g = lane >> 2, t = lane & 3;
  const int lo = pat == 0 ? 2 * sel : sel, hi = pat == 0 ? 2 * sel + 1 : sel + 2;
  return t == lo ? g : t == hi ? g + 8 : -1;
}
OJ_HD inline int sfb_col(int lane, int sel) { return (lane & 3) == sel ? (lane >> 2) : -1; }

OJ_HD inline uint32_t pack8(const uint8_t* c) {
  uint32_t w = 0;
  for (int i = 0; i < 8; ++i) w |= uint32_t(c[i] & 15) << (4 * i);
  return w;
}

OJ_HD inline uint32_t pack_scales4(const uint8_t* s) {
  return uint32_t(s[0]) | uint32_t(s[1]) << 8 | uint32_t(s[2]) << 16 | uint32_t(s[3]) << 24;
}

// codes [N][K] one per byte, scales [N][K/16] -> tiles [N/8][K/64] of W_TILE_U32 words.
inline void pack_weight(const uint8_t* codes, const uint8_t* scales, int N, int K, uint32_t* out) {
  const int KS = K / 64;
  for (int nt = 0; nt < N / 8; ++nt)
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t* tile = out + (size_t(nt) * KS + ks) * W_TILE_U32;
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3, n = nt * 8 + g;
        const uint8_t* row = codes + size_t(n) * K + ks * 64;
        tile[2 * lane + 0] = pack8(row + 8 * t);
        tile[2 * lane + 1] = pack8(row + 32 + 8 * t);
      }
      for (int g = 0; g < 8; ++g) tile[64 + g] = pack_scales4(scales + size_t(nt * 8 + g) * (K / 16) + ks * 4);
    }
}

inline void unpack_weight(const uint32_t* in, int N, int K, uint8_t* codes, uint8_t* scales) {
  const int KS = K / 64;
  for (int nt = 0; nt < N / 8; ++nt)
    for (int ks = 0; ks < KS; ++ks) {
      const uint32_t* tile = in + (size_t(nt) * KS + ks) * W_TILE_U32;
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3, n = nt * 8 + g;
        uint8_t* row = codes + size_t(n) * K + ks * 64;
        for (int i = 0; i < 8; ++i) {
          row[8 * t + i] = (tile[2 * lane] >> (4 * i)) & 15;
          row[32 + 8 * t + i] = (tile[2 * lane + 1] >> (4 * i)) & 15;
        }
      }
      for (int g = 0; g < 8; ++g)
        for (int j = 0; j < 4; ++j) scales[size_t(nt * 8 + g) * (K / 16) + ks * 4 + j] = (tile[64 + g] >> (8 * j)) & 255;
    }
}

// Read one code / scale of a packed [N, K] weight (inverse of pack_weight for a single element).
inline uint8_t weight_code(const uint32_t* packed, int K, int n, int k) {
  const int KS = K / 64, ks = k / 64, kk = k % 64, g = n % 8;
  const int reg = kk >= 32, t = (kk % 32) / 8, i = kk % 8;
  const uint32_t* tile = packed + (size_t(n / 8) * KS + ks) * W_TILE_U32;
  return (tile[2 * (4 * g + t) + reg] >> (4 * i)) & 15;
}
inline uint8_t weight_scale(const uint32_t* packed, int K, int n, int block) {
  const int KS = K / 64;
  const uint32_t* tile = packed + (size_t(n / 8) * KS + block / 4) * W_TILE_U32;
  return (tile[64 + n % 8] >> (8 * (block % 4))) & 255;
}

// Random packed weight without materialising float values: codes uniform over the 16 E2M1 codes, one scale per
// block of 16 from amax ~ |N(0, sigma)| * 2.5 (about what a block of 16 normal draws gives). Values don't affect timing.
inline void random_packed_weight(std::mt19937& rng, int N, int K, float sigma, uint32_t* out) {
  std::normal_distribution<float> nd(0.f, sigma);
  const int KS = K / 64;
  for (int nt = 0; nt < N / 8; ++nt)
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t* tile = out + (size_t(nt) * KS + ks) * W_TILE_U32;
      for (int i = 0; i < 64; ++i) tile[i] = uint32_t(rng());
      for (int g = 0; g < 8; ++g) {
        uint8_t sc[4];
        for (int j = 0; j < 4; ++j) {
          const uint8_t s = ue4m3_encode(std::fabs(nd(rng)) * 2.5f / 6.f);
          sc[j] = s ? s : uint8_t(1);
        }
        tile[64 + g] = pack_scales4(sc);
      }
    }
}

// Row r (0..15) of one activation tile: codes [K] one per byte, scales [K/16].
OJ_HD inline void pack_act_row(const uint8_t* codes, const uint8_t* scales, int K, int r, uint32_t* tile) {
  const int g = r & 7, hi = r >> 3;
  for (int ks = 0; ks < K / 64; ++ks) {
    uint32_t* step = tile + size_t(ks) * A_KSTEP_U32;
    for (int t = 0; t < 4; ++t) {
      const int lane = 4 * g + t;
      step[4 * lane + hi + 0] = pack8(codes + ks * 64 + 8 * t);        // a0 (row g) or a1 (row g+8)
      step[4 * lane + hi + 2] = pack8(codes + ks * 64 + 32 + 8 * t);   // a2 or a3
    }
    step[128 + r] = pack_scales4(scales + ks * 4);
  }
}

// ---- Fused up GEMM column order: packed column block j (128 columns) holds gate rows 64j..64j+63 then up rows
// I+64j..I+64j+63, so one CTA sees matching gate and up columns and can apply SwiGLU in its epilogue.
OJ_HD inline int gate_up_row(int c, int I) {
  const int j = c / 128, w = c % 128;
  return w < 64 ? 64 * j + w : I + 64 * j + (w - 64);
}
OJ_HD inline int gate_up_col(int r, int I) {
  return r < I ? 128 * (r / 64) + r % 64 : 128 * ((r - I) / 64) + 64 + (r - I) % 64;
}

// ---- Routing plan: pairs (row * slots + slot) grouped by expert, cut into items of at most `tile` pairs.
struct Plan {
  int E = 0, slots = 0, tile = 0;
  std::vector<int> members;  // pair ids, grouped by expert ascending, pair id ascending within an expert
  std::vector<int> items;    // (expert, first, count) triples
};

inline Plan make_plan(const std::vector<int>& picks, int rows, int slots, int E, int tile) {
  Plan p;
  p.E = E; p.slots = slots; p.tile = tile;
  std::vector<std::vector<int>> by(E);
  for (int i = 0; i < rows * slots; ++i) by[picks[i]].push_back(i);
  for (int e = 0; e < E; ++e) {
    for (size_t j = 0; j < by[e].size(); j += tile) {
      const int cnt = int(std::min<size_t>(tile, by[e].size() - j));
      p.items.push_back(e);
      p.items.push_back(int(p.members.size()) + int(j));  // members grows after this loop
      p.items.push_back(cnt);
    }
    p.members.insert(p.members.end(), by[e].begin(), by[e].end());
  }
  return p;
}

inline std::vector<int> random_picks(int rows, int slots, int E, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<int> picks(size_t(rows) * slots);
  std::vector<int> ids(E);
  for (int r = 0; r < rows; ++r) {
    for (int e = 0; e < E; ++e) ids[e] = e;
    for (int s = 0; s < slots; ++s) {  // partial Fisher-Yates: distinct experts per row
      std::uniform_int_distribution<int> d(s, E - 1);
      std::swap(ids[s], ids[d(rng)]);
      picks[size_t(r) * slots + s] = ids[s];
    }
  }
  return picks;
}

// W rows that all pick `ref` (one row of picks): row W-1 then sits at position W-1 in each of its experts' lists.
inline std::vector<int> invariance_picks(int W, const std::vector<int>& ref) {
  std::vector<int> picks;
  for (int r = 0; r < W; ++r) picks.insert(picks.end(), ref.begin(), ref.end());
  return picks;
}

inline void ref_gemm(const float* A, const float* W, int M, int N, int K, float* C) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += double(A[size_t(m) * K + k]) * W[size_t(n) * K + k];
      C[size_t(m) * N + n] = float(acc);
    }
}
