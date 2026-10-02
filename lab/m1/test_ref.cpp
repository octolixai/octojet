// CPU tests for ref.h. Build: clang++ -std=c++17 -O1 -Wall -Wextra -o /tmp/test_ref test_ref.cpp && /tmp/test_ref
#include "ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
      ++g_fail;                                                                \
    }                                                                          \
  } while (0)


// Original encoders (binary search / loop), kept here as the reference the closed-form ones must equal bit for bit.
static uint8_t e2m1_encode_ref(float x) {
  const uint8_t sign = (x < 0.f || (x == 0.f && std::signbit(x))) ? 8 : 0;
  float a = std::fabs(x);
  if (!(a == a)) return 0;
  const float mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
  uint8_t c = 7;
  for (int i = 0; i < 7; ++i) {
    if (a < mid[i]) { c = uint8_t(i); break; }
    if (a == mid[i]) { c = uint8_t((i & 1) ? i + 1 : i); break; }
  }
  return uint8_t(sign | c);
}
static uint8_t ue4m3_encode_ref(float x) {
  if (!(x > 0.f)) return 0;
  if (x >= 448.f) return 0x7E;
  int lo = 0, hi = 0x7E;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (ue4m3_decode(uint8_t(mid)) <= x) lo = mid; else hi = mid - 1;
  }
  if (lo == 0x7E) return 0x7E;
  const float a = ue4m3_decode(uint8_t(lo)), b = ue4m3_decode(uint8_t(lo + 1));
  if (x - a < b - x) return uint8_t(lo);
  if (x - a > b - x) return uint8_t(lo + 1);
  return uint8_t((lo & 1) ? lo + 1 : lo);
}

// Candidate inputs: every code value, every midpoint, one float either side of each, specials, and random floats.
static std::vector<float> encoder_inputs(float top, std::initializer_list<float> grid) {
  std::vector<float> v = {0.f, -0.f, 1e-30f, 1e-45f, INFINITY, -INFINITY, NAN, top, top * 2, 1e30f};
  std::vector<float> g(grid);
  for (size_t i = 0; i < g.size(); ++i) {
    const float pts[2] = {g[i], i + 1 < g.size() ? 0.5f * (g[i] + g[i + 1]) : g[i]};
    for (float p : pts)
      for (float q : {p, std::nextafter(p, 0.f), std::nextafter(p, INFINITY)}) { v.push_back(q); v.push_back(-q); }
  }
  std::mt19937 rng(21);
  std::uniform_real_distribution<float> u(0.f, top * 1.1f);
  std::uniform_int_distribution<uint32_t> bits(0, 0x7F7FFFFFu);
  for (int i = 0; i < 2000000; ++i) { v.push_back(u(rng)); uint32_t b = bits(rng); float f; memcpy(&f, &b, 4); v.push_back(f); }
  return v;
}

static void test_e2m1_encode_matches_reference() {
  auto in = encoder_inputs(6.f, {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f});
  int bad = 0;
  for (float x : in) bad += e2m1_encode(x) != e2m1_encode_ref(x);
  CHECK(bad == 0);
}

static void test_ue4m3_encode_matches_reference() {
  std::vector<float> grid;
  for (int b = 0; b <= 0x7E; ++b) grid.push_back(ue4m3_decode(uint8_t(b)));
  std::vector<float> in = encoder_inputs(448.f, {});
  for (size_t i = 0; i < grid.size(); ++i) {
    const float pts[2] = {grid[i], i + 1 < grid.size() ? 0.5f * (grid[i] + grid[i + 1]) : grid[i]};
    for (float p : pts)
      for (float q : {p, std::nextafter(p, 0.f), std::nextafter(p, INFINITY)}) in.push_back(q);
  }
  int bad = 0;
  float first = 0;
  for (float x : in)
    if (ue4m3_encode(x) != ue4m3_encode_ref(x)) { if (!bad) first = x; ++bad; }
  if (bad) std::printf("ue4m3 mismatch at %g: %d vs %d\n", first, ue4m3_encode(first), ue4m3_encode_ref(first));
  CHECK(bad == 0);
}

static void test_e2m1_table() {
  const float want[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  for (int c = 0; c < 8; ++c) {
    CHECK(e2m1_decode(uint8_t(c)) == want[c]);
    CHECK(e2m1_decode(uint8_t(c | 8)) == -want[c]);
    CHECK(e2m1_encode(want[c]) == c);
  }
}

static void test_e2m1_rounding() {
  CHECK(e2m1_encode(0.25f) == 0);   // tie 0 / 0.5 -> even code 0
  CHECK(e2m1_encode(0.75f) == 2);   // tie 0.5 / 1 -> even code 2 (1.0)
  CHECK(e2m1_encode(1.25f) == 2);   // tie 1 / 1.5 -> code 2
  CHECK(e2m1_encode(1.75f) == 4);   // tie 1.5 / 2 -> code 4
  CHECK(e2m1_encode(2.5f) == 4);    // tie 2 / 3 -> code 4
  CHECK(e2m1_encode(3.5f) == 6);    // tie 3 / 4 -> code 6
  CHECK(e2m1_encode(5.0f) == 6);    // tie 4 / 6 -> code 6
  CHECK(e2m1_encode(100.f) == 7);   // saturate
  CHECK(e2m1_encode(-100.f) == 15);
  CHECK(e2m1_encode(-0.6f) == 9);   // -0.5
}

static void test_ue4m3_roundtrip() {
  for (int b = 0; b < 0x7F; ++b) {  // every finite code decodes and re-encodes to itself
    float v = ue4m3_decode(uint8_t(b));
    CHECK(ue4m3_encode(v) == b);
  }
  CHECK(ue4m3_decode(0x08) == 0.015625f);  // 2^-6, smallest normal
  CHECK(ue4m3_decode(0x01) == 0.001953125f);  // 2^-9, smallest subnormal
  CHECK(ue4m3_decode(0x7E) == 448.f);
  CHECK(ue4m3_encode(1000.f) == 0x7E);  // saturate, never NaN
  CHECK(ue4m3_encode(0.f) == 0);
}

static void test_quantize_block() {
  float x[16];
  for (int i = 0; i < 16; ++i) x[i] = (i - 8) * 0.37f;
  uint8_t q[16], s;
  quantize_block16(x, 1.0f, q, &s);
  float amax = 8 * 0.37f;
  float sd = ue4m3_decode(s);
  CHECK(std::fabs(sd - amax / 6.f) <= amax / 6.f * 0.07f);  // UE4M3 has 3 mantissa bits
  for (int i = 0; i < 16; ++i) {
    float err = std::fabs(dequant(q[i], s, 1.0f) - x[i]);
    CHECK(err <= sd * 1.0f + 1e-6f);  // E2M1 step is at most 2 units of the scaled grid
  }
}

static void test_quantize_zero_block() {
  float x[16] = {0};
  uint8_t q[16], s = 99;
  quantize_block16(x, 1.0f, q, &s);
  CHECK(s == 0);
  for (int i = 0; i < 16; ++i) CHECK(q[i] == 0);
}

static void test_quantize_tiny_block() {
  float x[16];
  for (int i = 0; i < 16; ++i) x[i] = 1e-12f * (i + 1);  // scale underflows UE4M3 to 0
  uint8_t q[16], s = 99;
  quantize_block16(x, 1.0f, q, &s);
  for (int i = 0; i < 16; ++i) {
    float d = dequant(q[i], s, 1.0f);
    CHECK(std::isfinite(d));
  }
}

static void test_quantize_global_scale() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 3.f);
  float x[16];
  for (auto& v : x) v = nd(rng);
  uint8_t q1[16], s1, q2[16], s2;
  quantize_block16(x, 1.0f, q1, &s1);
  float g = 0.01f;  // same values with a small global scale keep the scale in range
  quantize_block16(x, g, q2, &s2);
  for (int i = 0; i < 16; ++i) {
    CHECK(std::isfinite(dequant(q2[i], s2, g)));
    CHECK(std::fabs(dequant(q2[i], s2, g) - x[i]) <= std::fabs(x[i]) * 0.5f + 1.0f);
  }
}

static void test_pack8() {
  uint8_t c[8] = {1, 2, 3, 4, 5, 6, 7, 15};
  CHECK(pack8(c) == 0xF7654321u);
}

static void test_weight_pack_roundtrip() {
  const int N = 16, K = 128;
  std::mt19937 rng(3);
  std::vector<uint8_t> codes(N * K), scales(N * K / 16), c2(N * K), s2(N * K / 16);
  for (auto& v : codes) v = rng() & 15;
  for (auto& v : scales) v = rng() % 0x7F;
  std::vector<uint32_t> packed(N / 8 * K / 64 * W_TILE_U32);
  pack_weight(codes.data(), scales.data(), N, K, packed.data());
  unpack_weight(packed.data(), N, K, c2.data(), s2.data());
  CHECK(codes == c2);
  CHECK(scales == s2);
}

static void test_weight_pack_layout() {
  // b0 of lane (g=2, t=1) in tile (n-tile 0, k-step 0) holds col 2, k 8..15.
  const int N = 8, K = 64;
  std::vector<uint8_t> codes(N * K, 0), scales(N * K / 16, 0);
  for (int k = 8; k < 16; ++k) codes[2 * K + k] = uint8_t(k - 7);  // 1..8
  std::vector<uint32_t> packed(W_TILE_U32);
  pack_weight(codes.data(), scales.data(), N, K, packed.data());
  const int lane = 2 * 4 + 1;
  CHECK(packed[2 * lane + 0] == 0x87654321u);
  CHECK(packed[2 * lane + 1] == 0u);
}

static void test_act_pack_layout() {
  // a1 of lane (g=3, t=0) holds row 11, k 0..7; a2 holds row 3, k 32..39.
  const int K = 64;
  std::vector<uint32_t> tile(A_KSTEP_U32, 0);
  uint8_t codes[64] = {0}, sc[4] = {0x38, 0x39, 0x3A, 0x3B};
  for (int k = 0; k < 8; ++k) codes[k] = uint8_t(k + 1);
  pack_act_row(codes, sc, K, 11, tile.data());
  CHECK(tile[4 * 12 + 1] == 0x87654321u);   // lane 12 = (g 3, t 0), reg a1
  CHECK(tile[128 + 11] == 0x3B3A3938u);     // row 11's scale word
  std::vector<uint8_t> c3(64, 0);
  for (int k = 32; k < 40; ++k) c3[k] = uint8_t(k - 31);
  uint8_t s0[4] = {0, 0, 0, 0};
  pack_act_row(c3.data(), s0, K, 3, tile.data());
  CHECK(tile[4 * 12 + 2] == 0x87654321u);   // lane 12, reg a2, row 3
}

static void test_plan_covers_all_pairs() {
  const int rows = 37, slots = 10, E = 512, tile = 32;
  auto picks = random_picks(rows, slots, E, 1234);
  Plan p = make_plan(picks, rows, slots, E, tile);
  std::vector<int> seen(rows * slots, 0);
  for (size_t i = 0; i < p.items.size(); i += 3) {
    int e = p.items[i], first = p.items[i + 1], cnt = p.items[i + 2];
    CHECK(cnt >= 1 && cnt <= tile);
    for (int j = first; j < first + cnt; ++j) {
      CHECK(picks[p.members[j]] == e);
      ++seen[p.members[j]];
    }
  }
  for (int v : seen) CHECK(v == 1);
}

static void test_plan_partial_tiles() {
  const int rows = 3, slots = 10, E = 512, tile = 32;
  auto picks = random_picks(rows, slots, E, 1);
  Plan p = make_plan(picks, rows, slots, E, tile);
  int total = 0;
  for (size_t i = 0; i < p.items.size(); i += 3) total += p.items[i + 2];
  CHECK(total == rows * slots);
}

static void test_plan_unused_experts() {
  std::vector<int> picks = {5, 9};  // one row, two slots, experts 5 and 9 only
  Plan p = make_plan(picks, 1, 2, 512, 32);
  CHECK(p.items.size() == 6);
  CHECK(p.items[0] == 5 && p.items[3] == 9);
}

static void test_random_picks_distinct() {
  auto picks = random_picks(64, 10, 512, 99);
  for (int r = 0; r < 64; ++r)
    for (int a = 0; a < 10; ++a)
      for (int b = a + 1; b < 10; ++b) CHECK(picks[r * 10 + a] != picks[r * 10 + b]);
}

static void test_ref_gemm() {
  float A[2 * 3] = {1, 2, 3, 4, 5, 6}, W[2 * 3] = {1, 0, 1, 0, 1, 0}, C[4];
  ref_gemm(A, W, 2, 2, 3, C);
  CHECK(C[0] == 4 && C[1] == 2 && C[2] == 10 && C[3] == 5);
}

// Every candidate scale mapping must give each A row (0..15) / B col (0..7) exactly one supplying lane.
static void test_scale_map_candidates_cover() {
  for (int c = 0; c < N_SFA_CANDIDATES; ++c) {
    int seen[16] = {0};
    for (int lane = 0; lane < 32; ++lane) {
      const int r = sfa_row(lane, SFA_CANDIDATES[c].sel, SFA_CANDIDATES[c].pat);
      if (r >= 0) ++seen[r];
    }
    for (int r = 0; r < 16; ++r) CHECK(seen[r] == 1);
  }
  for (int sel = 0; sel < 4; ++sel) {
    int seen[8] = {0};
    for (int lane = 0; lane < 32; ++lane) {
      const int n = sfb_col(lane, sel);
      if (n >= 0) ++seen[n];
    }
    for (int n = 0; n < 8; ++n) CHECK(seen[n] == 1);
  }
}

static void test_scale_map_default_is_first_candidate() {
  CHECK(SFA_CANDIDATES[0].sel == 0 && SFA_CANDIDATES[0].pat == 0);
  CHECK(sfa_row(0, 0, 0) == 0 && sfa_row(1, 0, 0) == 8 && sfa_row(2, 0, 0) == -1);  // lanes 4g, 4g+1
  CHECK(sfa_row(0, 0, 1) == 0 && sfa_row(2, 0, 1) == 8 && sfa_row(1, 0, 1) == -1);  // lanes 4g, 4g+2
  CHECK(sfb_col(4 * 5 + 2, 2) == 5 && sfb_col(4 * 5 + 2, 0) == -1);
}

static void test_packed_readers_match_unpack() {
  const int N = 16, K = 128;
  std::mt19937 rng(5);
  std::vector<uint8_t> codes(N * K), scales(N * K / 16);
  for (auto& v : codes) v = rng() & 15;
  for (auto& v : scales) v = rng() % 0x7F;
  std::vector<uint32_t> packed(N / 8 * K / 64 * W_TILE_U32);
  pack_weight(codes.data(), scales.data(), N, K, packed.data());
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      CHECK(weight_code(packed.data(), K, n, k) == codes[n * K + k]);
      CHECK(weight_scale(packed.data(), K, n, k / 16) == scales[n * (K / 16) + k / 16]);
    }
}

static void test_random_packed_weight() {
  const int N = 64, K = 256;
  std::mt19937 rng(9);
  std::vector<uint32_t> packed(N / 8 * K / 64 * W_TILE_U32);
  random_packed_weight(rng, N, K, 0.05f, packed.data());
  int nonzero_scales = 0, neg = 0, pos = 0;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const uint8_t s = weight_scale(packed.data(), K, n, k / 16);
      CHECK(s < 0x7F);
      if (k % 16 == 0 && s) ++nonzero_scales;
      const float v = e2m1_decode(weight_code(packed.data(), K, n, k));
      neg += v < 0; pos += v > 0;
    }
  CHECK(nonzero_scales == N * K / 16);
  CHECK(neg > N * K / 3 && pos > N * K / 3);  // roughly symmetric codes
}

// In a window of W rows that all pick the reference row's experts, the reference (last) row sits at position
// (W-1) % tile of an item, so windows 1..128 move it through both 16-row sub-tiles.
static void test_invariance_picks_move_row_through_tile() {
  const int slots = 10, tile = 32;
  const std::vector<int> ref = random_picks(1, slots, 512, 4);
  for (int W : {1, 2, 17, 32, 33, 50, 128}) {
    const std::vector<int> picks = invariance_picks(W, ref);
    CHECK(int(picks.size()) == W * slots);
    Plan p = make_plan(picks, W, slots, 512, tile);
    for (size_t i = 0; i < p.items.size(); i += 3)
      for (int j = 0; j < p.items[i + 2]; ++j)
        if (p.members[p.items[i + 1] + j] / slots == W - 1) CHECK(j == (W - 1) % tile);
  }
}

// Fused up GEMM: packed column block j (128 columns) = gate rows 64j..64j+63, then up rows I+64j..I+64j+63.
static void test_gate_up_mapping() {
  const int I = 640;
  std::vector<int> seen(2 * I, 0);
  for (int c = 0; c < 2 * I; ++c) {
    const int r = gate_up_row(c, I);
    CHECK(r >= 0 && r < 2 * I);
    ++seen[r];
    CHECK(gate_up_col(r, I) == c);
  }
  for (int v : seen) CHECK(v == 1);
  CHECK(gate_up_row(0, I) == 0 && gate_up_row(63, I) == 63);
  CHECK(gate_up_row(64, I) == I && gate_up_row(127, I) == I + 63);
  CHECK(gate_up_row(128, I) == 64 && gate_up_row(192, I) == I + 64);
}

static void test_plan_tile128() {
  const int rows = 8192, slots = 10, E = 512, tile = 128;
  auto picks = random_picks(rows, slots, E, 1234);
  Plan p = make_plan(picks, rows, slots, E, tile);
  int total = 0, big = 0;
  for (size_t i = 0; i < p.items.size(); i += 3) {
    CHECK(p.items[i + 2] >= 1 && p.items[i + 2] <= tile);
    total += p.items[i + 2];
    big += p.items[i + 2] > 64;
  }
  CHECK(total == rows * slots);
  CHECK(big > 0);  // at R = 8192 most items are well past half a tile
}

int main() {
  test_e2m1_table();
  test_e2m1_rounding();
  test_ue4m3_roundtrip();
  test_quantize_block();
  test_quantize_zero_block();
  test_quantize_tiny_block();
  test_quantize_global_scale();
  test_pack8();
  test_weight_pack_roundtrip();
  test_weight_pack_layout();
  test_act_pack_layout();
  test_plan_covers_all_pairs();
  test_plan_partial_tiles();
  test_plan_unused_experts();
  test_random_picks_distinct();
  test_ref_gemm();
  test_scale_map_candidates_cover();
  test_scale_map_default_is_first_candidate();
  test_packed_readers_match_unpack();
  test_random_packed_weight();
  test_invariance_picks_move_row_through_tile();
  test_gate_up_mapping();
  test_plan_tile128();
  test_e2m1_encode_matches_reference();
  test_ue4m3_encode_matches_reference();
  if (g_fail) {
    std::printf("%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("all ref tests passed\n");
  return 0;
}
