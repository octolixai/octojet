// Octojet: a prompt chunk's hyper-connection up projection with the stream mix in its epilogue (one kernel instead of
// the prefill matmul, a [rows, S*D] bf16 round trip and glue.hc_mix).
//
// Same bits as prefill_matmul then glue.hc_mix, by construction:
//  - the matmul: each output's fp32 chain is the shared prefill kernel's (qmm_prefill.cu): weights bf16(fma(q, s, b))
//    with fma.rn.bf16x2, inputs from the same fragments, one mma.m16n8k16 per 16 inputs in group order from zero.
//    Which block or warp computes a column never changes its chain, so the columns can be regrouped: a block takes
//    64 dims of every stream (one stored 64-column tile a stream), and each thread holds a dim's S streams;
//  - the mix: glue.hc_mix's fp32 operations in its order as Triton lowers them (exp: ex2.approx of x * log2(e);
//    fp32 division: div.full), bf16 roundings where it rounds. Its 32-dim group sums are a separate Triton kernel
//    (glue.hc_mix_sums), whose reduction is hc_mix's own.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

constexpr int GS = 32;                    // inputs per quantization group (the HC matrices)
constexpr int S = 4;                      // streams
constexpr int DB = 64;                    // dims of every stream a block: one stored tile each
constexpr int WN = 4;                     // warps across the dims (16 dims, 2 n8 tiles a stream, each)
constexpr int ROW = GS * 2;               // bytes of one input row a group
constexpr int CHUNKS = ROW / 16;
constexpr int TILE_BYTES = 64 * GS / 2;   // one stored 64-column tile's group block
constexpr int W_BYTES = S * TILE_BYTES;
constexpr int SB = S * DB * 2;            // bytes of the block's scales (and of its biases) a group

template <int BM, int WM, int STAGES>
struct UpTile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int X = BM * ROW;
    static constexpr int STAGE = X + W_BYTES + 2 * SB;
    static constexpr int SMEM = STAGES * STAGE;
};

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

// glue._bsig: bf16(1 / (1 + exp(-x))) as Triton emits it (math.exp -> ex2.approx.f32(x * 1.4426950408889634),
// fp32 divf -> div.full.f32)
__device__ __forceinline__ float bsig(float x) {
    const float e = ex2_approx(__fmul_rn(-x, 1.4426950408889634f));
    return bf(div_full(1.0f, __fadd_rn(1.0f, e)));
}

template <int BM, int WM, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) upmix_kernel(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, const __nv_bfloat16* __restrict__ normed,
        __nv_bfloat16* __restrict__ mixed, int M, int K, int D, int npad, int ldx) {
    using T = UpTile<BM, WM, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int m0 = blockIdx.x * BM, d0 = blockIdx.y * DB;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // group g's inputs, the S stored tiles holding columns st * D + d0 .. + 63, their scales and biases into stage s
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * CHUNKS; c += T::THREADS) {
            const int r = c / CHUNKS, ch = c % CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * ROW + swz<CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < W_BYTES / 16; c += T::THREADS) {
            const int st = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>((st * D + d0) / 64) * KG + g;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + W_BYTES;
        for (int c = tid; c < 2 * (SB / 16); c += T::THREADS) {
            const int which = c / (SB / 16), rest = c % (SB / 16);
            const int st = rest / (DB * 2 / 16), off = rest % (DB * 2 / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + st * D + d0 + off * 8;
            cp16(ps + which * SB + rest * 16, src);
        }
    };

    // n8 tile j = st * 2 + t: columns st * D + d0 + (wn * 2 + t) * 8 .. + 7
    float acc[T::MT][2 * S][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < 2 * S; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + W_BYTES);
        uint32_t words[2 * S], sv[2 * S], bv[2 * S];
#pragma unroll
        for (int j = 0; j < 2 * S; ++j) {
            const int st = j / 2, sub = wn * 2 + (j & 1);            // the stream, its n8 tile in the stored tile
            words[j] = pw[(st * 8 + sub) * 32 + lane];
            const int col = st * DB + sub * 8 + (lane >> 2);         // this lane's B-fragment column in the block
            sv[j] = ps[col] * 0x10001u;                              // (s, s) and (b, b) as bf16 pairs
            bv[j] = ps[S * DB + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * ROW + swz<CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < 2 * S; ++j) {
                const uint32_t b0 = fma2(pair(words[j], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(pair(words[j], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    __syncthreads();
    // glue.hc_mix: mixed = bf16((sum over streams in order of bf16(bsig(bf16(up_s)) * normed_s)) / S)
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const int dc = d0 + (wn * 2 + t) * 8 + (lane & 3) * 2;
                float total0 = 0.0f, total1 = 0.0f;
#pragma unroll
                for (int st = 0; st < S; ++st) {
                    const __nv_bfloat162 nv = *reinterpret_cast<const __nv_bfloat162*>(
                        normed + static_cast<size_t>(row) * (S * D) + st * D + dc);
                    const float u0 = bf(acc[i][st * 2 + t][2 * h]), u1 = bf(acc[i][st * 2 + t][2 * h + 1]);
                    total0 = __fadd_rn(total0, bf(__fmul_rn(bsig(u0), __low2float(nv))));
                    total1 = __fadd_rn(total1, bf(__fmul_rn(bsig(u1), __high2float(nv))));
                }
                *reinterpret_cast<__nv_bfloat162*>(mixed + static_cast<size_t>(row) * D + dc) =
                    __floats2bfloat162_rn(div_full(total0, static_cast<float>(S)), div_full(total1, static_cast<float>(S)));
            }
}

template <int BM, int WM, int STAGES>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
            const at::Tensor& normed, at::Tensor& mixed, int D) {
    using T = UpTile<BM, WM, STAGES>;
    const int M = x.size(0), K = x.size(1);
    auto kernel = upmix_kernel<BM, WM, STAGES>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    const dim3 grid((M + BM - 1) / BM, D / DB);
    kernel<<<grid, T::THREADS, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(normed.data_ptr()), reinterpret_cast<__nv_bfloat16*>(mixed.data_ptr()),
        M, K, D, static_cast<int>(scales.size(1)), M == 1 ? K : static_cast<int>(x.stride(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ``tile`` (0: 64 rows a block, 8 warps; 1: 128 rows, 8 warps; 2: 128 rows, 16 warps; 3: 32 rows, 4 warps) never
// changes a row's bits.
void hc_upmix_prefill_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales,
                           const at::Tensor& biases, const at::Tensor& normed, at::Tensor& mixed, int D, int tile) {
    switch (tile) {
        case 1: launch<128, 2, 4>(x, w, scales, biases, normed, mixed, D); break;
        case 2: launch<128, 4, 4>(x, w, scales, biases, normed, mixed, D); break;
        case 3: launch<32, 1, 4>(x, w, scales, biases, normed, mixed, D); break;
        default: launch<64, 2, 4>(x, w, scales, biases, normed, mixed, D); break;
    }
}
