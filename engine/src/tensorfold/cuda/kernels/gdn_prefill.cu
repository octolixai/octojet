// Prefill chain: steps never depend on where a chunk starts, so any chunking gives the same bits (not verify's).
// Two forms run the same arithmetic in the same order (chain_steps): ``chain_kernel`` stages STAGE rows into shared
// memory, then steps them; ``ahead_kernel`` (fp32 keys, 16-byte aligned rows) holds two stage buffers and copies the
// next STAGE rows with cp.async while it steps the current ones, so the loads leave the chain's critical path.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int DK = 128;
constexpr int DV = 128;
constexpr int STAGE = 32;                 // rows a stage (both forms)

__device__ __forceinline__ float4 load4f(const __nv_bfloat16* p) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

__device__ __forceinline__ float4 load4f(const float* p) { return *reinterpret_cast<const float4*>(p); }

// This thread's half of value row ``row``: 64 of its 128 key columns, interleaved by fours (8j + 4 half .. + 4).
__device__ __forceinline__ void load_state(float (&s)[64], const float* __restrict__ state, int head, int row,
                                           int half) {
    const float* s0 = state + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
}

__device__ __forceinline__ void store_state(const float (&s)[64], float* __restrict__ last, int head, int row,
                                            int half) {
    float* s1 = last + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
}

// The ``n`` staged rows from prompt row ``t0`` stepped in order: decay, read the memory, correct it, read out.
template <int ROWS>
__device__ __forceinline__ void chain_steps(float (&s)[64], const float4 (*ks)[DK / 4], const float4 (*qs)[DK / 4],
                                            const float* gs, const float* bs, const __nv_bfloat16 (*vs)[ROWS], int n,
                                            int t0, int head, int hv, int row, int half,
                                            __nv_bfloat16* __restrict__ y) {
    for (int tt = 0; tt < n; ++tt) {
        const float gt = gs[tt], bt = bs[tt];
        const size_t vat = (static_cast<size_t>(t0 + tt) * hv + head) * DV + row;
        const float vt = __bfloat162float(vs[tt][threadIdx.x >> 1]);
        float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float4 kk = ks[tt][2 * j + half];
            s[4 * j] = s[4 * j] * gt;
            s[4 * j + 1] = s[4 * j + 1] * gt;
            s[4 * j + 2] = s[4 * j + 2] * gt;
            s[4 * j + 3] = s[4 * j + 3] * gt;
            m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
            m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
            m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
            m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
        }
        float mem = (m[0] + m[1]) + (m[2] + m[3]);
        mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
        const float delta = (vt - mem) * bt;
        float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float4 kk = ks[tt][2 * j + half], qq = qs[tt][2 * j + half];
            s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
            s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
            s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
            s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
            o[0] = __fmaf_rn(s[4 * j], qq.x, o[0]);
            o[1] = __fmaf_rn(s[4 * j + 1], qq.y, o[1]);
            o[2] = __fmaf_rn(s[4 * j + 2], qq.z, o[2]);
            o[3] = __fmaf_rn(s[4 * j + 3], qq.w, o[3]);
        }
        float out = (o[0] + o[1]) + (o[2] + o[3]);
        out = out + __shfl_xor_sync(0xffffffffu, out, 1);
        if (half == 0) y[vat] = __float2bfloat16_rn(out);
    }
}

template <typename QK, int ROWS, int STEPS>
__global__ void __launch_bounds__(2 * ROWS) chain_kernel(
        const QK* __restrict__ q, const QK* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
        float* __restrict__ last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    __shared__ float4 ks[STEPS][DK / 4], qs[STEPS][DK / 4];
    __shared__ float gs[STEPS], bs[STEPS];
    __shared__ __nv_bfloat16 vs[STEPS][ROWS];
    const int head = blockIdx.x, row = blockIdx.y * ROWS + (threadIdx.x >> 1), half = threadIdx.x & 1;
    const int key_head = head / (hv / hk);
    float s[64];
    load_state(s, state, head, row, half);
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < n * (DK / 4); i += 2 * ROWS) {
            const int st = i / (DK / 4), c = i % (DK / 4);
            const size_t at = (static_cast<size_t>(t0 + st) * hk + key_head) * DK + 4 * c;
            ks[st][c] = load4f(k + at);
            qs[st][c] = load4f(q + at);
        }
        for (int i = threadIdx.x; i < n; i += 2 * ROWS) {
            gs[i] = g[static_cast<size_t>(t0 + i) * hv + head];
            bs[i] = beta[static_cast<size_t>(t0 + i) * hv + head];
        }
        for (int i = threadIdx.x; i < n * ROWS; i += 2 * ROWS) {
            const int st = i / ROWS, rr = i % ROWS;
            vs[st][rr] = v[(static_cast<size_t>(t0 + st) * hv + head) * DV + blockIdx.y * ROWS + rr];
        }
        __syncthreads();
        chain_steps<ROWS>(s, ks, qs, gs, bs, vs, n, t0, head, hv, row, half, y);
    }
    store_state(s, last, head, row, half);
}

// ---- ahead_kernel: double-buffered stages filled by cp.async ----

__device__ __forceinline__ void async16(void* dst, const void* src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void async4(void* dst, const void* src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int PENDING>
__device__ __forceinline__ void async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING) : "memory"); }

// One stage's rows as chain_steps reads them: fp32 keys and queries, bf16 values of this block's value rows, gates.
template <int ROWS>
struct Stage {
    float4 k[STAGE][DK / 4];
    float4 q[STAGE][DK / 4];
    __nv_bfloat16 v[STAGE][ROWS];
    float g[STAGE];
    float b[STAGE];
};

template <int ROWS>
constexpr int ahead_smem() { return static_cast<int>(2 * sizeof(Stage<ROWS>)); }

// Queue the copies of prompt rows [t0, t0 + n) into ``st`` as one cp.async group (only valid rows are copied).
template <int ROWS>
__device__ __forceinline__ void fill_stage(Stage<ROWS>& st, const float* __restrict__ q, const float* __restrict__ k,
                                           const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                           const float* __restrict__ beta, int t0, int n, int hk, int hv, int head,
                                           int key_head) {
    constexpr int THREADS = 2 * ROWS, KP = DK / 4, VP = ROWS / 8;    // 16-byte pieces: a key row, a value slice
    for (int i = threadIdx.x; i < n * KP; i += THREADS) {
        const int r = i / KP, c = i - r * KP;
        const size_t at = (static_cast<size_t>(t0 + r) * hk + key_head) * DK + 4 * c;
        async16(&st.k[r][c], k + at);
        async16(&st.q[r][c], q + at);
    }
    for (int i = threadIdx.x; i < n * VP; i += THREADS) {
        const int r = i / VP, c = i - r * VP;
        async16(&st.v[r][8 * c], v + (static_cast<size_t>(t0 + r) * hv + head) * DV + blockIdx.y * ROWS + 8 * c);
    }
    if (static_cast<int>(threadIdx.x) < n) {
        const size_t at = static_cast<size_t>(t0 + threadIdx.x) * hv + head;
        async4(&st.g[threadIdx.x], g + at);
        async4(&st.b[threadIdx.x], beta + at);
    }
    async_commit();
}

template <int ROWS>
__global__ void __launch_bounds__(2 * ROWS) ahead_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
        float* __restrict__ last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    static_assert(STAGE <= 2 * ROWS, "one thread a gate");
    extern __shared__ uint4 ahead_raw[];
    Stage<ROWS>* stage = reinterpret_cast<Stage<ROWS>*>(ahead_raw);
    const int head = blockIdx.x, row = blockIdx.y * ROWS + (threadIdx.x >> 1), half = threadIdx.x & 1;
    const int key_head = head / (hv / hk);
    fill_stage<ROWS>(stage[0], q, k, v, g, beta, 0, min(STAGE, W), hk, hv, head, key_head);
    float s[64];
    load_state(s, state, head, row, half);
    int cur = 0;
    for (int t0 = 0; t0 < W; t0 += STAGE) {
        const int n = min(STAGE, W - t0);
        if (t0 + STAGE < W) {
            // the other buffer was last read in the previous stage, which ended on a barrier
            fill_stage<ROWS>(stage[cur ^ 1], q, k, v, g, beta, t0 + STAGE, min(STAGE, W - t0 - STAGE), hk, hv, head,
                             key_head);
            async_wait<1>();          // this thread's copies of the current stage have landed
        } else {
            async_wait<0>();
        }
        __syncthreads();              // ... and every thread's
        const Stage<ROWS>& sc = stage[cur];
        chain_steps<ROWS>(s, sc.k, sc.q, sc.g, sc.b, sc.v, n, t0, head, hv, row, half, y);
        __syncthreads();              // the current buffer is refilled at the start of the next stage
        cur ^= 1;
    }
    store_state(s, last, head, row, half);
}

template <typename QK, int ROWS>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g, const at::Tensor& beta,
            const at::Tensor& state, at::Tensor& last, at::Tensor& y) {
    const int W = q.size(0), hk = q.size(1), hv = v.size(1);
    const dim3 grid(hv, DV / ROWS);
    chain_kernel<QK, ROWS, STAGE><<<grid, 2 * ROWS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const QK*>(q.data_ptr()), reinterpret_cast<const QK*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), g.data_ptr<float>(), beta.data_ptr<float>(),
        state.data_ptr<float>(), last.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), W, hk, hv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// false when the device cannot give the two stage buffers (the caller then takes chain_kernel)
template <int ROWS>
bool launch_ahead(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                  const at::Tensor& beta, const at::Tensor& state, at::Tensor& last, at::Tensor& y) {
    constexpr int SMEM = ahead_smem<ROWS>();
    static int ready = 0;             // 0 unknown, 1 set, -1 not available on this device
    if (ready == 0) {
        int dev = 0, optin = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if (optin < SMEM) {
            ready = -1;
        } else {
            auto* kern = ahead_kernel<ROWS>;
            C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            ready = 1;
        }
    }
    if (ready < 0) return false;
    const int W = q.size(0), hk = q.size(1), hv = v.size(1);
    const dim3 grid(hv, DV / ROWS);
    ahead_kernel<ROWS><<<grid, 2 * ROWS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        g.data_ptr<float>(), beta.data_ptr<float>(), state.data_ptr<float>(), last.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), W, hk, hv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

bool aligned16(const at::Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0; }

} // namespace

// The form ``ahead_kernel`` takes: fp32 keys and 16-byte aligned key and value rows (contiguous tensors are checked
// by the caller; every row then starts on 16 bytes when the base does, as DK = DV = 128).
bool gdn_prefill_ahead_ok(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    return q.scalar_type() == at::kFloat && aligned16(q) && aligned16(k) && aligned16(v);
}

// Fewer heads than SMs: a block takes 64 value rows, not 128, so the chunk still fills the GPU. ``ahead`` asks for the
// double-buffered form where it applies; the return says which ran (true: ahead_kernel).
bool gdn_prefill_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                      const at::Tensor& beta, const at::Tensor& state, at::Tensor& last, at::Tensor& y, int sms,
                      bool ahead) {
    const bool wide = v.size(1) >= sms;
    if (ahead && gdn_prefill_ahead_ok(q, k, v)) {
        if (wide ? launch_ahead<128>(q, k, v, g, beta, state, last, y) : launch_ahead<64>(q, k, v, g, beta, state, last, y))
            return true;
    }
    if (q.scalar_type() == at::kFloat) {
        if (wide) launch<float, 128>(q, k, v, g, beta, state, last, y); else launch<float, 64>(q, k, v, g, beta, state, last, y);
    } else {
        if (wide) launch<__nv_bfloat16, 128>(q, k, v, g, beta, state, last, y);
        else launch<__nv_bfloat16, 64>(q, k, v, g, beta, state, last, y);
    }
    return false;
}
