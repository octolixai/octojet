#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void hc_upmix_prefill_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, at::Tensor&, int, int);

// x (M, K) bf16 (the read-out's SiLU activations) times the up projection packed for the prefill matmul (group 32,
// [S*D, K]), then the stream mix with ``normed`` (M, S*D) bf16: ``mixed`` (M, D) bf16.
void upmix(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
           const at::Tensor& normed, at::Tensor mixed, int64_t n, int64_t streams, int64_t tile) {
    TORCH_CHECK(streams == 4, "the fused prompt read-out takes 4 streams");
    TORCH_CHECK(tile >= 0 && tile <= 3, "tile 0-3");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    const int64_t m = x.size(0), k = x.size(1), kg = k / 32, npad = (n + 127) / 128 * 128, d = n / streams;
    TORCH_CHECK(k % 32 == 0 && n % streams == 0 && d % 64 == 0, "K whole groups of 32; D a multiple of 64");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / 32, n padded to 128) bf16");
    TORCH_CHECK(normed.is_cuda() && normed.is_contiguous() && normed.scalar_type() == at::kBFloat16 &&
                normed.size(0) == m && normed.size(1) == n, "normed: (M, S*D) bf16, contiguous");
    TORCH_CHECK(mixed.is_cuda() && mixed.is_contiguous() && mixed.scalar_type() == at::kBFloat16 &&
                mixed.size(0) == m && mixed.size(1) == d, "mixed: (M, D) bf16, contiguous");
    c10::cuda::CUDAGuard guard(x.device());
    hc_upmix_prefill_cuda(x, w, scales, biases, normed, mixed, static_cast<int>(d), static_cast<int>(tile));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upmix", &upmix);
}
