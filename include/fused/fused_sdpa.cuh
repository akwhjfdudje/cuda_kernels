#pragma once

extern "C" CUDA_KERNELS_API void fusedScaledDotProductAttention(
    const float* Q, const float* K, const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v);
