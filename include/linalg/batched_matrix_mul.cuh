#pragma once

extern "C" CUDA_KERNELS_API void batchedMatrixMul(const float* A, const float* B, float* C, int M, int K, int N, int batch);
