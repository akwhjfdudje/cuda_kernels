#pragma once

extern "C" CUDA_KERNELS_API void matrixMul(const float* A, const float* B, float* C, int N);
