#pragma once

extern "C" CUDA_KERNELS_API void vectorPow(const float* A, const float* B, float* C, int N);
