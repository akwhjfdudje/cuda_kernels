#pragma once

extern "C" CUDA_KERNELS_API void vectorMul(const float* A, const float* B, float* C, int N);
