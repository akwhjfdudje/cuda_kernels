#pragma once

extern "C" CUDA_KERNELS_API void vectorAdd(const float* A, const float* B, float* C, int N);
