#pragma once

extern "C" CUDA_KERNELS_API void vectorGELU(const float* A, float* B, int N);
