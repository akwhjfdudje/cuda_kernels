#pragma once

extern "C" CUDA_KERNELS_API void vectorReLU(const float* A, float* B, int N);
