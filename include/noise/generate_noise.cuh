#pragma once

extern "C" CUDA_KERNELS_API void generateNoise(float* output, int N, float min_val, float max_val, unsigned int seed);
