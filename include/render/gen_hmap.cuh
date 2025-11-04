#pragma once

extern "C" CUDA_KERNELS_API void generateHeightmap(float* heightmap, int width, int height, float scale, int seed, float mix_ratio);
