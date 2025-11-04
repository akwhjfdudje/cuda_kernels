#pragma once

extern "C" CUDA_KERNELS_API void normalizeHeightmap(float* heightmap, int width, int height, float min_val, float max_val);
