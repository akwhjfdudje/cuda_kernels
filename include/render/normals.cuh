#pragma once
#include <cuda_runtime.h>

extern "C" CUDA_KERNELS_API void computeNormals(const float* heightmap, float3* normals, int width, int height);
