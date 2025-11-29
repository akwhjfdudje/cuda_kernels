#pragma once
#include <cuda_runtime.h>

extern "C" CUDA_KERNELS_API void generateMeshFromHeightmap(
    const float* d_heightmap,
    float3* d_vertices,
    float3* d_normals,
    float2* d_texcoords,
    int width,
    int height,
    float scaleX,
    float scaleY,
    float heightScale);
