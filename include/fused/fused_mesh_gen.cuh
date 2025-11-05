#pragma once
#include <cuda_runtime.h>

extern "C" CUDA_KERNELS_API void generateMeshToVBOs(
    const float* heightmap,
    int width,
    int height,
    float scale,
    cudaGraphicsResource_t vboVerticesRes,
    cudaGraphicsResource_t vboNormalsRes,
    cudaGraphicsResource_t vboTexcoordRes);
