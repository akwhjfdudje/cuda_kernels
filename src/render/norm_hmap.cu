/**
 * @file render/norm_hmap.cu
 * @brief CUDA kernel for normalizing a heightmap to [-1, 1].
 */

#include <cuda_runtime.h>
#include "render/render.cuh"

#define THREADS_PER_BLOCK 256

/**
 * @brief CUDA kernel normalizing a heightmap between [-1, 1].
 *
 * @param heightmap Output buffer of size (width * height)
 * @param width Width of the heightmap
 * @param height Height of the heightmap
 * @param min_val Minimum value of the heightmap
 * @param max_val Maximum value of the heightmap
 */
__global__ void normalizeHeightmapKernel(float* heightmap, int width, int height, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    float val = heightmap[idx];
    val = (val - min_val) / (max_val - min_val);  // map to [0,1]
    val = val * 2.0f - 1.0f;                      // map to [-1,1]
    heightmap[idx] = val;
}

/**
 * @brief Host launcher for normalizeHeightmap.
 */
extern "C" CUDA_KERNELS_API void normalizeHeightmap(float* heightmap, int width, int height, float min_val, float max_val) {
    int total = width * height;
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    normalizeHeightmapKernel<<<blocks, threads>>>(heightmap, width, height, min_val, max_val);
    cudaDeviceSynchronize();
}
