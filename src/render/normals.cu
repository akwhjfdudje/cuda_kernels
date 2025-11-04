/**
 * @file render/normals.cu
 * @brief CUDA kernel for computing normals from a heightmap using finite differences.
 */

#include <cuda_runtime.h>
#include <math.h>
#include "render/render.cuh"

#define THREADS_PER_BLOCK 256

/**
 * @brief CUDA kernel to compute surface normals from a heightmap using finite differences.
 *
 * Each normal is computed as the gradient of the heightmap at that point:
 * - x component = difference between left and right neighbors
 * - y component = a fixed up factor (for scaling)
 * - z component = difference between down and up neighbors
 *
 * Normals are then normalized to unit length.
 *
 * @param heightmap Pointer to the input heightmap array of size (width * height)
 * @param normals Pointer to the output normals array (float3) of size (width * height)
 * @param width Width of the heightmap
 * @param height Height of the heightmap
 *
 * @note Boundary pixels use their own value for missing neighbors.
 */
__global__ void computeNormalsKernel(const float* heightmap, float3* normals, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;

    // Finite differences
    float hL = (x > 0) ? heightmap[y * width + (x - 1)] : heightmap[y * width + x];
    float hR = (x < width - 1) ? heightmap[y * width + (x + 1)] : heightmap[y * width + x];
    float hD = (y > 0) ? heightmap[(y - 1) * width + x] : heightmap[y * width + x];
    float hU = (y < height - 1) ? heightmap[(y + 1) * width + x] : heightmap[y * width + x];

    // Compute normal
    float3 n;
    n.x = hL - hR;
    n.y = 2.0f;
    n.z = hD - hU;

    // Normalize
    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-6f) {
        n.x /= len;
        n.y /= len;
        n.z /= len;
    }

    normals[idx] = n;
}

/**
 * @brief Host launcher for computeNormals.
 */
extern "C" CUDA_KERNELS_API void computeNormals(const float* heightmap, float3* normals, int width, int height) {
    int total = width * height;
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    computeNormalsKernel<<<blocks, threads>>>(heightmap, normals, width, height);
    cudaDeviceSynchronize();
}
