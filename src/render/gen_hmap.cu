/**
 * @file gen_hmap.cu
 * @brief CUDA kernel for generating a terrain heightmap using Perlin noise and Voronoi distance fields.
 */

#include <cuda_runtime.h>
#include <math.h>
#include "render/render.cuh"

#define THREADS_PER_BLOCK 256

__device__ inline float hash(int x, int y, int seed = 1337) {
    int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

__device__ inline float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float perlin(float x, float y, int seed) {
    //
    // This function generates smooth, continuous noise based on grid points.
    // The algorithm calculates the corner values of a grid cell, hashes them, and then interpolates
    // between those values using smooth curves to produce continuous, natural-looking randomness.
    // 
    // x: The x-coordinate for noise sampling.
    // y: The y-coordinate for noise sampling.
    // seed: The seed for random number generation, ensuring different results per seed.
    // returns the Perlin noise value for the coordinates (x, y).
    //
    int x0 = floorf(x);
    int y0 = floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = fade(x - x0);
    float sy = fade(y - y0);

    float n00 = hash(x0, y0, seed);
    float n10 = hash(x1, y0, seed);
    float n01 = hash(x0, y1, seed);
    float n11 = hash(x1, y1, seed);

    float ix0 = lerp(n00, n10, sx);
    float ix1 = lerp(n01, n11, sx);
    return lerp(ix0, ix1, sy);
}

__device__ float voronoi(float x, float y, int cell_count, int seed) {
    //
    // This function calculates the distance from a given point (x, y) to the 
    // nearest point in a 2D grid of randomly distributed points (cells). It creates a 
    // "cellular" pattern where each point is assigned to the nearest "site" or cell center. 
    // The algorithm computes this by checking the distance to neighboring cells 
    // and selecting the smallest distance.
    // 
    // x: The x-coordinate to sample.
    // y: The y-coordinate to sample.
    // cell_count: The number of cells per dimension to sample from.
    // seed: The seed for random number generation, ensuring different results per seed.
    // returns the distance to the nearest Voronoi site.
    //
    int xi = floorf(x);
    int yi = floorf(y);

    float minDist = 1e10f;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int cx = xi + i;
            int cy = yi + j;

            float rx = hash(cx, cy, seed) * 0.5f + 0.5f;
            float ry = hash(cx, cy, seed + 42) * 0.5f + 0.5f;

            float dx = (cx + rx) - x;
            float dy = (cy + ry) - y;
            float dist = sqrtf(dx * dx + dy * dy);
            minDist = fminf(minDist, dist);
        }
    }
    return minDist;
}

/**
 * @brief CUDA kernel generating a heightmap combining Perlin and Voronoi noise.
 *
 * @param heightmap Output buffer of size (width * height)
 * @param width Width of the heightmap
 * @param height Height of the heightmap
 * @param scale Scaling factor for noise sampling
 * @param seed Random seed for noise generation
 * @param mix_ratio Blend ratio between Perlin and Voronoi fields (0 = pure Perlin, 1 = pure Voronoi)
 */
__global__ void generateHeightmapKernel(
    float* heightmap, int width, int height,
    float scale, int seed, float mix_ratio
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;

    float fx = x / scale;
    float fy = y / scale;

    // Multi-octave Perlin noise
    float amp = 1.0f;
    float freq = 1.0f;
    float noise_sum = 0.0f;
    for (int o = 0; o < 4; ++o) {
        noise_sum += perlin(fx * freq, fy * freq, seed + o * 17) * amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }

    // Voronoi distance field
    float v = voronoi(fx * 0.5f, fy * 0.5f, 8, seed + 999);

    // Combine fields: Perlin for broad features, Voronoi for structure
    float height_val = (1.0f - mix_ratio) * noise_sum + mix_ratio * (1.0f - v * 2.0f);

    // Clamp and store
    height_val = fmaxf(-1.0f, fminf(1.0f, height_val));
    heightmap[idx] = height_val;
}

/**
 * @brief Host launcher for heightmap generation kernel.
 *
 * @param heightmap Pointer to host output array (size: width * height)
 * @param width Width of the heightmap
 * @param height Height of the heightmap
 * @param scale Sampling scale for the noise fields
 * @param seed Random seed
 * @param mix_ratio Blend ratio between Perlin and Voronoi fields
 */
extern "C" CUDA_KERNELS_API void generateHeightmap(
    float* heightmap, int width, int height,
    float scale, int seed, float mix_ratio
) {
    float* d_out;
    size_t size = width * height * sizeof(float);
    cudaMalloc(&d_out, size);

    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((width * height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    generateHeightmapKernel<<<blocks, threads>>>(
        d_out, width, height, scale, seed, mix_ratio
    );

    cudaMemcpy(heightmap, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}
