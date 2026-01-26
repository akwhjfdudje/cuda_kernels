/**
 * @file noise/generate_noise.cu
 * @brief CUDA kernel for generating uniform random noise on the GPU.
 * 
 * This module provides efficient parallel generation of uniform random noise
 * using a Linear Congruential Generator (LCG) algorithm. The noise generation
 * is fully deterministic based on the provided seed, making it suitable for
 * reproducible results in graphics, simulations, and procedural generation.
 */

#include <cuda_runtime.h>
#include "noise/noise.cuh"
#include "utils/utils.cuh"

/**
 * @brief CUDA kernel for generating uniform random noise.
 * 
 * Each thread generates a random value for its corresponding output element.
 * The random values are uniformly distributed in the range [min_val, max_val).
 * 
 * @param output Pointer to device output array (size: N)
 * @param N Number of elements to generate
 * @param min_val Minimum value of the noise range (inclusive)
 * @param max_val Maximum value of the noise range (exclusive)
 * @param seed_base Base seed for random number generation. Each thread uses
 *                  a unique seed derived from this base and its thread index.
 */
__global__ void generateNoiseKernel(
    float* output,
    int N,
    float min_val,
    float max_val,
    unsigned int seed_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Generate unique seed for this thread
    // Using prime multipliers to ensure good distribution across threads
    unsigned int seed = seed_base + (unsigned int)(idx * 9781u + blockIdx.x * 6271u);
    
    // Generate random value in [0, 1)
    float rnd = lcg_random(seed);
    
    // Scale to [min_val, max_val)
    output[idx] = min_val + rnd * (max_val - min_val);
}

/**
 * @brief Host launcher for the noise generation kernel.
 * 
 * This function allocates device memory, launches the noise generation kernel,
 * and copies the results back to the host. The generated noise values are
 * uniformly distributed in the specified range [min_val, max_val).
 * 
 * @param output Pointer to host output array (size: N). Must be pre-allocated.
 * @param N Number of noise values to generate
 * @param min_val Minimum value of the noise range (inclusive)
 * @param max_val Maximum value of the noise range (exclusive)
 * @param seed Random seed for noise generation. Different seeds produce
 *             different noise patterns. The same seed will always produce
 *             the same sequence of values.
 * 
 * @note The output array must be pre-allocated with at least N elements.
 * @note Thread block size is set to 256 threads for optimal performance.
 * 
 * @example
 *   // Generate 1000 random values in range [-1.0, 1.0)
 *   std::vector<float> noise(1000);
 *   generateNoise(noise.data(), 1000, -1.0f, 1.0f, 12345u);
 */
extern "C" CUDA_KERNELS_API void generateNoise(
    float* output,
    int N,
    float min_val,
    float max_val,
    unsigned int seed
) {
    if (N <= 0) {
        return;
    }
    
    float* d_output;
    size_t size = N * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc(&d_output, size);
    
    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    generateNoiseKernel<<<blocks, threads>>>(d_output, N, min_val, max_val, seed);
    
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_output);
}
