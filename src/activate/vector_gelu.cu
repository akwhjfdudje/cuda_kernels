/**
 * @file activate/vector_gelu.cu
 * @brief Elementwise GELU of a floats array on the GPU.
 */

#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <math.h>
#include "activate/activate.cuh"

/**
 * @brief Performs elementwise GELU: B[i] = GELU(A[i])
 * @param A Pointer to input array A
 * @param B Pointer to output array B
 * @param N Number of elements
 */
__global__ void vectorGELUKernel(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = A[i];
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        B[i] = gelu;
    }
}

/**
 * @brief Host launcher for the elementwise GELU kernel.
 * 
 * @param A Pointer to device array A.
 * @param B Pointer to device array for results.
 * @param N Number of elements to process.
 */
extern "C" CUDA_KERNELS_API void vectorGELU(const float* A, float* B, int N) {
    float *d_A, *d_B;
    size_t size = N * sizeof(float);

    // Allocate GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorGELUKernel<<<blocks, threads>>>(d_A, d_B, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
}
