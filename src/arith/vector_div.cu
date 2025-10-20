/**
 * @file arith/vector_div.cu
 * @brief Elementwise division of two contiguous float arrays on the GPU.
 */

#include <cuda_runtime.h>
#include "arith/arith.cuh"

/**
 * @brief Performs elementwise division: C[i] = A[i] / B[i]
 * @param A Pointer to input array A
 * @param B Pointer to input array B
 * @param C Pointer to output array C
 * @param N Number of elements
 */
__global__ void vectorDivKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float d = B[i];
        C[i] = (fabsf(d) > 1e-8f) ? (A[i] / d) : 0.0f;
    }
}

/**
 * @brief Host launcher for the elementwise division kernel.
 * 
 * @param A Pointer to device array A.
 * @param B Pointer to device array B.
 * @param C Pointer to device array for results.
 * @param N Number of elements to process.
 */
void vectorDiv(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorDivKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
