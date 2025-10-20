#include <cuda_runtime.h>
#include "arith/arith.cuh"

/**
 * @file arith/vector_add.cu
 * @brief Elementwise addition of two contiguous float arrays on the GPU.
 */

/**
 * @brief Performs elementwise addition: C[i] = A[i] + B[i]
 * @param A Pointer to input array A
 * @param B Pointer to input array B
 * @param C Pointer to output array C
 * @param N Number of elements
 */
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void vectorAdd(const float* A, const float* B, float* C, int N) {
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
    vectorAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
