/**
 * @file arith/vector_log.cu
 * @brief Elementwise natural log of a floats array on the GPU.
 */

#include <cuda_runtime.h>
#include "arith/arith.cuh"

/**
 * @brief Performs elementwise logarithm: B[i] = log(A[i])
 * @param A Pointer to input array A
 * @param B Pointer to output array B
 * @param N Number of elements
 */
__global__ void vectorLogKernel(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        B[i] = logf(A[i]);
}

/**
 * @brief Host launcher for the elementwise natural logarithm kernel.
 * 
 * @param A Pointer to device array A.
 * @param B Pointer to device array for results.
 * @param N Number of elements to process.
 */
void vectorLog(const float* A, float* B, int N) {
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
    vectorLogKernel<<<blocks, threads>>>(d_A, d_B, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
}
