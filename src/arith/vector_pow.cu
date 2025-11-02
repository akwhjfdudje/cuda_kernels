/**
 * @file arith/vector_pow.cu
 * @brief Elementwise power of two contiguous float arrays on the GPU.
 */

#include <cuda_runtime.h>
#include "arith/arith.cuh"

/**
 * @brief Performs elementwise power: C[i] = A[i] ^ B[i]
 * @param A Pointer to base array A
 * @param B Pointer to exponent array B
 * @param C Pointer to output array C
 * @param N Number of elements
 */
__global__ void vectorPowKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = powf(A[i], B[i]);
}

/**
 * @brief Host launcher for the elementwise power kernel.
 * 
 * @param A Pointer to device array A.
 * @param B Pointer to device array B.
 * @param C Pointer to device array for results.
 * @param N Number of elements to process.
 */
extern "C" CUDA_KERNELS_API void vectorPow(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorPowKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
