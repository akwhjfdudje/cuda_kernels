/**
 * @file activate/vector_relu.cu
 * @brief Elementwise ReLU of a floats array on the GPU.
 */

#include <iostream>
#include <cuda_runtime.h>
#include "activate/activate.cuh"
#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error at " << __FILE__ << ":"      \
                  << __LINE__ << " — "                        \
                  << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)
/**
 * @brief Performs elementwise ReLU: B[i] = ReLU(A[i])
 * @param A Pointer to input array A
 * @param B Pointer to output array B
 * @param N Number of elements
 */
__global__ void vectorReLUKernel(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        B[i] = fmaxf(0.0f, A[i]);
}

/**
 * @brief Host launcher for the elementwise ReLU kernel.
 * 
 * @param A Pointer to device array A.
 * @param B Pointer to device array for results.
 * @param N Number of elements to process.
 */
void vectorReLU(const float* A, float* B, int N) {
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
    vectorReLUKernel<<<blocks, threads>>>(d_A, d_B, N);
    CHECK_CUDA(cudaGetLastError());           // catch launch errors

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
}
