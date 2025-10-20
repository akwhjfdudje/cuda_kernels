/**
 * @file reduce/reduce_sum.cu
 * @brief Reduction sum of an input array on the GPU.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include "reduce/reduce.cuh"

/**
 * @brief Performs a reduction sum: B = sum(A)
 * @param A Pointer to input array A
 * @param B Pointer to output array B
 * @param N Number of elements
 */
__global__ void reduceSumKernel(const float* A, float* B, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;

    if (i < N)
        sum = A[i];
    if (i + blockDim.x < N)
        sum += A[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // In-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        B[blockIdx.x] = sdata[0];
}

/**
 * @brief Host launcher for the reduce sum kernel.
 * 
 * @param A Pointer to device array A.
 * @param N Number of elements to process.
 * @return The reduced sum of A: sum(A)
 */
float reduceSum(const float* A, int N) {
    float *dIn, *dOut;
    size_t size = N * sizeof(float);
    cudaMalloc(&dIn, size);

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&dOut, blocks * sizeof(float));

    cudaMemcpy(dIn, A, size, cudaMemcpyHostToDevice);

    reduceSumKernel<<<blocks, threads, threads * sizeof(float)>>>(dIn, dOut, N);

    float *hOut = new float[blocks];
    cudaMemcpy(hOut, dOut, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; ++i)
        total += hOut[i];

    delete[] hOut;
    cudaFree(dIn);
    cudaFree(dOut);

    return total;
}
