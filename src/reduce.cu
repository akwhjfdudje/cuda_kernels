#include <cuda_runtime.h>
#include <cstdio>
#include "reduce.cuh"

// B = sum(A)
__global__ void reduceKernel(const float* A, float* B, int N) {
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

float reduceSum(const float* A, int N) {
    float *dIn, *dOut;
    size_t size = N * sizeof(float);
    cudaMalloc(&dIn, size);

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&dOut, blocks * sizeof(float));

    cudaMemcpy(dIn, A, size, cudaMemcpyHostToDevice);

    reduceKernel<<<blocks, threads, threads * sizeof(float)>>>(dIn, dOut, N);

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
