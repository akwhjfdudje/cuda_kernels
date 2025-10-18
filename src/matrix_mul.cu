#include <cuda_runtime.h>
#include <cstdio>
#include "matrix_mul.cuh"

#define TILE_DIM 16

// Kernel: C = A * B
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < N && t * TILE_DIM + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_DIM + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_DIM + threadIdx.y < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

void matrixMul(const float* A, const float* B, float* C, int N) {
    float *dA, *dB, *dC;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    matrixMulKernel<<<blocks, threads>>>(dA, dB, dC, N);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
