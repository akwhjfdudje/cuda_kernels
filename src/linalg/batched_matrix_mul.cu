/**
 * @file linalg/batched_matrix_mul.cu
 * @brief Batched matrix multiplication for (batch, N, N) tensors.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include "linalg/linalg.cuh"

#define TILE_DIM 16

/**
 * @brief Performs batched matrix multiplication on A and B: 
 *        C[i] = A[i] * B[i] for each batch i.
 * 
 * @param A Pointer to input array A (size = batch * N * N)
 * @param B Pointer to input array B (size = batch * N * N)
 * @param C Pointer to output array C (size = batch * N * N)
 * @param N Dimension of each square matrix
 * @param batch Number of matrices in the batch
 */
__global__ void batchedMatrixMulKernel(const float* A, const float* B, float* C, int N, int batch) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int batchIdx = blockIdx.z; // identify which matrix in the batch
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float value = 0.0f;

    const float* batchA = A + batchIdx * N * N;
    const float* batchB = B + batchIdx * N * N;
    float* batchC = C + batchIdx * N * N;

    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < N && t * TILE_DIM + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = batchA[row * N + t * TILE_DIM + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_DIM + threadIdx.y < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = batchB[(t * TILE_DIM + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        batchC[row * N + col] = value;
}

/**
 * @brief Host launcher for batched matrix multiplication.
 * 
 * @param A Pointer to host array A (batch × N × N)
 * @param B Pointer to host array B (batch × N × N)
 * @param C Pointer to host array C (batch × N × N)
 * @param N Dimension of each matrix
 * @param batch Number of matrices in the batch
 */
void batchedMatrixMul(const float* A, const float* B, float* C, int N, int batch) {
    float *dA, *dB, *dC;
    size_t size = (size_t)batch * N * N * sizeof(float);

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM,
                (N + TILE_DIM - 1) / TILE_DIM,
                batch);  // one z-dimension per matrix
    batchedMatrixMulKernel<<<blocks, threads>>>(dA, dB, dC, N, batch);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
