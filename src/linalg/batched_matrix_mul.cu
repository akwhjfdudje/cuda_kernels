/**
 * @file linalg/batched_matrix_mul.cu
 * @brief Batched matrix multiplication for (batch, M, K) × (batch, K, N).
 */

#include <cuda_runtime.h>
#include "linalg/linalg.cuh"

#define TILE_DIM 16

/**
 * @brief Computes C = A @ B for each batch entry.
 *
 * A: [batch, M, K]
 * B: [batch, K, N]
 * C: [batch, M, N]
 *
 * @param A Pointer to input A
 * @param B Pointer to input B
 * @param C Pointer to output C
 * @param M Rows in A and C
 * @param K Inner dimension
 * @param N Columns in B and C
 * @param batch Number of matrices in batch
 */
__global__ void batchedMatrixMulKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N, int batch)
{
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int batchIdx = blockIdx.z;

    // Output C[row, col]
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float value = 0.f;

    const float* batchA = A + batchIdx * (M * K);
    const float* batchB = B + batchIdx * (K * N);
    float* batchC       = C + batchIdx * (M * N);

    // Iterate tiles of K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // Load A tile: [M × K]
        int A_row = row;
        int A_col = t * TILE_DIM + threadIdx.x;

        tileA[threadIdx.y][threadIdx.x] =
            (A_row < M && A_col < K)
            ? batchA[A_row * K + A_col]
            : 0.f;

        // Load B tile: [K × N]
        int B_row = t * TILE_DIM + threadIdx.y;
        int B_col = col;

        tileB[threadIdx.y][threadIdx.x] =
            (B_row < K && B_col < N)
            ? batchB[B_row * N + B_col]
            : 0.f;

        __syncthreads();

        // Multiply tile
        for (int k = 0; k < TILE_DIM; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    // Write output
    if (row < M && col < N)
        batchC[row * N + col] = value;
}


/**
 * @brief Host launcher for batched matrix multiplication.
 */
extern "C" CUDA_KERNELS_API
void batchedMatrixMul(
    const float* A, const float* B, float* C,
    int M, int K, int N, int batch)
{
    float *dA, *dB, *dC;
    size_t sizeA = (size_t)batch * M * K * sizeof(float);
    size_t sizeB = (size_t)batch * K * N * sizeof(float);
    size_t sizeC = (size_t)batch * M * N * sizeof(float);

    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM,
        batch
    );

    batchedMatrixMulKernel<<<blocks, threads>>>(dA, dB, dC, M, K, N, batch);

    cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
