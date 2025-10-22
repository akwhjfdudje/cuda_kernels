/**
 * @file fused/fused_sdpa.cu
 * @brief Fused batched scaled dot-product attention kernel.
 *
 * Computes:
 *   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
 * in a fully parallelized way.
 *
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "fused/fused.cuh"
#include <cassert>
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
 * @brief Fused kernel: one block per query row.
 *
 * @param Q       Queries: [B, N, Dk]
 * @param K       Keys:    [B, N, Dk]
 * @param V       Values:  [B, N, Dv]
 * @param Output  Result:  [B, N, Dv]
 * @param batch   Number of batches
 * @param N       Sequence length
 * @param d_k     Key dimension
 * @param d_v     Value dimension
 */
__global__ void fusedScaledDotProductAttentionKernel(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v)
{
    int b = blockIdx.z;                   // batch index
    int q_row = blockIdx.y * blockDim.y + threadIdx.y; // query row
    int v_col = blockIdx.x * blockDim.x + threadIdx.x; // output column

    if (q_row >= N || v_col >= d_v || b >= batch) return;

    extern __shared__ float shmem[];      // shared memory for Q row
    float* q_vec = shmem + threadIdx.y * d_k;

    // Load Q row into registers / shared memory
    for (int i = threadIdx.x; i < d_k; i += blockDim.x)
        q_vec[i] = Q[b * N * d_k + q_row * d_k + i];

    __syncthreads();

    float max_val = -FLT_MAX;

    // Find max score
    for (int k_row = 0; k_row < N; ++k_row) {
        float score = 0.0f;
        for (int i = 0; i < d_k; ++i)
            score += q_vec[i] * K[b * N * d_k + k_row * d_k + i];
        score /= sqrtf((float)d_k);
        max_val = fmaxf(max_val, score);
    }

    // Compute softmax-weighted sum
    float sum_exp = 0.0f;
    float acc = 0.0f; // per-output dimension
    for (int j = 0; j < d_v; ++j) acc = 0.0f; // zero accumulator

    for (int k_row = 0; k_row < N; ++k_row) {
        float score = 0.0f;
        for (int i = 0; i < d_k; ++i)
            score += q_vec[i] * K[b * N * d_k + k_row * d_k + i];
        score /= sqrtf((float)d_k);

        float exp_score = expf(score - max_val);
        sum_exp += exp_score;

        for (int j = threadIdx.x; j < d_v; j += blockDim.x) {
            acc = exp_score * V[b * N * d_v + k_row * d_v + j];
            Output[b * N * d_v + q_row * d_v + j] += acc;
        }
    }

    // Normalize output
    for (int j = threadIdx.x; j < d_v; j += blockDim.x)
        Output[b * N * d_v + q_row * d_v + j] /= sum_exp;
}

/**
 * @brief Host launcher for fused scaled dot-product attention.
 */
void fusedScaledDotProductAttention(
    const float* Q, const float* K, const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v)
{
    float *d_Q, *d_K, *d_V, *d_out;
    size_t size = batch * N * d_v * sizeof(float);

    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_Q, Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);

    dim3 threads(32, 8); // 32 threads per row, 8 rows per block
    dim3 blocks((d_v + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y,
                batch);

    size_t shared_size = threads.y * d_k * sizeof(float);

    fusedScaledDotProductAttentionKernel<<<blocks, threads, shared_size>>>(d_Q, d_K, d_V, d_out, batch, N, d_k, d_v);

    cudaMemcpy(Output, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);

    cudaDeviceSynchronize();
}

