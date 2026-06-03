/**
 * @file fused/flash_attention.cu
 * @brief Streaming softmax attention kernel inspired by FlashAttention.
 *
 * Computes:
 *   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
 * without materializing the N x N attention matrix.
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "fused/fused.cuh"
#include "utils/utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kKeyTile = 32;

/**
 * @brief One block computes one query row and one tile of output value columns.
 *
 * Each thread owns one output column for a single query row. The softmax
 * denominator and output accumulator are updated online across K/V tiles.
 */
__global__ void flashAttentionKernel(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v)
{
    int v_col = blockIdx.x * blockDim.x + threadIdx.x;
    int q_row = blockIdx.y;
    int b = blockIdx.z;

    if (q_row >= N || b >= batch) return;

    extern __shared__ float q_vec[];

    for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
        q_vec[i] = Q[b * N * d_k + q_row * d_k + i];
    }
    __syncthreads();

    if (v_col >= d_v) return;

    const float scale = rsqrtf(static_cast<float>(d_k));
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (int tile_start = 0; tile_start < N; tile_start += kKeyTile) {
        int tile_end = (tile_start + kKeyTile < N) ? tile_start + kKeyTile : N;

        for (int k_row = tile_start; k_row < tile_end; ++k_row) {
            float score = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                score += q_vec[i] * K[b * N * d_k + k_row * d_k + i];
            }
            score *= scale;

            float next_max = fmaxf(row_max, score);
            float old_scale = (row_max == -FLT_MAX) ? 0.0f : expf(row_max - next_max);
            float score_scale = expf(score - next_max);

            acc = acc * old_scale + score_scale * V[b * N * d_v + k_row * d_v + v_col];
            row_sum = row_sum * old_scale + score_scale;
            row_max = next_max;
        }
    }

    Output[b * N * d_v + q_row * d_v + v_col] = acc / row_sum;
}

} // namespace

/**
 * @brief Host launcher for streaming FlashAttention-style scaled dot-product attention.
 */
extern "C" CUDA_KERNELS_API void flashAttention(
    const float* Q, const float* K, const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v)
{
    float *d_Q, *d_K, *d_V, *d_out;
    size_t q_size = static_cast<size_t>(batch) * N * d_k * sizeof(float);
    size_t k_size = static_cast<size_t>(batch) * N * d_k * sizeof(float);
    size_t v_size = static_cast<size_t>(batch) * N * d_v * sizeof(float);
    size_t out_size = static_cast<size_t>(batch) * N * d_v * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_Q, q_size));
    CHECK_CUDA(cudaMalloc(&d_K, k_size));
    CHECK_CUDA(cudaMalloc(&d_V, v_size));
    CHECK_CUDA(cudaMalloc(&d_out, out_size));

    CHECK_CUDA(cudaMemcpy(d_Q, Q, q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, K, k_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, V, v_size, cudaMemcpyHostToDevice));

    dim3 threads(kThreadsPerBlock);
    dim3 blocks((d_v + threads.x - 1) / threads.x, N, batch);
    size_t shared_size = static_cast<size_t>(d_k) * sizeof(float);

    flashAttentionKernel<<<blocks, threads, shared_size>>>(d_Q, d_K, d_V, d_out, batch, N, d_k, d_v);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(Output, d_out, out_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
}
