/**
 * @file alg/mcts.cu
 * @brief Generic parallel Monte Carlo Tree Search (MCTS) rollouts, purely random.
 *
 */

#include <cuda_runtime.h>
#include <cstdio>
#include "alg/alg.cuh"
#include "utils/utils.cuh"

#define MAX_ROLLOUTS 256

__global__ void MCTSRolloutsKernel(
    MCTSNode* nodes,
    int rollouts_per_node
) {
    int node_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= rollouts_per_node) return;

    __shared__ float rewards[MAX_ROLLOUTS];

    // Per-thread RNG seed
    unsigned int rng = (unsigned int)(node_idx * 9781u + tid * 6271u);

    // Simulate rollout
    float reward = 2.0f * lcg_random(rng) - 1.0f;
    rewards[tid] = reward;

    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rewards[tid] += rewards[tid + stride];
        }
        __syncthreads();
    }

    // Accumulate into node
    if (tid == 0) {
        atomicAdd(&nodes[node_idx].value, rewards[0]);
        atomicAdd(&nodes[node_idx].visit_count, rollouts_per_node);
    }
}

extern "C" CUDA_KERNELS_API
void MCTSRollouts(
    MCTSNode* nodes,      // host nodes
    int num_nodes,
    int rollouts_per_node
) {
    if (rollouts_per_node > MAX_ROLLOUTS) {
        printf("rollouts_per_node exceeds MAX_ROLLOUTS\n");
        return;
    }

    MCTSNode* d_nodes = nullptr;
    size_t size = (size_t)num_nodes * sizeof(MCTSNode);

    cudaMalloc(&d_nodes, size);
    cudaMemcpy(d_nodes, nodes, size, cudaMemcpyHostToDevice);

    // Round threads to next power of two
    int threads = 1;
    while (threads < rollouts_per_node) threads <<= 1;

    MCTSRolloutsKernel<<<num_nodes, threads>>>(d_nodes, rollouts_per_node);

    cudaDeviceSynchronize();

    cudaMemcpy(nodes, d_nodes, size, cudaMemcpyDeviceToHost);
    cudaFree(d_nodes);
}

