/**
 * @file alg/mcts.cu
 * @brief Generic parallel Monte Carlo Tree Search (MCTS) rollouts, purely random.
 *
 */

#include <cuda_runtime.h>
#include <cstdio>
#include "alg/alg.cuh"

#define MAX_ROLLOUTS 256

__device__ float lcg_random(unsigned int& state) {
    //
    // Linear Congruential Generator (LCG)
    //
    // This implements a fast, deterministic, device-safe RNG
    // using the LCG algorithm. It works entirely on the GPU
    // without any global state, making it suitable here.
    //
    // Algorithm:
    //   state = (a*state + c) mod m
    //   - 'state' is the current RNG state (seed), passed by reference.
    //   - 'a' is the multiplier constant (1664525u), chosen for good statistical properties.
    //   - 'c' is the increment constant (1013904223u), also chosen for LCG quality.
    //   - 'm' is implicitly 2^32 due to 32-bit unsigned integer overflow.
    //
    // The resulting 'state' is then converted to a floating-point number in [0,1):
    //   - Mask the lower 24 bits with 0x00FFFFFF to avoid using all 32 bits.
    //   - Divide by 2^24 (0x01000000) to normalize to the range [0,1).
    //

    // Update RNG state using LCG formula
    state = state * 1664525u + 1013904223u;

    // Convert to [0,1) using lower 24 bits
    return (state & 0x00FFFFFF) / float(0x01000000);
}

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

