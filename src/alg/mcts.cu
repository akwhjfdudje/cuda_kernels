/**
 * @file alg/mcts.cu
 * @brief Generic parallel Monte Carlo Tree Search (MCTS) rollouts kernel.
 */

#include <cuda_runtime.h>
#include "alg/alg.cuh"

// Number of threads per block in each dimension (adjust as needed)
#define TILE_DIM 16  

/**
 * @brief Simulates a random rollout for MCTS (placeholder for actual game logic).
 *
 * @return Random reward between -1 and 1
 */
__device__ float simulate_rollout() {
    // Simple random reward between -1 and 1 for demonstration purposes
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

/**
 * @brief Parallel MCTS rollouts kernel to update nodes based on random simulations.
 *
 * input: A list of MCTS nodes, each with value and visit count.
 * rollouts_per_node: The number of rollouts to perform per node in parallel.
 *
 * @param nodes Pointer to MCTS nodes array
 * @param num_nodes Total number of nodes in the MCTS tree
 * @param rollouts_per_node Number of rollouts per node to be simulated
 */
__global__ void MCTSRolloutsKernel(
    MCTSNode *nodes,        // MCTS nodes array
    int num_nodes,          // Total number of nodes
    int rollouts_per_node   // Number of rollouts per node
) {
    __shared__ float tile_value[TILE_DIM];  // Shared memory for holding partial values
    __shared__ int tile_visit_count[TILE_DIM];  // Shared memory for holding partial visit counts

    int idx = blockIdx.x * TILE_DIM + threadIdx.x;

    // Ensure thread index is within bounds
    if (idx >= num_nodes) return;

    // Access the node for this thread
    MCTSNode *node = &nodes[idx];

    // Perform rollouts for this node
    float total_reward = 0.0f;
    int total_visits = 0;
    for (int i = 0; i < rollouts_per_node; i++) {
        total_reward += simulate_rollout();
        total_visits++;
    }

    // Store the partial results in shared memory
    tile_value[threadIdx.x] = total_reward;
    tile_visit_count[threadIdx.x] = total_visits;

    __syncthreads();  // Synchronize threads within block

    // Aggregate results from all threads in the block (if necessary)
    if (threadIdx.x == 0) {
        float block_total_reward = 0.0f;
        int block_total_visits = 0;
        for (int i = 0; i < TILE_DIM; i++) {
            block_total_reward += tile_value[i];
            block_total_visits += tile_visit_count[i];
        }

        // Update node value and visit count using atomic operations
        atomicAdd(&node->value, block_total_reward);
        atomicAdd(&node->visit_count, block_total_visits);
    }
}

/**
 * @brief Host launcher for parallel MCTS rollouts.
 *
 * @param nodes Array of MCTS nodes
 * @param num_nodes Total number of nodes in the tree
 * @param rollouts_per_node Number of rollouts per node
 */
extern "C" CUDA_KERNELS_API
void MCTSRollouts(
    MCTSNode *nodes,          // Array of MCTS nodes (on host)
    int num_nodes,            // Total number of nodes
    int rollouts_per_node     // Number of rollouts per node
) {
    MCTSNode *d_nodes = nullptr;
    size_t nodes_size = (size_t)num_nodes * sizeof(MCTSNode);

    cudaMalloc(&d_nodes, nodes_size);
    cudaMemcpy(d_nodes, nodes, nodes_size, cudaMemcpyHostToDevice);

    int block_size = TILE_DIM;  // Threads per block in each dimension
    int grid_size = (num_nodes + TILE_DIM - 1) / TILE_DIM;  // Number of blocks

    MCTSRolloutsKernel<<<grid_size, block_size>>>(d_nodes, num_nodes, rollouts_per_node);

    cudaMemcpy(nodes, d_nodes, nodes_size, cudaMemcpyDeviceToHost);
    cudaFree(d_nodes);
}
