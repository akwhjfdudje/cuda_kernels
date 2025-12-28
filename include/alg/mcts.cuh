#pragma once

struct MCTSNode {
    float value;        // Accumulated value (reward)
    int visit_count;    // Number of rollouts (visits)
};

extern "C" CUDA_KERNELS_API void MCTSRollouts(MCTSNode *nodes, int num_nodes, int rollouts_per_node);
