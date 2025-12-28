#pragma once

extern "C" CUDA_KERNELS_API void MCTSRollouts(MCTSNode *nodes, int num_nodes, int rollouts_per_node);
