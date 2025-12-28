#include <gtest/gtest.h>
#include "alg/mcts.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

/**
 * @brief CPU reference implementation for a basic MCTS rollout simulation.
 *        Uses simple random rollouts to estimate node values.
 */
static void cpuMCTS(
    std::vector<MCTSNode>& nodes,
    int num_nodes,
    int rollouts_per_node)
{
    // Simple CPU-based simulation for comparison
    for (int i = 0; i < num_nodes; ++i) {
        float total_reward = 0.0f;
        int total_visits = 0;
        for (int j = 0; j < rollouts_per_node; ++j) {
            // Simple random reward between -1 and 1
            total_reward += (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            total_visits++;
        }
        // Update node value and visit count
        nodes[i].value += total_reward;
        nodes[i].visit_count += total_visits;
    }
}

/**
 * @brief Test for MCTS kernel with small number of nodes and rollouts.
 */
TEST(MCTSTest, SmallNodes) {
    int num_nodes = 10;
    int rollouts_per_node = 100;

    // Initialize MCTS nodes
    std::vector<MCTSNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes[i].value = 0.0f;
        nodes[i].visit_count = 0;
    }

    // Run MCTS on GPU
    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);

    // Run CPU reference
    std::vector<MCTSNode> ref_nodes = nodes;
    cpuMCTS(ref_nodes, num_nodes, rollouts_per_node);

    // Compare results
    for (int i = 0; i < num_nodes; ++i) {
        float value_diff = std::fabs(nodes[i].value - ref_nodes[i].value);
        float visit_count_diff = std::abs(nodes[i].visit_count - ref_nodes[i].visit_count);

        // Check if the difference is within an acceptable tolerance
        EXPECT_LT(value_diff, 0.5f) << "Mismatch in value at node " << i;
        EXPECT_LT(visit_count_diff, 5) << "Mismatch in visit count at node " << i;
    }
}

/**
 * @brief Test for MCTS kernel with larger number of nodes and rollouts.
 */
TEST(MCTSTest, LargeNodes) {
    int num_nodes = 1000;
    int rollouts_per_node = 50;

    // Initialize MCTS nodes
    std::vector<MCTSNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes[i].value = 0.0f;
        nodes[i].visit_count = 0;
    }

    // Run MCTS on GPU
    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);

    // Run CPU reference
    std::vector<MCTSNode> ref_nodes = nodes;
    cpuMCTS(ref_nodes, num_nodes, rollouts_per_node);

    // Compare results
    for (int i = 0; i < num_nodes; ++i) {
        float value_diff = std::fabs(nodes[i].value - ref_nodes[i].value);
        float visit_count_diff = std::abs(nodes[i].visit_count - ref_nodes[i].visit_count);

        // Check if the difference is within an acceptable tolerance
        EXPECT_LT(value_diff, 0.5f) << "Mismatch in value at node " << i;
        EXPECT_LT(visit_count_diff, 5) << "Mismatch in visit count at node " << i;
    }
}

/**
 * @brief Test for MCTS kernel with extreme values for rollouts.
 */
TEST(MCTSTest, ExtremeRollouts) {
    int num_nodes = 5;
    int rollouts_per_node = 1000;

    // Initialize MCTS nodes with larger values
    std::vector<MCTSNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes[i].value = 0.0f;
        nodes[i].visit_count = 0;
    }

    // Run MCTS on GPU
    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);

    // Run CPU reference
    std::vector<MCTSNode> ref_nodes = nodes;
    cpuMCTS(ref_nodes, num_nodes, rollouts_per_node);

    // Compare results
    for (int i = 0; i < num_nodes; ++i) {
        float value_diff = std::fabs(nodes[i].value - ref_nodes[i].value);
        float visit_count_diff = std::abs(nodes[i].visit_count - ref_nodes[i].visit_count);

        // Allow a slightly larger tolerance for extreme rollouts
        EXPECT_LT(value_diff, 1.0f) << "Mismatch in value at node " << i;
        EXPECT_LT(visit_count_diff, 10) << "Mismatch in visit count at node " << i;
    }
}
