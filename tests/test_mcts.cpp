#include <gtest/gtest.h>
#include "alg/alg.cuh"
#include <vector>
#include <cmath>
#include <algorithm>

TEST(MCTSRolloutsKernel, SmallNodes) {
    int num_nodes = 16;
    int rollouts_per_node = 32;

    std::vector<MCTSNode> nodes(num_nodes);

    for (auto& n : nodes) {
        n.value = 0.0f;
        n.visit_count = 0;
    }

    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);

    for (int i = 0; i < num_nodes; ++i) {
        EXPECT_EQ(nodes[i].visit_count, rollouts_per_node)
            << "Incorrect visit count at node " << i;

        EXPECT_TRUE(std::isfinite(nodes[i].value))
            << "Non-finite value at node " << i;

        EXPECT_GE(nodes[i].value, -rollouts_per_node)
            << "Value too small at node " << i;

        EXPECT_LE(nodes[i].value, rollouts_per_node)
            << "Value too large at node " << i;
    }
}

/**
 * @brief Larger stress test for stability.
 */
TEST(MCTSRolloutsKernel, ModerateNodes) {
    int num_nodes = 512;
    int rollouts_per_node = 128;

    std::vector<MCTSNode> nodes(num_nodes);

    for (auto& n : nodes) {
        n.value = 0.0f;
        n.visit_count = 0;
    }

    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);

    for (int i = 0; i < num_nodes; ++i) {
        EXPECT_EQ(nodes[i].visit_count, rollouts_per_node);
        EXPECT_TRUE(std::isfinite(nodes[i].value));
        EXPECT_GE(nodes[i].value, -rollouts_per_node);
        EXPECT_LE(nodes[i].value, rollouts_per_node);
    }
}
