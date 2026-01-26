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

TEST(MCTSRolloutsKernel, SingleNode) {
    int num_nodes = 1;
    int rollouts_per_node = 16;
    
    std::vector<MCTSNode> nodes(num_nodes);
    nodes[0].value = 0.0f;
    nodes[0].visit_count = 0;
    
    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);
    
    EXPECT_EQ(nodes[0].visit_count, rollouts_per_node);
    EXPECT_TRUE(std::isfinite(nodes[0].value));
    EXPECT_GE(nodes[0].value, -rollouts_per_node);
    EXPECT_LE(nodes[0].value, rollouts_per_node);
}

TEST(MCTSRolloutsKernel, SingleRollout) {
    int num_nodes = 10;
    int rollouts_per_node = 1;
    
    std::vector<MCTSNode> nodes(num_nodes);
    for (auto& n : nodes) {
        n.value = 0.0f;
        n.visit_count = 0;
    }
    
    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);
    
    for (int i = 0; i < num_nodes; ++i) {
        EXPECT_EQ(nodes[i].visit_count, rollouts_per_node);
        EXPECT_TRUE(std::isfinite(nodes[i].value));
        EXPECT_GE(nodes[i].value, -1.0f);
        EXPECT_LE(nodes[i].value, 1.0f);
    }
}

TEST(MCTSRolloutsKernel, DeterministicSeeds) {
    int num_nodes = 4;
    int rollouts_per_node = 8;
    
    std::vector<MCTSNode> nodes1(num_nodes);
    std::vector<MCTSNode> nodes2(num_nodes);
    
    for (auto& n : nodes1) {
        n.value = 0.0f;
        n.visit_count = 0;
    }
    for (auto& n : nodes2) {
        n.value = 0.0f;
        n.visit_count = 0;
    }
    
    MCTSRollouts(nodes1.data(), num_nodes, rollouts_per_node);
    MCTSRollouts(nodes2.data(), num_nodes, rollouts_per_node);
    
    // Values should be deterministic (same seed pattern)
    for (int i = 0; i < num_nodes; ++i) {
        EXPECT_EQ(nodes1[i].visit_count, nodes2[i].visit_count);
        EXPECT_FLOAT_EQ(nodes1[i].value, nodes2[i].value);
    }
}

TEST(MCTSRolloutsKernel, LargeBatch) {
    int num_nodes = 1024;
    int rollouts_per_node = 64;
    
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

TEST(MCTSRolloutsKernel, PreInitializedNodes) {
    int num_nodes = 8;
    int rollouts_per_node = 16;
    
    std::vector<MCTSNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes[i].value = static_cast<float>(i);
        nodes[i].visit_count = i;
    }
    
    MCTSRollouts(nodes.data(), num_nodes, rollouts_per_node);
    
    // Visit counts should be updated
    for (int i = 0; i < num_nodes; ++i) {
        EXPECT_EQ(nodes[i].visit_count, i + rollouts_per_node);
        EXPECT_TRUE(std::isfinite(nodes[i].value));
    }
}
