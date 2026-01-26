#include <gtest/gtest.h>
#include "noise/noise.cuh"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

/**
 * @brief Test that noise generation produces values within the specified range.
 */
TEST(GenerateNoise, RangeCheck) {
    int N = 10000;
    std::vector<float> output(N);
    float min_val = -1.0f;
    float max_val = 1.0f;
    unsigned int seed = 42u;
    
    generateNoise(output.data(), N, min_val, max_val, seed);
    
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(output[i], min_val) << "Value at index " << i << " is below minimum";
        EXPECT_LT(output[i], max_val) << "Value at index " << i << " is at or above maximum";
    }
}

/**
 * @brief Test that noise generation is deterministic with the same seed.
 */
TEST(GenerateNoise, Deterministic) {
    int N = 1000;
    std::vector<float> output1(N);
    std::vector<float> output2(N);
    float min_val = 0.0f;
    float max_val = 1.0f;
    unsigned int seed = 12345u;
    
    generateNoise(output1.data(), N, min_val, max_val, seed);
    generateNoise(output2.data(), N, min_val, max_val, seed);
    
    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(output1[i], output2[i]) << "Values differ at index " << i;
    }
}

/**
 * @brief Test that different seeds produce different noise patterns.
 */
TEST(GenerateNoise, DifferentSeeds) {
    int N = 1000;
    std::vector<float> output1(N);
    std::vector<float> output2(N);
    float min_val = 0.0f;
    float max_val = 1.0f;
    
    generateNoise(output1.data(), N, min_val, max_val, 100u);
    generateNoise(output2.data(), N, min_val, max_val, 200u);
    
    // Check that at least some values are different
    bool all_same = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(output1[i] - output2[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Different seeds produced identical output";
}

/**
 * @brief Test uniform distribution properties of the generated noise.
 * 
 * This test checks that the mean of the generated values is approximately
 * at the center of the range, which is a basic property of uniform distribution.
 */
TEST(GenerateNoise, UniformDistribution) {
    int N = 100000;
    std::vector<float> output(N);
    float min_val = 0.0f;
    float max_val = 1.0f;
    unsigned int seed = 999u;
    
    generateNoise(output.data(), N, min_val, max_val, seed);
    
    // Calculate mean
    float sum = std::accumulate(output.begin(), output.end(), 0.0f);
    float mean = sum / N;
    float expected_mean = (min_val + max_val) / 2.0f;
    
    // For a uniform distribution, mean should be close to the center
    // Allow 5% tolerance for statistical variance
    EXPECT_NEAR(mean, expected_mean, 0.05f) << "Mean deviates too much from expected center";
}

/**
 * @brief Test noise generation with different ranges.
 */
TEST(GenerateNoise, DifferentRanges) {
    int N = 1000;
    std::vector<float> output(N);
    
    // Test positive range
    generateNoise(output.data(), N, 10.0f, 20.0f, 1u);
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(output[i], 10.0f);
        EXPECT_LT(output[i], 20.0f);
    }
    
    // Test negative range
    generateNoise(output.data(), N, -5.0f, -1.0f, 2u);
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(output[i], -5.0f);
        EXPECT_LT(output[i], -1.0f);
    }
    
    // Test range spanning zero
    generateNoise(output.data(), N, -10.0f, 10.0f, 3u);
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(output[i], -10.0f);
        EXPECT_LT(output[i], 10.0f);
    }
}

/**
 * @brief Test noise generation with large arrays.
 */
TEST(GenerateNoise, LargeArrays) {
    int N = 1 << 20; // 1M elements
    std::vector<float> output(N);
    float min_val = -100.0f;
    float max_val = 100.0f;
    unsigned int seed = 456u;
    
    generateNoise(output.data(), N, min_val, max_val, seed);
    
    // Check a sample of values
    for (int i = 0; i < 1000; ++i) {
        int idx = i * (N / 1000);
        EXPECT_GE(output[idx], min_val);
        EXPECT_LT(output[idx], max_val);
    }
}

/**
 * @brief Test edge case: single element.
 */
TEST(GenerateNoise, SingleElement) {
    std::vector<float> output(1);
    float min_val = 0.0f;
    float max_val = 1.0f;
    unsigned int seed = 789u;
    
    generateNoise(output.data(), 1, min_val, max_val, seed);
    
    EXPECT_GE(output[0], min_val);
    EXPECT_LT(output[0], max_val);
}

/**
 * @brief Test edge case: empty array (should not crash).
 */
TEST(GenerateNoise, EmptyArray) {
    std::vector<float> output(1);
    unsigned int seed = 111u;
    
    // Should not crash or produce errors
    generateNoise(output.data(), 0, 0.0f, 1.0f, seed);
}

/**
 * @brief Test that all values in a large sample are unique (or at least well-distributed).
 * 
 * This is a basic check that the RNG is producing varied output.
 */
TEST(GenerateNoise, ValueDiversity) {
    int N = 10000;
    std::vector<float> output(N);
    float min_val = 0.0f;
    float max_val = 1.0f;
    unsigned int seed = 555u;
    
    generateNoise(output.data(), N, min_val, max_val, seed);
    
    // Count unique values (with some tolerance for floating point)
    std::sort(output.begin(), output.end());
    int unique_count = 1;
    for (int i = 1; i < N; ++i) {
        if (std::abs(output[i] - output[i-1]) > 1e-6f) {
            unique_count++;
        }
    }
    
    // For a good RNG, we should have many unique values
    // Allow at least 90% uniqueness
    float uniqueness_ratio = static_cast<float>(unique_count) / N;
    EXPECT_GT(uniqueness_ratio, 0.9f) << "Too many duplicate values in output";
}
