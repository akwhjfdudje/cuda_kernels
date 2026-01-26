#include <gtest/gtest.h>
#include "reduce/reduce.cuh"
#include <vector>
#include <numeric>

TEST(ReduceSumKernel, SumArray) {
    int N = 512;
    std::vector<float> A(N);
    for (int i = 0; i < N; ++i) A[i] = 1.0f;

    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, N * 1.0f, 1e-5);
}

TEST(ReduceSumKernel, HugeArray) {
    int N = 1 << 15;
    std::vector<float> A(N);
    for (int i = 0; i < N; ++i) A[i] = 1.0f;

    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, N * 1.0f, 1e-5);
}

TEST(ReduceSumKernel, SingleElement) {
    std::vector<float> A = {42.5f};
    float result = reduceSum(A.data(), 1);
    EXPECT_NEAR(result, 42.5f, 1e-5);
}

TEST(ReduceSumKernel, ZeroValues) {
    int N = 1000;
    std::vector<float> A(N, 0.0f);
    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, 0.0f, 1e-5);
}

TEST(ReduceSumKernel, NegativeValues) {
    int N = 100;
    std::vector<float> A(N);
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        A[i] = -static_cast<float>(i + 1);
        expected += A[i];
    }
    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, expected, 1e-5);
}

TEST(ReduceSumKernel, MixedSigns) {
    int N = 100;
    std::vector<float> A(N);
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        A[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
        expected += A[i];
    }
    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, expected, 1e-5);
}

TEST(ReduceSumKernel, SequentialNumbers) {
    int N = 1000;
    std::vector<float> A(N);
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
        expected += A[i];
    }
    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, expected, 1e-3f);
}

TEST(ReduceSumKernel, LargeNumbers) {
    int N = 100;
    std::vector<float> A(N);
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        A[i] = 1e6f * static_cast<float>(i + 1);
        expected += A[i];
    }
    float result = reduceSum(A.data(), N);
    // For large numbers, use relative tolerance instead of absolute
    // Expected sum is ~5.05e9, so 1e-2 absolute tolerance is too strict
    float rel_tol = std::max(1e-3f * std::abs(expected), 1e4f); // 0.1% relative or 10k absolute
    EXPECT_NEAR(result, expected, rel_tol);
}

TEST(ReduceSumKernel, SmallNumbers) {
    int N = 1000;
    std::vector<float> A(N);
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        A[i] = 1e-6f * static_cast<float>(i + 1);
        expected += A[i];
    }
    float result = reduceSum(A.data(), N);
    // For small numbers, floating-point precision limits apply
    // Expected sum is ~0.0005, so 1e-8 absolute tolerance is too strict
    // Use relative tolerance: 0.1% relative or 1e-6 absolute (whichever is larger)
    float rel_tol = std::max(1e-3f * std::abs(expected), 1e-6f);
    EXPECT_NEAR(result, expected, rel_tol);
}

TEST(ReduceSumKernel, AlternatingPattern) {
    int N = 100;
    std::vector<float> A(N);
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        A[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        expected += A[i];
    }
    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, expected, 1e-5);
}
