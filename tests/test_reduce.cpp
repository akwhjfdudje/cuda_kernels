#include <gtest/gtest.h>
#include "reduce.cuh"
#include <vector>
#include <numeric>

TEST(ReduceKernel, SumArray) {
    int N = 512;
    std::vector<float> A(N);
    for (int i = 0; i < N; ++i) A[i] = 1.0f;

    float result = reduceSum(A.data(), N);
    EXPECT_NEAR(result, N * 1.0f, 1e-5);
}
