#include <gtest/gtest.h>
#include "linalg/linalg.cuh"
#include <vector>

TEST(MatrixMulKernel, BasicMultiplication) {
    int N = 16;
    std::vector<float> A(N*N), B(N*N), C(N*N);

    for (int i = 0; i < N*N; ++i) {
        A[i] = (i % N) * 0.1f;
        B[i] = (i % N) * 0.2f;
    }

    matrixMul(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N*N; ++i) {
        EXPECT_TRUE(std::isfinite(C[i]));
    }
}
