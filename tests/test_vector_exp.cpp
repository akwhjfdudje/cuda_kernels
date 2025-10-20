#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorExpKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    vectorExp(A.data(), B.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = expf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorExpKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    vectorExp(A.data(), B.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = expf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}
