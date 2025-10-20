#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorLogKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    vectorLog(A.data(), B.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = logf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorLogKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    vectorLog(A.data(), B.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = logf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}
