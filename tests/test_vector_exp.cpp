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

TEST(VectorExpKernel, SingleElement) {
    std::vector<float> A = {1.0f}, B(1);
    vectorExp(A.data(), B.data(), 1);
    EXPECT_NEAR(B[0], expf(1.0f), 1e-5);
}

TEST(VectorExpKernel, ZeroInput) {
    int N = 100;
    std::vector<float> A(N, 0.0f), B(N);
    vectorExp(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(B[i], 1.0f, 1e-5);
    }
}

TEST(VectorExpKernel, NegativeValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = -static_cast<float>(i);
    }
    vectorExp(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = expf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorExpKernel, SmallValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = -10.0f + 0.1f * static_cast<float>(i);
    }
    vectorExp(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = expf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorExpKernel, LargeValues) {
    int N = 50;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }
    vectorExp(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = expf(A[i]);
        float tol = 1e-3f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorExpKernel, FractionalValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 0.1f * static_cast<float>(i);
    }
    vectorExp(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = expf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorExpKernel, IdentityProperty) {
    int N = 100;
    std::vector<float> A(N, 0.0f), B(N);
    vectorExp(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(B[i], 1.0f, 1e-5);
    }
}
