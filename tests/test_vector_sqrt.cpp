#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorSqrtKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    vectorSqrt(A.data(), B.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = sqrtf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorSqrtKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    vectorSqrt(A.data(), B.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = sqrtf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorSqrtKernel, SingleElement) {
    std::vector<float> A = {16.0f}, B(1);
    vectorSqrt(A.data(), B.data(), 1);
    EXPECT_NEAR(B[0], 4.0f, 1e-5);
}

TEST(VectorSqrtKernel, ZeroInput) {
    int N = 100;
    std::vector<float> A(N, 0.0f), B(N);
    vectorSqrt(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(B[i], 0.0f, 1e-5);
    }
}

TEST(VectorSqrtKernel, PerfectSquares) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>((i + 1) * (i + 1));
    }
    vectorSqrt(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(B[i], static_cast<float>(i + 1), 1e-4f);
    }
}

TEST(VectorSqrtKernel, SmallValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e-6f * static_cast<float>(i + 1);
    }
    vectorSqrt(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = sqrtf(A[i]);
        float tol = 1e-6f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorSqrtKernel, LargeValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e6f * static_cast<float>(i + 1);
    }
    vectorSqrt(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = sqrtf(A[i]);
        float tol = 1e-2f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorSqrtKernel, FractionalValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 0.1f + 0.01f * static_cast<float>(i);
    }
    vectorSqrt(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = sqrtf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}
