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

TEST(VectorLogKernel, SingleElement) {
    std::vector<float> A = {2.71828f}, B(1);
    vectorLog(A.data(), B.data(), 1);
    EXPECT_NEAR(B[0], 1.0f, 1e-4f);
}

TEST(VectorLogKernel, OneInput) {
    int N = 100;
    std::vector<float> A(N, 1.0f), B(N);
    vectorLog(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(B[i], 0.0f, 1e-5);
    }
}

TEST(VectorLogKernel, PositiveValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    vectorLog(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = logf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorLogKernel, SmallValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 0.001f + 0.01f * static_cast<float>(i);
    }
    vectorLog(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = logf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorLogKernel, LargeValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e6f * static_cast<float>(i + 1);
    }
    vectorLog(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = logf(A[i]);
        float tol = 1e-4f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}

TEST(VectorLogKernel, PowersOfE) {
    int N = 10;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = expf(static_cast<float>(i));
    }
    vectorLog(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(B[i], static_cast<float>(i), 1e-4f);
    }
}

TEST(VectorLogKernel, FractionalValues) {
    int N = 100;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 0.1f + 0.1f * static_cast<float>(i);
    }
    vectorLog(A.data(), B.data(), N);
    for (int i = 0; i < N; ++i) {
        float expected = logf(A[i]);
        float tol = 1e-5f * fmax(1.0f, fabs(expected));
        EXPECT_NEAR(B[i], expected, tol);
    }
}
