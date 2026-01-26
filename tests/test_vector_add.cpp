#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorAddKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorAdd(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-5);
    }
}

TEST(VectorAddKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorAdd(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-5);
    }
}

TEST(VectorAddKernel, SingleElement) {
    std::vector<float> A = {5.5f}, B = {3.2f}, C(1);
    vectorAdd(A.data(), B.data(), C.data(), 1);
    EXPECT_NEAR(C[0], 8.7f, 1e-5);
}

TEST(VectorAddKernel, ZeroValues) {
    int N = 100;
    std::vector<float> A(N, 0.0f), B(N, 0.0f), C(N);
    vectorAdd(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 0.0f, 1e-5);
    }
}

TEST(VectorAddKernel, NegativeValues) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = -static_cast<float>(i);
        B[i] = -static_cast<float>(N - i);
    }
    vectorAdd(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-5);
    }
}

TEST(VectorAddKernel, MixedSigns) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
        B[i] = (i % 3 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }
    vectorAdd(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-5);
    }
}

TEST(VectorAddKernel, LargeNumbers) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e6f * static_cast<float>(i);
        B[i] = 1e6f * static_cast<float>(N - i);
    }
    vectorAdd(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-2f);
    }
}

TEST(VectorAddKernel, SmallNumbers) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e-6f * static_cast<float>(i);
        B[i] = 1e-6f * static_cast<float>(N - i);
    }
    vectorAdd(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] + B[i], 1e-8f);
    }
}

TEST(VectorAddKernel, IdentityProperty) {
    int N = 256;
    std::vector<float> A(N), B(N, 0.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }
    vectorAdd(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}
