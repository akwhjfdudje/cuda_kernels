#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorMulKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorMul(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] * B[i], 1e-5);
    }
}

TEST(VectorMulKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    vectorMul(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] * B[i], 1e-5);
    }
}

TEST(VectorMulKernel, SingleElement) {
    std::vector<float> A = {5.5f}, B = {3.2f}, C(1);
    vectorMul(A.data(), B.data(), C.data(), 1);
    EXPECT_NEAR(C[0], 17.6f, 1e-5);
}

TEST(VectorMulKernel, ZeroValues) {
    int N = 100;
    std::vector<float> A(N, 0.0f), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        B[i] = static_cast<float>(i);
    }
    vectorMul(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 0.0f, 1e-5);
    }
}

TEST(VectorMulKernel, NegativeValues) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = -static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1);
    }
    vectorMul(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] * B[i], 1e-5);
    }
}

TEST(VectorMulKernel, IdentityProperty) {
    int N = 256;
    std::vector<float> A(N), B(N, 1.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }
    vectorMul(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

TEST(VectorMulKernel, SmallNumbers) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e-6f * static_cast<float>(i + 1);
        B[i] = 1e-6f * static_cast<float>(N - i);
    }
    vectorMul(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] * B[i], 1e-10f);
    }
}

TEST(VectorMulKernel, MixedSigns) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1);
    }
    vectorMul(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] * B[i], 1e-5);
    }
}
