#include <gtest/gtest.h>
#include "arith/arith.cuh"
#include <vector>
#include <cmath>

TEST(VectorDivKernel, SmallArrays) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(N - i + 1);
    }

    vectorDiv(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-5);
    }
}

TEST(VectorDivKernel, LargeArrays) {
    int N = 1 << 15;
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(N - i + 1);
    }

    vectorDiv(A.data(), B.data(), C.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-5);
    }
}

TEST(VectorDivKernel, SingleElement) {
    std::vector<float> A = {10.0f}, B = {2.5f}, C(1);
    vectorDiv(A.data(), B.data(), C.data(), 1);
    EXPECT_NEAR(C[0], 4.0f, 1e-5);
}

TEST(VectorDivKernel, IdentityProperty) {
    int N = 256;
    std::vector<float> A(N), B(N, 1.0f), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    vectorDiv(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

TEST(VectorDivKernel, SelfDivision) {
    int N = 100;
    std::vector<float> A(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    vectorDiv(A.data(), A.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], 1.0f, 1e-5);
    }
}

TEST(VectorDivKernel, NegativeValues) {
    int N = 256;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = -static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1);
    }
    vectorDiv(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-5);
    }
}

TEST(VectorDivKernel, SmallNumbers) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e-6f * static_cast<float>(i + 1);
        B[i] = 1e-6f * static_cast<float>(N - i + 1);
    }
    vectorDiv(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-8f);
    }
}

TEST(VectorDivKernel, LargeNumbers) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = 1e6f * static_cast<float>(i + 1);
        B[i] = 1e6f * static_cast<float>(N - i + 1);
    }
    vectorDiv(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-2f);
    }
}

TEST(VectorDivKernel, MixedSigns) {
    int N = 100;
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1);
    }
    vectorDiv(A.data(), B.data(), C.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-5);
    }
}
