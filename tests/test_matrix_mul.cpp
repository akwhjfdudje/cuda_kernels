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

TEST(MatrixMulKernel, BasicCorrectness) {
    int N = 8;
    std::vector<float> A(N * N), B(N * N), C(N * N), C_ref(N * N);

    // Initialize matrices with a simple pattern
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>((i % N) + 1);
        B[i] = static_cast<float>((i / N) + 1);
    }

    // Compute on GPU
    matrixMul(A.data(), B.data(), C.data(), N);

    // Compute reference result on CPU
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C_ref[row * N + col] = sum;
        }
    }

    // Compare GPU result vs CPU reference
    for (int i = 0; i < N * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-3f) << "Mismatch at index " << i;
    }
}

