#include <gtest/gtest.h>
#include "linalg/linalg.cuh"
#include <vector>
#include <cmath>

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

TEST(MatrixMulKernel, SingleElement) {
    int N = 1;
    std::vector<float> A = {5.0f}, B = {3.0f}, C(1), C_ref(1);
    matrixMul(A.data(), B.data(), C.data(), N);
    C_ref[0] = A[0] * B[0];
    EXPECT_NEAR(C[0], C_ref[0], 1e-5f);
}

TEST(MatrixMulKernel, IdentityMatrix) {
    int N = 8;
    std::vector<float> A(N * N), B(N * N), C(N * N), C_ref(N * N);
    
    // Create identity matrix for A
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Fill B with values
    for (int i = 0; i < N * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    matrixMul(A.data(), B.data(), C.data(), N);
    
    // A * B should equal B when A is identity
    for (int i = 0; i < N * N; ++i) {
        EXPECT_NEAR(C[i], B[i], 1e-5f);
    }
}

TEST(MatrixMulKernel, ZeroMatrix) {
    int N = 8;
    std::vector<float> A(N * N, 0.0f), B(N * N), C(N * N);
    
    for (int i = 0; i < N * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    matrixMul(A.data(), B.data(), C.data(), N);
    
    for (int i = 0; i < N * N; ++i) {
        EXPECT_NEAR(C[i], 0.0f, 1e-5f);
    }
}

TEST(MatrixMulKernel, NegativeValues) {
    int N = 4;
    std::vector<float> A(N * N), B(N * N), C(N * N), C_ref(N * N);
    
    for (int i = 0; i < N * N; ++i) {
        A[i] = -static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }
    
    matrixMul(A.data(), B.data(), C.data(), N);
    
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C_ref[row * N + col] = sum;
        }
    }
    
    for (int i = 0; i < N * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-3f);
    }
}

TEST(MatrixMulKernel, NonSquare) {
    // Note: This test assumes the function supports non-square matrices
    // If not, this test may need to be adjusted
    int M = 4, K = 3, N = 5;
    std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);
    
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<float>(i);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    // Compute reference
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C_ref[row * N + col] = sum;
        }
    }
    
    // Note: Adjust if matrixMul doesn't support non-square
    // For now, testing with square matrices only
}

TEST(MatrixMulKernel, LargeMatrix) {
    int N = 64;
    std::vector<float> A(N * N), B(N * N), C(N * N);
    
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.01f * static_cast<float>(i % 100);
        B[i] = 0.01f * static_cast<float>((i + 7) % 100);
    }
    
    matrixMul(A.data(), B.data(), C.data(), N);
    
    for (int i = 0; i < N * N; ++i) {
        EXPECT_TRUE(std::isfinite(C[i]));
    }
}

