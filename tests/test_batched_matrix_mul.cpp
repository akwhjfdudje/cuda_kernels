#include <gtest/gtest.h>
#include "linalg/linalg.cuh"
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @brief CPU reference implementation for batched matrix multiplication:
 *        C[b] = A[b] (M×K)  ×  B[b] (K×N)
 */
static void cpuBatchedMatMul(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        int M, int K, int N,
        int batch)
{
    for (int b = 0; b < batch; ++b) {
        const float* A_b = &A[b * M * K];
        const float* B_b = &B[b * K * N];
        float*       C_b = &C[b * M * N];

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {

                float sum = 0.0f;
                for (int t = 0; t < K; ++t)
                    sum += A_b[i * K + t] * B_b[t * N + j];

                C_b[i * N + j] = sum;
            }
        }
    }
}

/**
 * @brief Small test: non-square and multi-batch.
 */
TEST(BatchedMatrixMulKernel, SmallBatch) {

    int M = 8;
    int K = 5;
    int N = 6;
    int batch = 4;

    std::vector<float> A(batch * M * K);
    std::vector<float> B(batch * K * N);
    std::vector<float> C(batch * M * N);
    std::vector<float> C_ref(batch * M * N);

    // Initialize input matrices
    for (int i = 0; i < batch * M * K; ++i)
        A[i] = static_cast<float>((i * 7) % 11);

    for (int i = 0; i < batch * K * N; ++i)
        B[i] = static_cast<float>((i * 3) % 13);

    // Run CUDA kernel
    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);

    // Run reference CPU implementation
    cpuBatchedMatMul(A, B, C_ref, M, K, N, batch);

    // Compare all results
    for (int i = 0; i < batch * M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Moderate batch test, more realistic transformer-like sizes.
 */
TEST(BatchedMatrixMulKernel, ModerateBatch) {

    int M = 32;   // sequence length or output dimension
    int K = 16;   // head dimension
    int N = 24;   // model dimension or next projection
    int batch = 8;

    std::vector<float> A(batch * M * K);
    std::vector<float> B(batch * K * N);
    std::vector<float> C(batch * M * N);
    std::vector<float> C_ref(batch * M * N);

    // Fill with small random-ish values
    for (int i = 0; i < batch * M * K; ++i)
        A[i] = 0.1f * (i % 10);

    for (int i = 0; i < batch * K * N; ++i)
        B[i] = 0.1f * ((i + 7) % 10);

    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);
    cpuBatchedMatMul(A, B, C_ref, M, K, N, batch);

    for (int i = 0; i < batch * M * N; ++i) {
        float expected = C_ref[i];
        float tol = 1e-3f * std::max(1.0f, (float)fabs(expected));
        EXPECT_NEAR(C[i], expected, tol) << "Mismatch at index " << i;
    }
}

TEST(BatchedMatrixMulKernel, SingleBatch) {
    int M = 4, K = 3, N = 5, batch = 1;
    std::vector<float> A(batch * M * K), B(batch * K * N), C(batch * M * N), C_ref(batch * M * N);
    
    for (int i = 0; i < batch * M * K; ++i) {
        A[i] = static_cast<float>(i);
    }
    for (int i = 0; i < batch * K * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);
    cpuBatchedMatMul(A, B, C_ref, M, K, N, batch);
    
    for (int i = 0; i < batch * M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }
}

TEST(BatchedMatrixMulKernel, IdentityMatrices) {
    int M = 4, K = 4, N = 4, batch = 3;
    std::vector<float> A(batch * M * K), B(batch * K * N), C(batch * M * N), C_ref(batch * M * N);
    
    // Create identity matrices for A
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                A[b * M * K + i * K + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    // Fill B with values
    for (int i = 0; i < batch * K * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);
    
    // A * B should equal B when A is identity
    for (int i = 0; i < batch * K * N; ++i) {
        EXPECT_NEAR(C[i], B[i], 1e-5f);
    }
}

TEST(BatchedMatrixMulKernel, ZeroMatrices) {
    int M = 4, K = 3, N = 5, batch = 2;
    std::vector<float> A(batch * M * K, 0.0f), B(batch * K * N), C(batch * M * N);
    
    for (int i = 0; i < batch * K * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);
    
    for (int i = 0; i < batch * M * N; ++i) {
        EXPECT_NEAR(C[i], 0.0f, 1e-5f);
    }
}

TEST(BatchedMatrixMulKernel, NegativeValues) {
    int M = 3, K = 2, N = 3, batch = 2;
    std::vector<float> A(batch * M * K), B(batch * K * N), C(batch * M * N), C_ref(batch * M * N);
    
    for (int i = 0; i < batch * M * K; ++i) {
        A[i] = -static_cast<float>(i);
    }
    for (int i = 0; i < batch * K * N; ++i) {
        B[i] = static_cast<float>(i);
    }
    
    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);
    cpuBatchedMatMul(A, B, C_ref, M, K, N, batch);
    
    for (int i = 0; i < batch * M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }
}

TEST(BatchedMatrixMulKernel, LargeBatch) {
    int M = 8, K = 4, N = 8, batch = 32;
    std::vector<float> A(batch * M * K), B(batch * K * N), C(batch * M * N);
    
    for (int i = 0; i < batch * M * K; ++i) {
        A[i] = 0.1f * static_cast<float>(i % 10);
    }
    for (int i = 0; i < batch * K * N; ++i) {
        B[i] = 0.1f * static_cast<float>((i + 7) % 10);
    }
    
    batchedMatrixMul(A.data(), B.data(), C.data(), M, K, N, batch);
    
    for (int i = 0; i < batch * M * N; ++i) {
        EXPECT_TRUE(std::isfinite(C[i]));
    }
}
