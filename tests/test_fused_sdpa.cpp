#include <gtest/gtest.h>
#include "fused/fused.cuh"
#include <vector>
#include <float.h>
#include <math.h>

/**
 * @brief Reference CPU implementation of scaled dot-product attention.
 */
static void scaledDotProductAttentionCPU(
    const float* Q, const float* K, const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v)
{
    for (int b = 0; b < batch; ++b) {
        for (int q_row = 0; q_row < N; ++q_row) {
            std::vector<float> scores(N);
            float max_val = -FLT_MAX;

            // Compute Q K^T / sqrt(d_k)
            for (int k_row = 0; k_row < N; ++k_row) {
                float s = 0.0f;
                for (int i = 0; i < d_k; ++i)
                    s += Q[b * N * d_k + q_row * d_k + i] *
                         K[b * N * d_k + k_row * d_k + i];
                s /= sqrtf((float)d_k);
                scores[k_row] = s;
                if (s > max_val) max_val = s;
            }

            // Softmax normalization
            float sum_exp = 0.0f;
            for (int k_row = 0; k_row < N; ++k_row) {
                scores[k_row] = expf(scores[k_row] - max_val);
                sum_exp += scores[k_row];
            }
            for (int k_row = 0; k_row < N; ++k_row)
                scores[k_row] /= sum_exp;

            // Weighted sum V
            for (int j = 0; j < d_v; ++j) {
                float val = 0.0f;
                for (int k_row = 0; k_row < N; ++k_row)
                    val += scores[k_row] * V[b * N * d_v + k_row * d_v + j];
                Output[b * N * d_v + q_row * d_v + j] = val;
            }
        }
    }
}

/**
 * @brief Basic correctness test for fused scaled dot-product attention.
 */
TEST(FusedScaledDotProductAttention, BasicCorrectness) {
    const int B = 2, N = 4, Dk = 8, Dv = 8;
    std::vector<float> Q(B*N*Dk), K(B*N*Dk), V(B*N*Dv);
    std::vector<float> Out(B*N*Dv, 0.0f), Ref(B*N*Dv, 0.0f);

    // Fill Q, K, V with a small random pattern
    for (int i = 0; i < B*N*Dk; ++i)
        Q[i] = static_cast<float>((i % Dk) - Dk/2) / Dk;
    for (int i = 0; i < B*N*Dk; ++i)
        K[i] = static_cast<float>((i % Dk) - Dk/2) / Dk;
    for (int i = 0; i < B*N*Dv; ++i)
        V[i] = static_cast<float>((i % Dv) - Dv/2) / Dv;

    // Compute CPU reference
    scaledDotProductAttentionCPU(Q.data(), K.data(), V.data(), Ref.data(), B, N, Dk, Dv);

    // Compute GPU output
    fusedScaledDotProductAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);

    // Compare outputs
    for (int i = 0; i < B*N*Dv; ++i)
        EXPECT_NEAR(Out[i], Ref[i], 1e-3f) << "Mismatch at index " << i;

    // Optional: check row-wise softmax sum indirectly via output
    for (int b = 0; b < B; ++b) {
        for (int q_row = 0; q_row < N; ++q_row) {
            float row_sum = 0.0f;
            for (int j = 0; j < Dv; ++j)
                row_sum += Out[b*N*Dv + q_row*Dv + j];
            EXPECT_TRUE(std::isfinite(row_sum));
        }
    }
}

TEST(FusedScaledDotProductAttention, SingleBatch) {
    const int B = 1, N = 4, Dk = 8, Dv = 8;
    std::vector<float> Q(B*N*Dk), K(B*N*Dk), V(B*N*Dv);
    std::vector<float> Out(B*N*Dv, 0.0f), Ref(B*N*Dv, 0.0f);
    
    for (int i = 0; i < B*N*Dk; ++i) {
        Q[i] = 0.1f * static_cast<float>((i % Dk) - Dk/2);
        K[i] = 0.1f * static_cast<float>((i % Dk) - Dk/2);
    }
    for (int i = 0; i < B*N*Dv; ++i) {
        V[i] = 0.1f * static_cast<float>((i % Dv) - Dv/2);
    }
    
    scaledDotProductAttentionCPU(Q.data(), K.data(), V.data(), Ref.data(), B, N, Dk, Dv);
    fusedScaledDotProductAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);
    
    for (int i = 0; i < B*N*Dv; ++i) {
        EXPECT_NEAR(Out[i], Ref[i], 1e-3f);
    }
}

TEST(FusedScaledDotProductAttention, SingleSequence) {
    const int B = 2, N = 1, Dk = 4, Dv = 4;
    std::vector<float> Q(B*N*Dk), K(B*N*Dk), V(B*N*Dv);
    std::vector<float> Out(B*N*Dv, 0.0f), Ref(B*N*Dv, 0.0f);
    
    for (int i = 0; i < B*N*Dk; ++i) {
        Q[i] = 0.1f * static_cast<float>(i);
        K[i] = 0.1f * static_cast<float>(i);
    }
    for (int i = 0; i < B*N*Dv; ++i) {
        V[i] = 0.1f * static_cast<float>(i);
    }
    
    scaledDotProductAttentionCPU(Q.data(), K.data(), V.data(), Ref.data(), B, N, Dk, Dv);
    fusedScaledDotProductAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);
    
    for (int i = 0; i < B*N*Dv; ++i) {
        EXPECT_NEAR(Out[i], Ref[i], 1e-3f);
    }
}

TEST(FusedScaledDotProductAttention, LargeBatch) {
    const int B = 8, N = 16, Dk = 32, Dv = 32;
    std::vector<float> Q(B*N*Dk), K(B*N*Dk), V(B*N*Dv);
    std::vector<float> Out(B*N*Dv, 0.0f);
    
    for (int i = 0; i < B*N*Dk; ++i) {
        Q[i] = 0.01f * static_cast<float>((i % Dk) - Dk/2);
        K[i] = 0.01f * static_cast<float>((i % Dk) - Dk/2);
    }
    for (int i = 0; i < B*N*Dv; ++i) {
        V[i] = 0.01f * static_cast<float>((i % Dv) - Dv/2);
    }
    
    fusedScaledDotProductAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);
    
    for (int i = 0; i < B*N*Dv; ++i) {
        EXPECT_TRUE(std::isfinite(Out[i]));
    }
}

TEST(FusedScaledDotProductAttention, ZeroInput) {
    const int B = 2, N = 4, Dk = 8, Dv = 8;
    std::vector<float> Q(B*N*Dk, 0.0f), K(B*N*Dk, 0.0f), V(B*N*Dv, 0.0f);
    std::vector<float> Out(B*N*Dv, 0.0f);
    
    fusedScaledDotProductAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);
    
    for (int i = 0; i < B*N*Dv; ++i) {
        EXPECT_NEAR(Out[i], 0.0f, 1e-5f);
    }
}

TEST(FusedScaledDotProductAttention, NegativeValues) {
    const int B = 2, N = 4, Dk = 8, Dv = 8;
    std::vector<float> Q(B*N*Dk), K(B*N*Dk), V(B*N*Dv);
    std::vector<float> Out(B*N*Dv, 0.0f), Ref(B*N*Dv, 0.0f);
    
    for (int i = 0; i < B*N*Dk; ++i) {
        Q[i] = -0.1f * static_cast<float>(i % Dk);
        K[i] = -0.1f * static_cast<float>(i % Dk);
    }
    for (int i = 0; i < B*N*Dv; ++i) {
        V[i] = -0.1f * static_cast<float>(i % Dv);
    }
    
    scaledDotProductAttentionCPU(Q.data(), K.data(), V.data(), Ref.data(), B, N, Dk, Dv);
    fusedScaledDotProductAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);
    
    for (int i = 0; i < B*N*Dv; ++i) {
        EXPECT_NEAR(Out[i], Ref[i], 1e-3f);
    }
}

