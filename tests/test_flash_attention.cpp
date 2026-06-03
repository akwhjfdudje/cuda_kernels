#include <gtest/gtest.h>
#include "fused/fused.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

static void attentionCPU(
    const float* Q, const float* K, const float* V,
    float* Output,
    int batch, int N, int d_k, int d_v)
{
    for (int b = 0; b < batch; ++b) {
        for (int q_row = 0; q_row < N; ++q_row) {
            std::vector<float> scores(N);
            float max_score = -FLT_MAX;

            for (int k_row = 0; k_row < N; ++k_row) {
                float score = 0.0f;
                for (int i = 0; i < d_k; ++i) {
                    score += Q[b * N * d_k + q_row * d_k + i] *
                             K[b * N * d_k + k_row * d_k + i];
                }
                score /= std::sqrt(static_cast<float>(d_k));
                scores[k_row] = score;
                max_score = std::max(max_score, score);
            }

            float sum_exp = 0.0f;
            for (int k_row = 0; k_row < N; ++k_row) {
                scores[k_row] = std::exp(scores[k_row] - max_score);
                sum_exp += scores[k_row];
            }

            for (int j = 0; j < d_v; ++j) {
                float value = 0.0f;
                for (int k_row = 0; k_row < N; ++k_row) {
                    value += (scores[k_row] / sum_exp) *
                             V[b * N * d_v + k_row * d_v + j];
                }
                Output[b * N * d_v + q_row * d_v + j] = value;
            }
        }
    }
}

TEST(FlashAttention, MatchesReference) {
    const int B = 2, N = 8, Dk = 16, Dv = 16;
    std::vector<float> Q(B * N * Dk), K(B * N * Dk), V(B * N * Dv);
    std::vector<float> Out(B * N * Dv, 0.0f), Ref(B * N * Dv, 0.0f);

    for (int i = 0; i < B * N * Dk; ++i) {
        Q[i] = 0.07f * static_cast<float>((i % 11) - 5);
        K[i] = 0.05f * static_cast<float>((i % 13) - 6);
    }
    for (int i = 0; i < B * N * Dv; ++i) {
        V[i] = 0.03f * static_cast<float>((i % 17) - 8);
    }

    attentionCPU(Q.data(), K.data(), V.data(), Ref.data(), B, N, Dk, Dv);
    flashAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);

    for (int i = 0; i < B * N * Dv; ++i) {
        EXPECT_NEAR(Out[i], Ref[i], 1e-3f) << "Mismatch at index " << i;
    }
}

TEST(FlashAttention, HandlesDifferentKeyAndValueDimensions) {
    const int B = 1, N = 6, Dk = 8, Dv = 20;
    std::vector<float> Q(B * N * Dk), K(B * N * Dk), V(B * N * Dv);
    std::vector<float> Out(B * N * Dv, 0.0f), Ref(B * N * Dv, 0.0f);

    for (int i = 0; i < B * N * Dk; ++i) {
        Q[i] = static_cast<float>((i % 7) - 3) / 9.0f;
        K[i] = static_cast<float>((i % 5) - 2) / 7.0f;
    }
    for (int i = 0; i < B * N * Dv; ++i) {
        V[i] = static_cast<float>((i % 9) - 4) / 11.0f;
    }

    attentionCPU(Q.data(), K.data(), V.data(), Ref.data(), B, N, Dk, Dv);
    flashAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);

    for (int i = 0; i < B * N * Dv; ++i) {
        EXPECT_NEAR(Out[i], Ref[i], 1e-3f);
    }
}

TEST(FlashAttention, SingleTokenReturnsValue) {
    const int B = 2, N = 1, Dk = 4, Dv = 5;
    std::vector<float> Q(B * N * Dk, 0.25f);
    std::vector<float> K(B * N * Dk, -0.5f);
    std::vector<float> V(B * N * Dv);
    std::vector<float> Out(B * N * Dv, 0.0f);

    for (int i = 0; i < B * N * Dv; ++i) {
        V[i] = static_cast<float>(i) * 0.125f;
    }

    flashAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);

    for (int i = 0; i < B * N * Dv; ++i) {
        EXPECT_NEAR(Out[i], V[i], 1e-5f);
    }
}

TEST(FlashAttention, ProducesFiniteValuesForLargerInput) {
    const int B = 4, N = 64, Dk = 32, Dv = 24;
    std::vector<float> Q(B * N * Dk), K(B * N * Dk), V(B * N * Dv);
    std::vector<float> Out(B * N * Dv, 0.0f);

    for (int i = 0; i < B * N * Dk; ++i) {
        Q[i] = 0.01f * static_cast<float>((i % 19) - 9);
        K[i] = 0.01f * static_cast<float>((i % 23) - 11);
    }
    for (int i = 0; i < B * N * Dv; ++i) {
        V[i] = 0.02f * static_cast<float>((i % 29) - 14);
    }

    flashAttention(Q.data(), K.data(), V.data(), Out.data(), B, N, Dk, Dv);

    for (float value : Out) {
        EXPECT_TRUE(std::isfinite(value));
    }
}
