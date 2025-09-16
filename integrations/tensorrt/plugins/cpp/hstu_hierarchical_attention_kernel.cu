#include <cuda_runtime.h>
#include <cmath>

__global__ void hstu_hierarchical_attention_kernel(
    const float* query, const float* key, const float* value, const float* level_mask, float* output,
    int B, int H, int S, int D, int NUM_LEVELS, int LEVEL_SIZE) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int s_q = threadIdx.x;

    if (b >= B || h >= H || s_q >= S) return;

    float output_acc[256]; // Assume D <= 256
    for (int d = 0; d < D; ++d) {
        output_acc[d] = 0.0f;
    }
    float total_weight = 0.0f;

    const float* q = query + (b * H + h) * S * D + s_q * D;

    for (int level = 0; level < NUM_LEVELS; ++level) {
        int level_start = level * LEVEL_SIZE;
        int level_end = min((level + 1) * LEVEL_SIZE, S);

        if (level_start >= S) break;

        float level_max_score = -1e20f;
        float level_sum_exp = 0.0f;
        float level_weighted_sum[256] = {0.0f}; // Assume D <= 256

        for (int s_kv = level_start; s_kv < level_end; ++s_kv) {
            float mask = level_mask[((b * H + h) * S + s_q) * S + s_kv];
            if (mask == 0) continue;

            const float* k = key + (b * H + h) * S * D + s_kv * D;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += q[d] * k[d];
            }
            score /= sqrtf(D);

            float level_decay = powf(0.9f, level);
            score *= level_decay;

            if (score > level_max_score) {
                level_max_score = score;
            }
        }

        for (int s_kv = level_start; s_kv < level_end; ++s_kv) {
            float mask = level_mask[((b * H + h) * S + s_q) * S + s_kv];
            if (mask == 0) continue;

            const float* k = key + (b * H + h) * S * D + s_kv * D;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += q[d] * k[d];
            }
            score /= sqrtf(D);
            float level_decay = powf(0.9f, level);
            score *= level_decay;

            float weight = expf(score - level_max_score);
            level_sum_exp += weight;

            const float* v = value + (b * H + h) * S * D + s_kv * D;
            for (int d = 0; d < D; ++d) {
                level_weighted_sum[d] += weight * v[d];
            }
        }

        if (level_sum_exp > 0) {
            float level_fusion_weight = (float)(level + 1) / NUM_LEVELS;
            total_weight += level_fusion_weight;
            for (int d = 0; d < D; ++d) {
                output_acc[d] += level_fusion_weight * level_weighted_sum[d] / level_sum_exp;
            }
        }
    }

    float* out = output + (b * H + h) * S * D + s_q * D;
    if (total_weight > 0) {
        for (int d = 0; d < D; ++d) {
            out[d] = output_acc[d] / total_weight;
        }
    } else {
        for (int d = 0; d < D; ++d) {
            out[d] = output_acc[d];
        }
    }
}

void hstu_hierarchical_attention_kernel_driver(const float* query, const float* key, const float* value, const float* level_mask, float* output, int B, int H, int S, int D, int num_levels, int level_size, cudaStream_t stream) {
    dim3 grid(B, H);
    dim3 block(S);
    hstu_hierarchical_attention_kernel<<<grid, block, 0, stream>>>(query, key, value, level_mask, output, B, H, S, D, num_levels, level_size);
}