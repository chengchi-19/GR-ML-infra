#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * @brief HSTU分层注意力CUDA Kernel
 * 实现多层级的注意力计算机制
 *
 * @param query Query张量 [B, H, S, D]
 * @param key Key张量 [B, H, S, D]
 * @param value Value张量 [B, H, S, D]
 * @param level_mask 层级掩码 [B, H, S, S]
 * @param output 输出张量 [B, H, S, D]
 * @param B 批次大小
 * @param H 注意力头数
 * @param S 序列长度
 * @param D 每个头的维度
 * @param NUM_LEVELS 层级数量
 * @param LEVEL_SIZE 每层大小
 */
__global__ void hstu_hierarchical_attention_kernel(
    const float* query, const float* key, const float* value, const float* level_mask,
    float* output, int B, int H, int S, int D, int NUM_LEVELS, int LEVEL_SIZE) {

    // 计算当前线程处理的索引
    int b = blockIdx.x;          // batch index
    int h = blockIdx.y;          // head index
    int s_q = threadIdx.x;       // query sequence index

    // 边界检查
    if (b >= B || h >= H || s_q >= S) return;

    // 输出累积器，假设D <= 256
    float output_acc[256];
    for (int d = 0; d < D && d < 256; ++d) {
        output_acc[d] = 0.0f;
    }

    float total_weight = 0.0f;

    // 获取当前query向量的起始位置
    const float* q = query + ((b * H + h) * S + s_q) * D;

    // 遍历所有层级
    for (int level = 0; level < NUM_LEVELS; ++level) {
        int level_start = level * LEVEL_SIZE;
        int level_end = min((level + 1) * LEVEL_SIZE, S);

        if (level_start >= S) break;

        // 层级内的注意力计算
        float level_max_score = -1e20f;
        float level_sum_exp = 0.0f;
        float level_weighted_sum[256] = {0.0f}; // 假设D <= 256

        // 第一遍扫描：找到最大分数
        for (int s_kv = level_start; s_kv < level_end; ++s_kv) {
            // 检查掩码
            float mask = level_mask[((b * H + h) * S + s_q) * S + s_kv];
            if (mask == 0.0f) continue;

            // 计算注意力分数
            const float* k = key + ((b * H + h) * S + s_kv) * D;
            float score = 0.0f;

            for (int d = 0; d < D; ++d) {
                score += q[d] * k[d];
            }

            // 缩放因子和层级衰减
            score /= sqrtf((float)D);
            float level_decay = powf(0.9f, (float)level);
            score *= level_decay;

            if (score > level_max_score) {
                level_max_score = score;
            }
        }

        // 第二遍扫描：计算softmax和加权求和
        for (int s_kv = level_start; s_kv < level_end; ++s_kv) {
            // 检查掩码
            float mask = level_mask[((b * H + h) * S + s_q) * S + s_kv];
            if (mask == 0.0f) continue;

            // 重新计算注意力分数
            const float* k = key + ((b * H + h) * S + s_kv) * D;
            float score = 0.0f;

            for (int d = 0; d < D; ++d) {
                score += q[d] * k[d];
            }

            score /= sqrtf((float)D);
            float level_decay = powf(0.9f, (float)level);
            score *= level_decay;

            // 计算softmax权重
            float weight = expf(score - level_max_score);
            level_sum_exp += weight;

            // 加权求和value
            const float* v = value + ((b * H + h) * S + s_kv) * D;
            for (int d = 0; d < D && d < 256; ++d) {
                level_weighted_sum[d] += weight * v[d];
            }
        }

        // 归一化并融合到最终输出
        if (level_sum_exp > 0.0f) {
            float level_fusion_weight = (float)(level + 1) / NUM_LEVELS;
            total_weight += level_fusion_weight;

            for (int d = 0; d < D && d < 256; ++d) {
                output_acc[d] += level_fusion_weight * level_weighted_sum[d] / level_sum_exp;
            }
        }
    }

    // 写入最终输出
    float* out = output + ((b * H + h) * S + s_q) * D;
    if (total_weight > 0.0f) {
        for (int d = 0; d < D; ++d) {
            out[d] = output_acc[d] / total_weight;
        }
    } else {
        for (int d = 0; d < D; ++d) {
            out[d] = 0.0f;
        }
    }
}

/**
 * @brief Kernel启动函数
 * 启动hstu_hierarchical_attention_kernel进行分层注意力计算
 *
 * @param query Query张量指针
 * @param key Key张量指针
 * @param value Value张量指针
 * @param level_mask 层级掩码指针
 * @param output 输出张量指针
 * @param B 批次大小
 * @param H 注意力头数
 * @param S 序列长度
 * @param D 每个头的维度
 * @param num_levels 层级数量
 * @param level_size 每层大小
 * @param stream CUDA流
 */
void hstu_hierarchical_attention_kernel_driver(
    const float* query, const float* key, const float* value, const float* level_mask,
    float* output, int B, int H, int S, int D, int num_levels, int level_size, cudaStream_t stream) {

    // 计算Grid和Block配置
    dim3 grid(B, H);    // 每个batch和head组合启动一个线程块
    dim3 block(S);      // 每个sequence position一个线程

    // 如果序列长度超过最大线程数，需要调整block配置
    if (S > 1024) {
        // 对于很长的序列，可能需要使用不同的策略
        // 这里简化处理，限制block大小为1024
        block.x = 1024;
        // 可以考虑在kernel内部添加循环来处理超长序列
    }

    // 启动CUDA Kernel
    hstu_hierarchical_attention_kernel<<<grid, block, 0, stream>>>(
        query, key, value, level_mask, output, B, H, S, D, num_levels, level_size);

    // 可选：同步检查kernel启动是否成功
    #ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    #endif
}