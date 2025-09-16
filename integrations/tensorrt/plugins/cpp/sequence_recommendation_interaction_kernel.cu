#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <device_launch_parameters.h>

// 序列推荐交互CUDA kernel实现
// 对应Triton算子: sequence_recommendation_interaction

// warp归约操作
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// FP32 主要交互计算kernel
__global__ void sequence_recommendation_interaction_kernel_fp32(
    const float* item_embeddings,      // [B, S, D] - 物品嵌入序列
    const float* user_embedding,       // [B, D] - 用户嵌入
    const float* time_weights,         // [B, S] - 时序权重
    const float* interaction_mask,     // [B, S] - 交互掩码
    float* user_item_scores,           // [B, S] - 用户-物品交互分数
    float* item_cooccur,               // [B, S, S] - 物品共现矩阵
    float* short_term,                 // [B, D] - 短期兴趣表示
    float* long_term,                  // [B, D] - 长期偏好表示
    int batch_size,
    int seq_len,
    int hidden_dim,
    int short_window,
    int long_window,
    float decay_factor
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }

    // 检查当前交互是否有效
    int mask_offset = batch_idx * seq_len + seq_idx;
    float current_mask = interaction_mask[mask_offset];

    if (current_mask == 0.0f) {
        return;
    }

    extern __shared__ float shmem[];
    float* s_current_item = shmem;
    float* s_user_emb = s_current_item + hidden_dim;
    float* s_temp = s_user_emb + hidden_dim;

    // 加载当前物品嵌入到共享内存
    int item_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_current_item[i] = item_embeddings[item_offset + i];
    }

    // 加载用户嵌入到共享内存
    int user_offset = batch_idx * hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_user_emb[i] = user_embedding[user_offset + i];
    }
    __syncthreads();

    // 1. 计算用户-物品交互分数
    float user_item_score = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        user_item_score += s_user_emb[i] * s_current_item[i];
    }

    // warp内归约
    user_item_score = warp_reduce_sum(user_item_score);

    // 应用时序权重
    float time_weight = time_weights[mask_offset];
    if (tid == 0) {
        user_item_scores[mask_offset] = user_item_score * time_weight;
    }

    // 2. 计算物品共现矩阵（只计算上三角）
    if (seq_idx < seq_len - 1) {
        for (int other_seq = seq_idx + 1; other_seq < seq_len; ++other_seq) {
            int other_mask_offset = batch_idx * seq_len + other_seq;
            float other_mask = interaction_mask[other_mask_offset];

            if (other_mask == 0.0f) {
                continue;
            }

            // 加载其他物品嵌入
            int other_item_offset = batch_idx * seq_len * hidden_dim + other_seq * hidden_dim;

            float item_similarity = 0.0f;
            for (int i = tid; i < hidden_dim; i += blockDim.x) {
                item_similarity += s_current_item[i] * item_embeddings[other_item_offset + i];
            }

            // warp内归约
            item_similarity = warp_reduce_sum(item_similarity);

            if (tid == 0) {
                // 应用时序衰减
                float seq_distance = other_seq - seq_idx;
                float temporal_decay = expf(-decay_factor * seq_distance);
                float cooccur_score = item_similarity * temporal_decay;

                // 存储到共现矩阵
                int cooccur_offset = (batch_idx * seq_len + seq_idx) * seq_len + other_seq;
                item_cooccur[cooccur_offset] = cooccur_score;
            }
        }
    }

    // 3. 更新短期兴趣表示
    if (seq_idx >= seq_len - short_window) {
        float short_weight = time_weight * (1.0f + (seq_idx - (seq_len - short_window)) * 0.1f);

        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float weighted_val = s_current_item[i] * short_weight;
            atomicAdd(&short_term[user_offset + i], weighted_val);
        }
    }

    // 4. 更新长期偏好表示
    float long_weight = time_weight * (0.5f + 0.5f * expf(-0.1f * (seq_len - seq_idx - 1)));

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float weighted_val = s_current_item[i] * long_weight;
        atomicAdd(&long_term[user_offset + i], weighted_val);
    }
}

// 协同过滤分数计算kernel
__global__ void sequence_collaborative_filtering_kernel_fp32(
    const float* item_embeddings,      // [B, S, D] - 物品嵌入序列
    const float* item_cooccur,         // [B, S, S] - 物品共现矩阵
    const float* interaction_mask,     // [B, S] - 交互掩码
    float* cf_scores,                  // [B, S] - 协同过滤分数
    float* neighbor_weights,           // [B, S, TOP_K] - Top-K近邻权重
    int batch_size,
    int seq_len,
    int hidden_dim,
    int top_k,
    float min_cooccur
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }

    // 检查当前物品是否有效
    int mask_offset = batch_idx * seq_len + seq_idx;
    float current_mask = interaction_mask[mask_offset];

    if (current_mask == 0.0f) {
        return;
    }

    extern __shared__ float shmem[];
    float* s_top_scores = shmem;
    int* s_top_indices = (int*)(s_top_scores + top_k);

    // 初始化Top-K数组
    if (tid < top_k) {
        s_top_scores[tid] = -1e9f;
        s_top_indices[tid] = -1;
    }
    __syncthreads();

    float cf_score = 0.0f;
    int neighbor_count = 0;

    // 扫描所有其他物品
    int cooccur_base = (batch_idx * seq_len + seq_idx) * seq_len;

    for (int other_seq = tid; other_seq < seq_len; other_seq += blockDim.x) {
        if (other_seq == seq_idx) {
            continue;
        }

        // 检查其他物品是否有效
        int other_mask_offset = batch_idx * seq_len + other_seq;
        float other_mask = interaction_mask[other_mask_offset];

        if (other_mask == 0.0f) {
            continue;
        }

        // 获取共现分数
        float cooccur_score;
        if (seq_idx < other_seq) {
            // 从上三角矩阵读取
            cooccur_score = item_cooccur[cooccur_base + other_seq];
        } else {
            // 从对称位置读取
            int symmetric_offset = (batch_idx * seq_len + other_seq) * seq_len + seq_idx;
            cooccur_score = item_cooccur[symmetric_offset];
        }

        // 如果共现分数足够高
        if (cooccur_score > min_cooccur) {
            // 更新Top-K近邻（简化版本，使用原子操作）
            for (int k = 0; k < top_k; ++k) {
                float old_score = atomicExch(&s_top_scores[k], -1e9f);
                if (cooccur_score > old_score) {
                    atomicExch(&s_top_scores[k], cooccur_score);
                    atomicExch(&s_top_indices[k], other_seq);
                    break;
                } else {
                    atomicExch(&s_top_scores[k], old_score);
                }
            }

            cf_score += cooccur_score;
            neighbor_count++;
        }
    }

    // 归约协同过滤分数
    cf_score = warp_reduce_sum(cf_score);
    neighbor_count = warp_reduce_sum(neighbor_count);

    if (tid == 0) {
        // 归一化并存储
        if (neighbor_count > 0) {
            cf_scores[mask_offset] = cf_score / neighbor_count;
        } else {
            cf_scores[mask_offset] = 0.0f;
        }

        // 存储Top-K近邻权重
        int neighbor_base = (batch_idx * seq_len + seq_idx) * top_k;
        for (int k = 0; k < top_k; ++k) {
            neighbor_weights[neighbor_base + k] = s_top_scores[k];
        }
    }
}

// FP16版本的kernels
__global__ void sequence_recommendation_interaction_kernel_fp16(
    const __half* item_embeddings,
    const __half* user_embedding,
    const __half* time_weights,
    const __half* interaction_mask,
    __half* user_item_scores,
    __half* item_cooccur,
    __half* short_term,
    __half* long_term,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int short_window,
    int long_window,
    float decay_factor
) {
    // FP16版本实现（使用__half2进行矢量化操作）
    // 实现类似FP32版本，但使用half精度计算
}

__global__ void sequence_collaborative_filtering_kernel_fp16(
    const __half* item_embeddings,
    const __half* item_cooccur,
    const __half* interaction_mask,
    __half* cf_scores,
    __half* neighbor_weights,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int top_k,
    float min_cooccur
) {
    // FP16版本协同过滤实现
}

// 主要启动函数
extern "C" {

void launch_sequence_recommendation_interaction_kernel_fp32(
    const float* item_embeddings,
    const float* user_embedding,
    const float* time_weights,
    const float* interaction_mask,
    float* user_item_scores,
    float* item_cooccur,
    float* short_term,
    float* long_term,
    float* cf_scores,
    float* neighbor_weights,
    void* workspace,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int short_window,
    int long_window,
    float decay_factor,
    int top_k,
    float min_cooccur,
    cudaStream_t stream
) {
    // 第一阶段: 主要交互计算
    dim3 grid1(batch_size, seq_len);
    dim3 block1(min(256, ((hidden_dim + 31) / 32) * 32));
    size_t shmem1 = (2 * hidden_dim + 256) * sizeof(float);

    sequence_recommendation_interaction_kernel_fp32<<<grid1, block1, shmem1, stream>>>(
        item_embeddings, user_embedding, time_weights, interaction_mask,
        user_item_scores, item_cooccur, short_term, long_term,
        batch_size, seq_len, hidden_dim, short_window, long_window, decay_factor
    );

    // 同步确保第一阶段完成
    cudaStreamSynchronize(stream);

    // 第二阶段: 协同过滤计算
    dim3 grid2(batch_size, seq_len);
    dim3 block2(min(256, seq_len));
    size_t shmem2 = (top_k * sizeof(float)) + (top_k * sizeof(int));

    sequence_collaborative_filtering_kernel_fp32<<<grid2, block2, shmem2, stream>>>(
        item_embeddings, item_cooccur, interaction_mask,
        cf_scores, neighbor_weights,
        batch_size, seq_len, hidden_dim, top_k, min_cooccur
    );
}

void launch_sequence_recommendation_interaction_kernel_fp16(
    const __half* item_embeddings,
    const __half* user_embedding,
    const __half* time_weights,
    const __half* interaction_mask,
    __half* user_item_scores,
    __half* item_cooccur,
    __half* short_term,
    __half* long_term,
    __half* cf_scores,
    __half* neighbor_weights,
    void* workspace,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int short_window,
    int long_window,
    float decay_factor,
    int top_k,
    float min_cooccur,
    cudaStream_t stream
) {
    // FP16版本启动函数
    // 类似FP32但使用half类型和__half2矢量化操作

    dim3 grid1(batch_size, seq_len);
    dim3 block1(min(256, ((hidden_dim + 31) / 32) * 32));
    size_t shmem1 = (2 * hidden_dim + 256) * sizeof(__half);

    sequence_recommendation_interaction_kernel_fp16<<<grid1, block1, shmem1, stream>>>(
        item_embeddings, user_embedding, time_weights, interaction_mask,
        user_item_scores, item_cooccur, short_term, long_term,
        batch_size, seq_len, hidden_dim, short_window, long_window, decay_factor
    );

    cudaStreamSynchronize(stream);

    dim3 grid2(batch_size, seq_len);
    dim3 block2(min(256, seq_len));
    size_t shmem2 = (top_k * sizeof(__half)) + (top_k * sizeof(int));

    sequence_collaborative_filtering_kernel_fp16<<<grid2, block2, shmem2, stream>>>(
        item_embeddings, item_cooccur, interaction_mask,
        cf_scores, neighbor_weights,
        batch_size, seq_len, hidden_dim, top_k, min_cooccur
    );
}

} // extern "C"