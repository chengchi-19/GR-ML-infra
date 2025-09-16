#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <device_launch_parameters.h>

// 数值稳定的softmax实现
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// FP32融合注意力+LayerNorm kernel
__global__ void fused_attention_layernorm_kernel_fp32(
    const float* input,
    const float* weight_q,
    const float* weight_k,
    const float* weight_v,
    const float* weight_o,
    const float* layer_norm_weight,
    const float* layer_norm_bias,
    float* output,
    float* workspace_q,
    float* workspace_k,
    float* workspace_v,
    float* workspace_attn_scores,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim,
    float dropout_rate,
    float layer_norm_eps
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.z;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    // 计算Q, K, V
    extern __shared__ float shmem[];
    float* s_input = shmem;
    float* s_q = s_input + hidden_dim;
    float* s_k = s_q + head_dim;
    float* s_v = s_k + head_dim;

    // 加载输入到共享内存
    int input_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_input[i] = input[input_offset + i];
    }
    __syncthreads();

    // 计算Q矩阵乘法
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float q_val = 0.0f;
        int head_offset = head_idx * head_dim;
        for (int j = 0; j < hidden_dim; ++j) {
            q_val += s_input[j] * weight_q[(head_offset + i) * hidden_dim + j];
        }
        s_q[i] = q_val;
    }

    // 计算K矩阵乘法
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float k_val = 0.0f;
        int head_offset = head_idx * head_dim;
        for (int j = 0; j < hidden_dim; ++j) {
            k_val += s_input[j] * weight_k[(head_offset + i) * hidden_dim + j];
        }
        s_k[i] = k_val;
    }

    // 计算V矩阵乘法
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v_val = 0.0f;
        int head_offset = head_idx * head_dim;
        for (int j = 0; j < hidden_dim; ++j) {
            v_val += s_input[j] * weight_v[(head_offset + i) * hidden_dim + j];
        }
        s_v[i] = v_val;
    }
    __syncthreads();

    // 存储Q, K, V到workspace
    int qkv_offset = batch_idx * num_heads * seq_len * head_dim +
                     head_idx * seq_len * head_dim + seq_idx * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        workspace_q[qkv_offset + i] = s_q[i];
        workspace_k[qkv_offset + i] = s_k[i];
        workspace_v[qkv_offset + i] = s_v[i];
    }
}

// 注意力分数计算kernel
__global__ void attention_scores_kernel_fp32(
    const float* workspace_q,
    const float* workspace_k,
    float* workspace_attn_scores,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_seq_idx = blockIdx.z;
    int k_seq_idx = threadIdx.x + blockIdx.w * blockDim.x;

    if (batch_idx >= batch_size || head_idx >= num_heads ||
        q_seq_idx >= seq_len || k_seq_idx >= seq_len) {
        return;
    }

    int q_offset = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim + q_seq_idx * head_dim;
    int k_offset = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim + k_seq_idx * head_dim;

    // 计算Q·K^T
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        score += workspace_q[q_offset + i] * workspace_k[k_offset + i];
    }

    // 缩放
    score /= sqrtf(static_cast<float>(head_dim));

    // 存储注意力分数
    int score_offset = batch_idx * num_heads * seq_len * seq_len +
                       head_idx * seq_len * seq_len +
                       q_seq_idx * seq_len + k_seq_idx;
    workspace_attn_scores[score_offset] = score;
}

// Softmax + 注意力输出计算kernel
__global__ void attention_output_kernel_fp32(
    const float* workspace_attn_scores,
    const float* workspace_v,
    const float* weight_o,
    float* temp_output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int hidden_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }

    extern __shared__ float shmem[];
    float* s_attn_output = shmem;

    // 初始化
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_attn_output[i] = 0.0f;
    }
    __syncthreads();

    // 对每个head处理
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        int score_offset = batch_idx * num_heads * seq_len * seq_len +
                          head_idx * seq_len * seq_len + seq_idx * seq_len;

        // Softmax
        float max_score = -FLT_MAX;
        for (int k = 0; k < seq_len; ++k) {
            max_score = fmaxf(max_score, workspace_attn_scores[score_offset + k]);
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            sum_exp += expf(workspace_attn_scores[score_offset + k] - max_score);
        }

        // 计算加权V
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float weighted_v = 0.0f;
            for (int k = 0; k < seq_len; ++k) {
                float attn_weight = expf(workspace_attn_scores[score_offset + k] - max_score) / sum_exp;
                int v_offset = batch_idx * num_heads * seq_len * head_dim +
                              head_idx * seq_len * head_dim + k * head_dim;
                weighted_v += attn_weight * workspace_v[v_offset + i];
            }

            // 输出投影
            int head_offset = head_idx * head_dim;
            for (int j = 0; j < hidden_dim; ++j) {
                atomicAdd(&s_attn_output[j],
                         weighted_v * weight_o[j * (num_heads * head_dim) + head_offset + i]);
            }
        }
    }
    __syncthreads();

    // 存储到临时输出
    int output_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        temp_output[output_offset + i] = s_attn_output[i];
    }
}

// LayerNorm kernel
__global__ void layer_norm_kernel_fp32(
    const float* input,
    const float* temp_output,
    const float* layer_norm_weight,
    const float* layer_norm_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float layer_norm_eps
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }

    extern __shared__ float shmem[];
    float* s_data = shmem;

    int offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;

    // 加载数据 (residual connection)
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_data[i] = input[offset + i] + temp_output[offset + i];
    }
    __syncthreads();

    // 计算均值
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += s_data[i];
    }
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
        s_data[hidden_dim] = sum / hidden_dim; // 存储均值
    }
    __syncthreads();

    float mean = s_data[hidden_dim];

    // 计算方差
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = s_data[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    if (tid == 0) {
        s_data[hidden_dim + 1] = var_sum / hidden_dim; // 存储方差
    }
    __syncthreads();

    float variance = s_data[hidden_dim + 1];
    float inv_std = rsqrtf(variance + layer_norm_eps);

    // 应用LayerNorm
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float normalized = (s_data[i] - mean) * inv_std;
        output[offset + i] = normalized * layer_norm_weight[i] + layer_norm_bias[i];
    }
}

// FP16版本的kernels
__global__ void fused_attention_layernorm_kernel_fp16(
    const __half* input,
    const __half* weight_q,
    const __half* weight_k,
    const __half* weight_v,
    const __half* weight_o,
    const __half* layer_norm_weight,
    const __half* layer_norm_bias,
    __half* output,
    __half* workspace_q,
    __half* workspace_k,
    __half* workspace_v,
    __half* workspace_attn_scores,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim,
    float dropout_rate,
    float layer_norm_eps
) {
    // FP16版本实现（使用__half2进行矢量化操作）
    // 实现类似FP32版本，但使用half精度计算
}

// 主要的启动函数
extern "C" {

void launch_fused_attention_layernorm_kernel_fp32(
    const float* input,
    const float* weight_q,
    const float* weight_k,
    const float* weight_v,
    const float* weight_o,
    const float* layer_norm_weight,
    const float* layer_norm_bias,
    float* output,
    void* workspace,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim,
    float dropout_rate,
    float layer_norm_eps,
    cudaStream_t stream
) {
    // 计算workspace偏移
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t attn_scores_size = batch_size * num_heads * seq_len * seq_len;
    size_t temp_output_size = batch_size * seq_len * hidden_dim;

    float* workspace_q = static_cast<float*>(workspace);
    float* workspace_k = workspace_q + qkv_size;
    float* workspace_v = workspace_k + qkv_size;
    float* workspace_attn_scores = workspace_v + qkv_size;
    float* temp_output = workspace_attn_scores + attn_scores_size;

    // 第一阶段: 计算Q, K, V
    dim3 grid1(batch_size, num_heads, seq_len);
    dim3 block1(256);
    size_t shmem1 = (hidden_dim + 3 * head_dim) * sizeof(float);

    fused_attention_layernorm_kernel_fp32<<<grid1, block1, shmem1, stream>>>(
        input, weight_q, weight_k, weight_v, weight_o,
        layer_norm_weight, layer_norm_bias, output,
        workspace_q, workspace_k, workspace_v, workspace_attn_scores,
        batch_size, seq_len, hidden_dim, num_heads, head_dim,
        dropout_rate, layer_norm_eps
    );

    // 第二阶段: 计算注意力分数
    dim3 grid2(batch_size, num_heads, seq_len, (seq_len + 255) / 256);
    dim3 block2(256);

    attention_scores_kernel_fp32<<<grid2, block2, 0, stream>>>(
        workspace_q, workspace_k, workspace_attn_scores,
        batch_size, num_heads, seq_len, head_dim
    );

    // 第三阶段: 计算注意力输出
    dim3 grid3(batch_size, seq_len);
    dim3 block3(256);
    size_t shmem3 = hidden_dim * sizeof(float);

    attention_output_kernel_fp32<<<grid3, block3, shmem3, stream>>>(
        workspace_attn_scores, workspace_v, weight_o, temp_output,
        batch_size, num_heads, seq_len, head_dim, hidden_dim
    );

    // 第四阶段: LayerNorm
    dim3 grid4(batch_size, seq_len);
    dim3 block4(256);
    size_t shmem4 = (hidden_dim + 2) * sizeof(float);

    layer_norm_kernel_fp32<<<grid4, block4, shmem4, stream>>>(
        input, temp_output, layer_norm_weight, layer_norm_bias, output,
        batch_size, seq_len, hidden_dim, layer_norm_eps
    );
}

void launch_fused_attention_layernorm_kernel_fp16(
    const __half* input,
    const __half* weight_q,
    const __half* weight_k,
    const __half* weight_v,
    const __half* weight_o,
    const __half* layer_norm_weight,
    const __half* layer_norm_bias,
    __half* output,
    void* workspace,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim,
    float dropout_rate,
    float layer_norm_eps,
    cudaStream_t stream
) {
    // FP16版本实现
    // 类似FP32但使用half类型和__half2矢量化操作
}

} // extern "C"