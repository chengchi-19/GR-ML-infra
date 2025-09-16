#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <NvInferRuntimeCommon.h>

// 用于CUDA错误检查的辅助宏
#define CUDA_CHECK(call) do { \n    cudaError_t error = call; \n    if (error != cudaSuccess) { \n        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \n        assert(0); \n    } \n} while(0)

// 线程束级别的求和归约
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 线程块级别的求和归约
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

// FP32实现：带多级池化的分层序列融合
__global__ void hierarchical_fusion_kernel_fp32(
    const float* input,       // 输入序列特征, 形状: [B, S, D]
    float* output,            // 输出融合特征, 形状: [B, S, D]
    const int* level_masks,   // (可选) 层级掩码, 形状: [B, S, L]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_levels,
    float fusion_strength     // 融合强度, 用于控制融合特征与原始特征的加权
) {
    // 每个线程块处理一个序列中的一个位置 (batch_idx, seq_idx)
    // 块内线程沿着特征维度 (hidden_dim) 并行处理
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    // 边界检查
    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= hidden_dim) {
        return;
    }

    // 加载当前位置的原始特征
    int input_offset = (batch_idx * seq_len + seq_idx) * hidden_dim + dim_idx;
    float original_feature = input[input_offset];
    float fused_feature = original_feature; // 初始化融合后特征为原始特征

    // 逐层进行分层融合
    for (int level = 1; level < num_levels; ++level) {
        // 计算当前层级的块(chunk)大小，逐层减半
        int chunk_size = seq_len >> level;
        if (chunk_size == 0) {
            break;
        }

        // 计算当前位置所属的块边界
        int chunk_id = seq_idx / chunk_size;
        int chunk_start = chunk_id * chunk_size;
        int chunk_end = min((chunk_id + 1) * chunk_size, seq_len);
        int chunk_length = chunk_end - chunk_start;

        // (可选) 检查层级掩码
        if (level_masks != nullptr) {
            int mask_offset = (batch_idx * seq_len + seq_idx) * num_levels + level;
            if (level_masks[mask_offset] == 0) {
                continue; // 如果当前层被掩码，则跳过融合
            }
        }

        // 在当前块内进行平均池化
        // 同一线程块内的所有线程协同计算块内特征的均值
        float chunk_sum = 0.0f;
        for (int i = chunk_start; i < chunk_end; ++i) {
            int offset = (batch_idx * seq_len + i) * hidden_dim + dim_idx;
            chunk_sum += input[offset];
        }
        float chunk_avg = (chunk_length > 0) ? (chunk_sum / chunk_length) : 0.0f;

        // 线性融合: 将池化后的高级特征与原始特征加权融合
        // fused = original * (1 - strength) + pooled * strength
        fused_feature = original_feature * (1.0f - fusion_strength) + chunk_avg * fusion_strength;
    }

    // 将最终的融合结果写回全局内存
    output[input_offset] = fused_feature;
}

// FP16实现: 分层序列融合
__global__ void hierarchical_fusion_kernel_fp16(
    const __half* input,      // 输入序列特征, 形状: [B, S, D]
    __half* output,           // 输出融合特征, 形状: [B, S, D]
    const int* level_masks,   // (可选) 层级掩码, 形状: [B, S, L]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_levels,
    float fusion_strength
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= hidden_dim) {
        return;
    }

    int input_offset = (batch_idx * seq_len + seq_idx) * hidden_dim + dim_idx;
    __half original_feature = input[input_offset];
    __half fused_feature = original_feature;

    for (int level = 1; level < num_levels; ++level) {
        int chunk_size = seq_len >> level;
        if (chunk_size == 0) break;

        int chunk_id = seq_idx / chunk_size;
        int chunk_start = chunk_id * chunk_size;
        int chunk_end = min((chunk_id + 1) * chunk_size, seq_len);
        int chunk_length = chunk_end - chunk_start;

        // 为了精度，中间的累加和求均值使用FP32
        float chunk_sum = 0.0f;
        for (int i = chunk_start; i < chunk_end; ++i) {
            int offset = (batch_idx * seq_len + i) * hidden_dim + dim_idx;
            chunk_sum += __half2float(input[offset]);
        }
        float chunk_avg = (chunk_length > 0) ? (chunk_sum / chunk_length) : 0.0f;

        // 线性融合
        float original_fp32 = __half2float(original_feature);
        float fused_fp32 = original_fp32 * (1.0f - fusion_strength) + chunk_avg * fusion_strength;
        fused_feature = __float2half(fused_fp32);
    }

    output[input_offset] = fused_feature;
}

// Kernel启动函数 (C接口)
extern "C" void launch_hierarchical_fusion(
    const void* input,
    void* output,
    const int* level_masks,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_levels,
    float fusion_strength,
    nvinfer1::DataType dtype,
    cudaStream_t stream
) {
    // 启动网格：每个block负责一个序列位置
    dim3 grid(batch_size, seq_len);
    // 启动块：块内线程数等于特征维度，或最大256
    dim3 block(min(hidden_dim, 256));

    if (dtype == nvinfer1::DataType::kFLOAT) {
        hierarchical_fusion_kernel_fp32<<<grid, block, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            level_masks,
            batch_size, seq_len, hidden_dim, num_levels, fusion_strength
        );
    } else if (dtype == nvinfer1::DataType::kHALF) {
        hierarchical_fusion_kernel_fp16<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(input),
            static_cast<__half*>(output),
            level_masks,
            batch_size, seq_len, hidden_dim, num_levels, fusion_strength
        );
    }

    CUDA_CHECK(cudaGetLastError());
}