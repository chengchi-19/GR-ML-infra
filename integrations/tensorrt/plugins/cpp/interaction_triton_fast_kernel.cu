#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Warp内归约求和辅助函数
 * 使用shuffle指令在warp内高效归约
 */
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief 块内归约求和辅助函数
 * 利用共享内存实现块内线程间的高效归约
 */
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // 用于warp间通信的共享内存
    int lane = threadIdx.x % 32; // Warp内的线程ID
    int wid = threadIdx.x / 32;  // Warp ID

    // Warp内归约
    val = warpReduceSum(val);

    // Warp 0号线程将Warp内归约结果写入共享内存
    if (lane == 0) shared[wid] = val;

    __syncthreads(); // 确保所有Warp都已完成写入

    // 第一个Warp从共享内存读取数据并进行最终归约
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

/**
 * @brief 特征交互CUDA Kernel
 * 计算所有特征对之间的点积交互
 *
 * @param emb 输入特征嵌入 [B, F, D]
 * @param out 输出特征交互 [B, F*(F-1)/2]
 * @param B 批次大小
 * @param F 特征数量
 * @param D 嵌入维度
 */
__global__ void interaction_kernel(const float* emb, float* out, int B, int F, int D) {
    // 每个线程块计算一个特征对的点积
    int b = blockIdx.x / (F * (F - 1) / 2); // Batch index
    int pair_idx = blockIdx.x % (F * (F - 1) / 2); // Feature pair index

    if (b >= B) return;

    // 根据pair_idx解码出特征i和j的索引
    int i = 0;
    int j = 1;
    int current_pair = 0;

    // 查找对应的特征对索引
    for (int row = 0; row < F; ++row) {
        for (int col = row + 1; col < F; ++col) {
            if (current_pair == pair_idx) {
                i = row;
                j = col;
                goto found_pair; // 跳出双重循环
            }
            current_pair++;
        }
    }

    found_pair:
    // 块内线程并行计算点积
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float emb_i = emb[b * F * D + i * D + d];
        float emb_j = emb[b * F * D + j * D + d];
        sum += emb_i * emb_j;
    }

    // 块内归约(Reduction)
    sum = blockReduceSum(sum);

    // 线程0将最终结果写回
    if (threadIdx.x == 0) {
        out[b * (F * (F - 1) / 2) + pair_idx] = sum;
    }
}

/**
 * @brief Kernel启动函数
 * 启动interaction_kernel进行特征交互计算
 *
 * @param emb 输入特征嵌入指针
 * @param out 输出特征交互指针
 * @param B 批次大小
 * @param F 特征数量
 * @param D 嵌入维度
 * @param stream CUDA流
 */
void interaction_kernel_driver(const float* emb, float* out, int B, int F, int D, cudaStream_t stream) {
    // 计算输出特征对数量
    int out_pairs = F * (F - 1) / 2;

    // 计算Grid和Block配置
    dim3 grid(B * out_pairs); // 每个特征对启动一个线程块
    dim3 block(256); // 块内线程数，经验值为256

    // 启动CUDA Kernel
    interaction_kernel<<<grid, block, 0, stream>>>(emb, out, B, F, D);

    // 可选：同步检查kernel启动是否成功
    #ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    #endif
}