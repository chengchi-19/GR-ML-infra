#include <cuda_runtime.h>

__global__ void interaction_kernel(const float* emb, float* out, int B, int F, int D) {
    // 每个线程块计算一个特征对的点积
    int b = blockIdx.x / (F * (F - 1) / 2); // Batch index
    int pair_idx = blockIdx.x % (F * (F - 1) / 2); // Feature pair index

    if (b >= B) return;

    // 根据pair_idx解码出特征i和j的索引
    int i = 0;
    int j = 1;
    int current_pair = 0;
    for (int row = 0; row < F; ++row) {
        for (int col = row + 1; col < F; ++col) {
            if (current_pair == pair_idx) {
                i = row;
                j = col;
                break;
            }
            current_pair++;
        }
        if (current_pair > pair_idx) break;
    }

    // 块内线程并行计算点积
    float sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += emb[b * F * D + i * D + d] * emb[b * F * D + j * D + d];
    }

    // 块内归约(Reduction)
    sum = blockReduceSum(sum);

    // 线程0将最终结果写回
    if (threadIdx.x == 0) {
        out[b * (F * (F - 1) / 2) + pair_idx] = sum;
    }
}

// Kernel启动函数
void interaction_kernel_driver(const float* emb, float* out, int B, int F, int D, cudaStream_t stream) {
    int out_pairs = F * (F - 1) / 2;
    dim3 grid(B * out_pairs); // 每个特征对启动一个线程块
    dim3 block(256); // 块内线程数
    interaction_kernel<<<grid, block, 0, stream>>>(emb, out, B, F, D);
}

// 块内归约求和辅助函数
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

// Warp内归约求和辅助函数
__inline__ __device__ float warpReduceSum(float val) {
    // 使用shuffle指令在warp内高效归约
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}