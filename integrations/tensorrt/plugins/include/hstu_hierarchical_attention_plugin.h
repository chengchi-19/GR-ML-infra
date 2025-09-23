#pragma once

#include "tensorrt_plugin_base.h"
#include <string>
#include <vector>

namespace gr_ml_infra {
namespace tensorrt_plugins {

/**
 * @brief HSTU分层注意力 TensorRT插件
 * 对应Triton算子: hstu_hierarchical_attention
 *
 * 功能:
 * 1. 实现HSTU模型中的分层注意力机制
 * 2. 支持多层级(multi-level)的注意力计算
 * 3. 优化内存访问模式以提高性能
 *
 * 输入:
 * - query: [B, H, S, D] - Query张量
 * - key: [B, H, S, D] - Key张量
 * - value: [B, H, S, D] - Value张量
 * - level_mask: [B, H, S, S] - 层级掩码
 * 输出:
 * - output: [B, H, S, D] - 注意力计算结果
 */
class HSTUHierarchicalAttentionPlugin : public GRPluginBase {
public:
    /**
     * @brief 构造函数
     * @param num_levels 层级数量
     * @param level_size 每层的大小
     */
    HSTUHierarchicalAttentionPlugin(int32_t num_levels, int32_t level_size);

    /**
     * @brief 反序列化构造函数
     * @param serial_data 序列化数据
     * @param serial_length 数据长度
     */
    HSTUHierarchicalAttentionPlugin(const void* serial_data, size_t serial_length);

    ~HSTUHierarchicalAttentionPlugin() override = default;

    // IPluginV2DynamicExt接口
    int32_t getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index,
        const nvinfer1::DimsExprs* inputs,
        int32_t nb_inputs,
        nvinfer1::IExprBuilder& expr_builder
    ) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* in_out,
        int32_t nb_inputs,
        int32_t nb_outputs
    ) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nb_inputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nb_outputs
    ) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nb_inputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nb_outputs
    ) const noexcept override;

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* input_desc,
        const nvinfer1::PluginTensorDesc* output_desc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

    // IPluginV2接口
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    int32_t num_levels_;  ///< 分层数量
    int32_t level_size_;  ///< 每层大小

    // CUDA kernel启动参数结构
    struct KernelParams {
        int32_t batch_size;    ///< 批次大小
        int32_t num_heads;     ///< 注意力头数
        int32_t seq_length;    ///< 序列长度
        int32_t head_dim;      ///< 每个头的维度
        int32_t num_levels;    ///< 层级数量
        int32_t level_size;    ///< 每层大小
    };

    /**
     * @brief 反序列化内部数据
     * @param serialData 序列化数据
     * @param serialLength 数据长度
     */
    void deserialize(const void* serialData, size_t serialLength);
};

/**
 * @brief HSTU分层注意力插件创建器
 */
class HSTUHierarchicalAttentionPluginCreator : public GRPluginCreatorBase {
public:
    HSTUHierarchicalAttentionPluginCreator();

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serial_data,
        size_t serial_length
    ) noexcept override;

private:
    static constexpr const char* PLUGIN_NAME = "HSTUHierarchicalAttention";
    static constexpr const char* PLUGIN_VERSION = "1.0";
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra