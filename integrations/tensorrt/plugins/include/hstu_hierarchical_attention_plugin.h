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
 */
class HSTUHierarchicalAttentionPlugin : public GRPluginBase {
public:
    HSTUHierarchicalAttentionPlugin(int num_levels, int level_size);

    // 反序列化构造函数
    HSTUHierarchicalAttentionPlugin(const void* serial_data, size_t serial_length);

    ~HSTUHierarchicalAttentionPlugin() override = default;

    // IPluginV2DynamicExt接口
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs, nvinfer1::IExprBuilder& expr_builder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs, int nb_outputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs, const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs, const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc, const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2接口
    const char* getPluginType() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    int num_levels_;
    int level_size_;

    void deserialize(const void* serialData, size_t serialLength);
};

/**
 * @brief HSTU分层注意力插件创建器
 */
class HSTUHierarchicalAttentionPluginCreator : public GRPluginCreatorBase {
public:
    HSTUHierarchicalAttentionPluginCreator();

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serial_data, size_t serial_length) noexcept override;

private:
    static constexpr const char* PLUGIN_NAME = "HSTUHierarchicalAttention";
    static constexpr const char* PLUGIN_VERSION = "1.0";
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra