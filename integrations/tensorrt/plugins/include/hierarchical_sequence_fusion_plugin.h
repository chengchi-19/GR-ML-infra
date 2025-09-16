#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include "tensorrt_plugin_base.h"

namespace gr_ml_infra {
namespace tensorrt_plugins {

/**
 * @brief HSTU模型专用分层序列融合插件
 * 对应Triton算子: hierarchical_sequence_fusion
 *
 * 功能:
 * 1. 对多层级的序列表示进行融合
 * 2. 支持可学习的层级权重
 * 3. 优化长序列依赖的建模
 */
class HierarchicalSequenceFusionPlugin : public GRPluginBase {
public:
    HierarchicalSequenceFusionPlugin(
        int32_t hidden_dim,
        int32_t num_levels = 3
    ) : GRPluginBase("HierarchicalSequenceFusion", "1.0"),
        hidden_dim_(hidden_dim), num_levels_(num_levels) {}

    // 反序列化构造函数
    HierarchicalSequenceFusionPlugin(const void* serialData, size_t serialLength);

    ~HierarchicalSequenceFusionPlugin() override = default;

    // IPluginV2DynamicExt接口
    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

    // IPluginV2接口
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new HierarchicalSequenceFusionPlugin(hidden_dim_, num_levels_);
    }

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    int32_t hidden_dim_;
    int32_t num_levels_;

    void deserialize(const void* serialData, size_t serialLength);
};

/**
 * @brief 分层序列融合插件创建器
 * 对应Triton算子: hierarchical_sequence_fusion
 *
 * 功能:
 * 1. 用于创建 HierarchicalSequenceFusionPlugin 插件实例
 * 2. 支持通过插件字段集合创建插件
 * 3. 支持从序列化数据反序列化创建插件
 */
class HierarchicalSequenceFusionPluginCreator : public GRPluginCreatorBase {
public:
    HierarchicalSequenceFusionPluginCreator();

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength
    ) noexcept override;

private:
    static constexpr const char* PLUGIN_NAME = "HierarchicalSequenceFusion";
    static constexpr const char* PLUGIN_VERSION = "1.0";
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra