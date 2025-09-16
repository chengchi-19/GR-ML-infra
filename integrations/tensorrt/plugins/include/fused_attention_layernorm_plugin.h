#pragma once

#include "tensorrt_plugin_base.h"

namespace gr_ml_infra {
namespace tensorrt_plugins {

/**
 * 融合多头注意力+LayerNorm TensorRT插件
 * 对应Triton算子: fused_attention_layernorm
 */
class FusedAttentionLayerNormPlugin : public GRPluginBase {
public:
    FusedAttentionLayerNormPlugin(
        int32_t hidden_dim,
        int32_t num_heads,
        float dropout_rate = 0.1f,
        float layer_norm_eps = 1e-5f
    );

    // 反序列化构造函数
    FusedAttentionLayerNormPlugin(const void* serialData, size_t serialLength);

    ~FusedAttentionLayerNormPlugin() override = default;

    // IPluginV2DynamicExt接口
    int32_t getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs
    ) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nbOutputs
    ) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs
    ) const noexcept override;

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
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
    int32_t hidden_dim_;
    int32_t num_heads_;
    int32_t head_dim_;
    float dropout_rate_;
    float layer_norm_eps_;

    // CUDA kernel启动参数
    struct KernelParams {
        int32_t batch_size;
        int32_t seq_len;
        int32_t hidden_dim;
        int32_t num_heads;
        int32_t head_dim;
        float dropout_rate;
        float layer_norm_eps;
    };

    void deserialize(const void* serialData, size_t serialLength);
};

/**
 * 融合注意力LayerNorm插件创建器
 */
class FusedAttentionLayerNormPluginCreator : public GRPluginCreatorBase {
public:
    FusedAttentionLayerNormPluginCreator();

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
    static constexpr const char* PLUGIN_NAME = "FusedAttentionLayerNorm";
    static constexpr const char* PLUGIN_VERSION = "1.0";
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra