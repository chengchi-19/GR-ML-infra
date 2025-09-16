#pragma once

#include "tensorrt_plugin_base.h"

namespace gr_ml_infra {
namespace tensorrt_plugins {

/**
 * 序列推荐交互 TensorRT插件
 * 对应Triton算子: sequence_recommendation_interaction
 *
 * 功能：
 * 1. 用户-物品交互分数计算
 * 2. 物品共现矩阵构建
 * 3. 短期/长期兴趣表示
 * 4. 协同过滤分数计算
 */
class SequenceRecommendationInteractionPlugin : public GRPluginBase {
public:
    SequenceRecommendationInteractionPlugin(
        int32_t seq_len,
        int32_t hidden_dim,
        int32_t short_window = 8,
        int32_t long_window = 32,
        float decay_factor = 0.1f,
        int32_t top_k = 8,
        float min_cooccur = 0.1f
    );

    // 反序列化构造函数
    SequenceRecommendationInteractionPlugin(const void* serialData, size_t serialLength);

    ~SequenceRecommendationInteractionPlugin() override = default;

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
    // 插件参数
    int32_t seq_len_;
    int32_t hidden_dim_;
    int32_t short_window_;
    int32_t long_window_;
    float decay_factor_;
    int32_t top_k_;
    float min_cooccur_;

    // CUDA kernel启动参数
    struct KernelParams {
        int32_t batch_size;
        int32_t seq_len;
        int32_t hidden_dim;
        int32_t short_window;
        int32_t long_window;
        float decay_factor;
        int32_t top_k;
        float min_cooccur;
    };

    void deserialize(const void* serialData, size_t serialLength);

    // 工作空间大小计算
    size_t calculateWorkspaceSize(int32_t batch_size, int32_t seq_len, int32_t hidden_dim) const;
};

/**
 * 序列推荐交互插件创建器
 */
class SequenceRecommendationInteractionPluginCreator : public GRPluginCreatorBase {
public:
    SequenceRecommendationInteractionPluginCreator();

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
    static constexpr const char* PLUGIN_NAME = "SequenceRecommendationInteraction";
    static constexpr const char* PLUGIN_VERSION = "1.0";

    // 插件属性定义
    std::vector<nvinfer1::PluginField> plugin_attributes_;
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra