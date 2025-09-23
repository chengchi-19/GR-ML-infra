#pragma once

#include "tensorrt_plugin_base.h"
#include <string>
#include <vector>

namespace gr_ml_infra {
namespace tensorrt_plugins {

/**
 * @brief InteractionTritonFast TensorRT插件
 * 对应Triton算子: interaction_triton_fast
 *
 * 功能:
 * 1. 实现高效的特征交互计算
 * 2. 支持动态batch size和特征维度
 * 3. 优化CUDA内核实现以提高性能
 *
 * 输入: [B, F, D] - batch_size, feature_count, embedding_dim
 * 输出: [B, F*(F-1)/2] - 特征交互后的结果
 */
class InteractionTritonFastPlugin : public GRPluginBase {
public:
    /**
     * @brief 构造函数
     * @param block_size CUDA线程块大小，用于优化内存访问
     */
    InteractionTritonFastPlugin(int32_t block_size);

    /**
     * @brief 反序列化构造函数
     * @param serial_data 序列化数据
     * @param serial_length 数据长度
     */
    InteractionTritonFastPlugin(const void* serial_data, size_t serial_length);

    ~InteractionTritonFastPlugin() override = default;

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
    int32_t block_size_;  ///< CUDA线程块大小

    // CUDA kernel启动参数结构
    struct KernelParams {
        int32_t batch_size;     ///< 批次大小
        int32_t feature_count;  ///< 特征数量
        int32_t embedding_dim;  ///< 嵌入维度
        int32_t block_size;     ///< 线程块大小
    };

    /**
     * @brief 反序列化内部数据
     * @param serialData 序列化数据
     * @param serialLength 数据长度
     */
    void deserialize(const void* serialData, size_t serialLength);
};

/**
 * @brief InteractionTritonFast插件创建器
 */
class InteractionTritonFastPluginCreator : public GRPluginCreatorBase {
public:
    InteractionTritonFastPluginCreator();

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
    static constexpr const char* PLUGIN_NAME = "InteractionTritonFast";
    static constexpr const char* PLUGIN_VERSION = "1.0";
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra