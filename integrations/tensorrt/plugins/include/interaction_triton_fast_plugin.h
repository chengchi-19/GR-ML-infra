#pragma once

#include "tensorrt_plugin_base.h"
#include <string>
#include <vector>

namespace gr_ml_infra {
namespace tensorrt_plugins {

class InteractionTritonFastPlugin : public GRPluginBase {
public:
    InteractionTritonFastPlugin(int block_size);
    // 反序列化构造函数
    InteractionTritonFastPlugin(const void* serial_data, size_t serial_length);
    ~InteractionTritonFastPlugin() override = default;

    // IPluginV2DynamicExt 接口
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs, nvinfer1::IExprBuilder& expr_builder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs, int nb_outputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs, const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nb_inputs, const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* input_desc, const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 接口
    const char* getPluginType() const noexcept override;
    int getNbOutputs() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    int block_size_;
    void deserialize(const void* serialData, size_t serialLength);
};

class InteractionTritonFastPluginCreator : public GRPluginCreatorBase {
public:
    InteractionTritonFastPluginCreator();
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serial_data, size_t serial_length) noexcept override;
private:
    static constexpr const char* PLUGIN_NAME = "InteractionTritonFast";
    static constexpr const char* PLUGIN_VERSION = "1.0";
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra