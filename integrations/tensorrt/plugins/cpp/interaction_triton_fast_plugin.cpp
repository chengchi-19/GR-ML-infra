#include "interaction_triton_fast_plugin.h"
#include <cassert>
#include <cstring>
#include <iostream>

namespace gr_ml_infra {
namespace tensorrt_plugins {

// CUDA Kernel函数声明
extern "C" {
void interaction_kernel_driver(const float* emb, float* out, int B, int F, int D, cudaStream_t stream);
}

// Kernel头文件引用
void interaction_kernel_driver(const float* emb, float* out, int B, int F, int D, cudaStream_t stream);

// InteractionTritonFastPlugin 插件实现
InteractionTritonFastPlugin::InteractionTritonFastPlugin(int32_t block_size)
    : GRPluginBase("InteractionTritonFast", "1.0"), block_size_(block_size) {
    logInfo("初始化 InteractionTritonFastPlugin，block_size=" + std::to_string(block_size_));
}

// 反序列化构造函数
InteractionTritonFastPlugin::InteractionTritonFastPlugin(const void* serial_data, size_t serial_length)
    : GRPluginBase("InteractionTritonFast", "1.0") {
    deserialize(serial_data, serial_length);
}


int32_t InteractionTritonFastPlugin::getNbOutputs() const noexcept {
    return 1; // 特征交互后只输出一个结果张量
}

// 克隆插件实例
nvinfer1::IPluginV2DynamicExt* InteractionTritonFastPlugin::clone() const noexcept {
    auto plugin = new InteractionTritonFastPlugin(block_size_);
    plugin->setPluginNamespace(plugin_namespace_.c_str());
    return plugin;
}

// 获取输出维度
nvinfer1::DimsExprs InteractionTritonFastPlugin::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder
) noexcept {
    assert(output_index == 0);
    assert(nb_inputs == 1);
    nvinfer1::DimsExprs output_dims;
    output_dims.nbDims = 2;
    output_dims.d[0] = inputs[0].d[0]; // Batch size保持不变
    const auto F_expr = inputs[0].d[1]; // 特征数量F
    if (F_expr->isConstant()) {
        int F = F_expr->getConstantValue();
        output_dims.d[1] = expr_builder.constant(F * (F - 1) / 2); // 输出维度为 F*(F-1)/2
    } else {
        // 处理动态形状
        auto one = expr_builder.constant(1);
        auto two = expr_builder.constant(2);
        auto F_minus_1 = expr_builder.operation(nvinfer1::ElementWiseOperation::kSUB, *F_expr, *one);
        auto F_times_F_minus_1 = expr_builder.operation(nvinfer1::ElementWiseOperation::kPROD, *F_expr, *F_minus_1);
        output_dims.d[1] = expr_builder.operation(nvinfer1::ElementWiseOperation::kFLOOR_DIV, *F_times_F_minus_1, *two);
    }
    return output_dims;
}

// 检查格式支持
bool InteractionTritonFastPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs
) noexcept {
    const auto& desc = in_out[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    // 只支持FP32
    bool isValidType = (desc.type == nvinfer1::DataType::kFLOAT);
    if (!isValidType) return false;

    // 检查所有张量的格式是否一致
    if (pos == 0) {
        // 输入张量：必须是FP32格式
        return (desc.type == nvinfer1::DataType::kFLOAT &&
                desc.format == nvinfer1::TensorFormat::kLINEAR);
    } else if (pos == 1) {
        // 输出张量：必须与输入保持一致
        return (desc.type == in_out[0].type &&
                desc.format == in_out[0].format);
    }
    return true;
}

// 配置插件
void InteractionTritonFastPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nb_outputs
) noexcept {
    assert(nb_inputs == 1);
    assert(nb_outputs == 1);
}

// 获取工作空间大小
size_t InteractionTritonFastPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nb_outputs
) const noexcept {
    return 0; // 此插件不需要额外工作空间
}

// 执行插件
int32_t InteractionTritonFastPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    const auto& input_dims = input_desc[0].dims;

    // 验证输入维度
    if (input_dims.nbDims != 3) {
        logError("输入张量维度必须为3D: [B, F, D]");
        return -1;
    }

    int32_t B = input_dims.d[0];  // batch_size
    int32_t F = input_dims.d[1];  // feature_count
    int32_t D = input_dims.d[2];  // embedding_dim

    try {
        // 只支持FP32数据类型
        if (input_desc[0].type != nvinfer1::DataType::kFLOAT) {
            logError("不支持的数据类型，只支持FP32");
            return -1;
        }

        // 调用CUDA kernel
        interaction_kernel_driver(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]),
            B, F, D, stream
        );

        // 检查CUDA错误
        CUDA_CHECK(cudaGetLastError());
        return 0;

    } catch (const std::exception& e) {
        logError("Kernel执行失败: " + std::string(e.what()));
        return -1;
    }
}

// 获取序列化大小
size_t InteractionTritonFastPlugin::getSerializationSize() const noexcept {
    return GRPluginBase::getSerializationSize() + sizeof(block_size_);
}

// 序列化插件
void InteractionTritonFastPlugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    size_t offset = 0;
    GRPluginBase::serialize(buf);
    offset += GRPluginBase::getSerializationSize();
    std::memcpy(buf + offset, &block_size_, sizeof(block_size_));
}

// 反序列化插件
void InteractionTritonFastPlugin::deserialize(const void* serialData, size_t serialLength) {
    const char* buf = static_cast<const char*>(serialData);
    size_t offset = 0;
    offset += GRPluginBase::getSerializationSize();
    std::memcpy(&block_size_, buf + offset, sizeof(block_size_));
}

// InteractionTritonFastPluginCreator 插件创建器实现
InteractionTritonFastPluginCreator::InteractionTritonFastPluginCreator()
    : GRPluginCreatorBase(PLUGIN_NAME, PLUGIN_VERSION) {
    plugin_fields_.emplace_back("block_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    field_collection_.nbFields = plugin_fields_.size();
    field_collection_.fields = plugin_fields_.data();
}

// 创建插件
nvinfer1::IPluginV2* InteractionTritonFastPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
    try {
        int32_t block_size = getPluginField<int32_t>(fc, "block_size", 64);
        auto plugin = new InteractionTritonFastPlugin(block_size);
        plugin->setPluginNamespace(plugin_namespace_.c_str());
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "创建InteractionTritonFastPlugin失败: " << e.what() << std::endl;
        return nullptr;
    }
}

// 反序列化插件
nvinfer1::IPluginV2* InteractionTritonFastPluginCreator::deserializePlugin(const char* name, const void* serial_data, size_t serial_length) noexcept {
    try {
        auto plugin = new InteractionTritonFastPlugin(serial_data, serial_length);
        plugin->setPluginNamespace(plugin_namespace_.c_str());
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "反序列化InteractionTritonFastPlugin失败: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra