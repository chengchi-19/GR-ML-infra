#include "hstu_hierarchical_attention_plugin.h"
#include <cassert>
#include <cstring>
#include <iostream>

namespace gr_ml_infra {
namespace tensorrt_plugins {

// CUDA Kernel函数声明
extern "C" {
void hstu_hierarchical_attention_kernel_driver(
    const float* query, const float* key, const float* value, const float* level_mask,
    float* output, int B, int H, int S, int D, int num_levels, int level_size, cudaStream_t stream);
}

// HSTUHierarchicalAttentionPlugin 插件实现
HSTUHierarchicalAttentionPlugin::HSTUHierarchicalAttentionPlugin(int32_t num_levels, int32_t level_size)
    : GRPluginBase("HSTUHierarchicalAttention", "1.0"), num_levels_(num_levels), level_size_(level_size) {
    logInfo("初始化 HSTUHierarchicalAttentionPlugin，num_levels=" + std::to_string(num_levels_) +
            ", level_size=" + std::to_string(level_size_));
}

// 反序列化构造函数
HSTUHierarchicalAttentionPlugin::HSTUHierarchicalAttentionPlugin(const void* serial_data, size_t serial_length)
    : GRPluginBase("HSTUHierarchicalAttention", "1.0") {
    deserialize(serial_data, serial_length);
}

int32_t HSTUHierarchicalAttentionPlugin::getNbOutputs() const noexcept {
    return 1; // 分层注意力只输出一个结果张量
}

// 克隆插件实例
nvinfer1::IPluginV2DynamicExt* HSTUHierarchicalAttentionPlugin::clone() const noexcept {
    auto plugin = new HSTUHierarchicalAttentionPlugin(num_levels_, level_size_);
    plugin->setPluginNamespace(plugin_namespace_.c_str());
    return plugin;
}

// 获取输出维度
nvinfer1::DimsExprs HSTUHierarchicalAttentionPlugin::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder
) noexcept {
    assert(output_index == 0);
    assert(nb_inputs == 4); // query, key, value, level_mask

    // 输出维度与query维度相同: [B, H, S, D]
    return inputs[0];
}

// 检查格式支持
bool HSTUHierarchicalAttentionPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs
) noexcept {
    assert(nb_inputs == 4);  // query, key, value, level_mask
    assert(nb_outputs == 1); // output

    const auto& desc = in_out[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }

    // 检查输入输出数据类型一致性
    if (pos == 0) {
        // query: 必须是FP32格式
        return (desc.type == nvinfer1::DataType::kFLOAT);
    } else if (pos < nb_inputs + nb_outputs) {
        // 所有张量都必须与query保持一致的数据类型
        return (desc.type == in_out[0].type &&
                desc.format == in_out[0].format);
    }

    return false;
}

// 配置插件
void HSTUHierarchicalAttentionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nb_outputs
) noexcept {
    assert(nb_inputs == 4);  // query, key, value, level_mask
    assert(nb_outputs == 1); // output
}

// 获取工作空间大小
size_t HSTUHierarchicalAttentionPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nb_outputs
) const noexcept {
    return 0; // 此插件不需要额外工作空间
}

// 执行插件
int32_t HSTUHierarchicalAttentionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    // 获取输入维度信息
    const auto& query_dims = input_desc[0].dims;

    // 验证输入维度
    if (query_dims.nbDims != 4) {
        logError("输入张量维度必须为4D: [B, H, S, D]");
        return -1;
    }

    int32_t B = query_dims.d[0];  // batch_size
    int32_t H = query_dims.d[1];  // num_heads
    int32_t S = query_dims.d[2];  // seq_length
    int32_t D = query_dims.d[3];  // head_dim

    try {
        // 只支持FP32数据类型
        if (input_desc[0].type != nvinfer1::DataType::kFLOAT) {
            logError("不支持的数据类型，只支持FP32");
            return -1;
        }

        // 调用CUDA kernel
        hstu_hierarchical_attention_kernel_driver(
            static_cast<const float*>(inputs[0]),  // query
            static_cast<const float*>(inputs[1]),  // key
            static_cast<const float*>(inputs[2]),  // value
            static_cast<const float*>(inputs[3]),  // level_mask
            static_cast<float*>(outputs[0]),       // output
            B, H, S, D, num_levels_, level_size_, stream
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
size_t HSTUHierarchicalAttentionPlugin::getSerializationSize() const noexcept {
    return GRPluginBase::getSerializationSize() + sizeof(num_levels_) + sizeof(level_size_);
}

// 序列化插件
void HSTUHierarchicalAttentionPlugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    size_t offset = 0;

    GRPluginBase::serialize(buf);
    offset += GRPluginBase::getSerializationSize();

    std::memcpy(buf + offset, &num_levels_, sizeof(num_levels_));
    offset += sizeof(num_levels_);

    std::memcpy(buf + offset, &level_size_, sizeof(level_size_));
}

// 反序列化插件
void HSTUHierarchicalAttentionPlugin::deserialize(const void* serialData, size_t serialLength) {
    const char* buf = static_cast<const char*>(serialData);
    size_t offset = 0;

    offset += GRPluginBase::getSerializationSize();

    std::memcpy(&num_levels_, buf + offset, sizeof(num_levels_));
    offset += sizeof(num_levels_);

    std::memcpy(&level_size_, buf + offset, sizeof(level_size_));
}

// HSTUHierarchicalAttentionPluginCreator 插件创建器实现
HSTUHierarchicalAttentionPluginCreator::HSTUHierarchicalAttentionPluginCreator()
    : GRPluginCreatorBase(PLUGIN_NAME, PLUGIN_VERSION) {
    plugin_fields_.emplace_back("num_levels", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_fields_.emplace_back("level_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    field_collection_.nbFields = plugin_fields_.size();
    field_collection_.fields = plugin_fields_.data();
}

// 创建插件
nvinfer1::IPluginV2* HSTUHierarchicalAttentionPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc
) noexcept {
    try {
        int32_t num_levels = getPluginField<int32_t>(fc, "num_levels", 4);
        int32_t level_size = getPluginField<int32_t>(fc, "level_size", 0);
        auto plugin = new HSTUHierarchicalAttentionPlugin(num_levels, level_size);
        plugin->setPluginNamespace(plugin_namespace_.c_str());
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "创建HSTUHierarchicalAttentionPlugin失败: " << e.what() << std::endl;
        return nullptr;
    }
}

// 反序列化插件
nvinfer1::IPluginV2* HSTUHierarchicalAttentionPluginCreator::deserializePlugin(
    const char* name,
    const void* serial_data,
    size_t serial_length
) noexcept {
    try {
        auto plugin = new HSTUHierarchicalAttentionPlugin(serial_data, serial_length);
        plugin->setPluginNamespace(plugin_namespace_.c_str());
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "反序列化HSTUHierarchicalAttentionPlugin失败: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra