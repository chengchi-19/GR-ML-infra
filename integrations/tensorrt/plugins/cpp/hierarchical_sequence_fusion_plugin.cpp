#include "hierarchical_sequence_fusion_plugin.h"
#include <cassert>

// Kernel启动函数的C接口声明
extern "C" void launch_hierarchical_fusion(
    const void* input,
    void* output,
    const int* level_masks,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_levels,
    float fusion_strength,
    nvinfer1::DataType dtype,
    cudaStream_t stream
);

namespace gr_ml_infra {
namespace tensorrt_plugins {

// 构造函数: 用于插件创建
HierarchicalSequenceFusionPlugin::HierarchicalSequenceFusionPlugin(
    int32_t hidden_dim, int32_t num_levels, float fusion_strength
) : GRPluginBase("HierarchicalSequenceFusion", "1.0"),
    hidden_dim_(hidden_dim),
    num_levels_(num_levels),
    fusion_strength_(fusion_strength) {}

// 构造函数: 用于反序列化
HierarchicalSequenceFusionPlugin::HierarchicalSequenceFusionPlugin(
    const void* serialData, size_t serialLength
) : GRPluginBase("HierarchicalSequenceFusion", "1.0") {
    deserialize(serialData, serialLength);
}

// 获取输出张量的维度
nvinfer1::DimsExprs HierarchicalSequenceFusionPlugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    // 输入: 0: sequence_tensor [B, S, D], 1: level_masks [B, S, L] (可选)
    // 输出维度与输入序列张量完全相同
    assert(outputIndex == 0 && (nbInputs == 1 || nbInputs == 2));
    return inputs[0];
}

// 执行插件的核心逻辑
int32_t HierarchicalSequenceFusionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    try {
        const auto& dims = inputDesc[0].dims;
        // 如果提供了第二个输入(level_masks)，则获取其指针，否则为nullptr
        const int* level_masks = (nbInputs > 1 && inputDesc[1].dims.nbDims > 0) ? static_cast<const int*>(inputs[1]) : nullptr;

        // 调用外部的CUDA Kernel启动函数
        launch_hierarchical_fusion(
            inputs[0],           // 主输入
            outputs[0],          // 主输出
            level_masks,         // 可选的掩码输入
            dims.d[0],           // batch_size
            dims.d[1],           // seq_len
            hidden_dim_,
            num_levels_,
            fusion_strength_,
            inputDesc[0].type,   // 数据类型 (FP32/FP16)
            stream
        );
        return 0;
    } catch (const std::exception& e) {
        logError("Enqueue failed in HierarchicalSequenceFusionPlugin: " + std::string(e.what()));
        return -1;
    }
}

// 获取序列化时需要的大小
size_t HierarchicalSequenceFusionPlugin::getSerializationSize() const noexcept {
    return sizeof(hidden_dim_) + sizeof(num_levels_) + sizeof(fusion_strength_);
}

// 序列化插件状态
void HierarchicalSequenceFusionPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    size_t offset = 0;
    std::memcpy(d + offset, &hidden_dim_, sizeof(hidden_dim_));
    offset += sizeof(hidden_dim_);
    std::memcpy(d + offset, &num_levels_, sizeof(num_levels_));
    offset += sizeof(num_levels_);
    std::memcpy(d + offset, &fusion_strength_, sizeof(fusion_strength_));
}

// 反序列化插件状态
void HierarchicalSequenceFusionPlugin::deserialize(const void* serialData, size_t serialLength) {
    const char* d = static_cast<const char*>(serialData);
    size_t offset = 0;
    std::memcpy(&hidden_dim_, d + offset, sizeof(hidden_dim_));
    offset += sizeof(hidden_dim_);
    std::memcpy(&num_levels_, d + offset, sizeof(num_levels_));
    offset += sizeof(num_levels_);
    std::memcpy(&fusion_strength_, d + offset, sizeof(fusion_strength_));
}

// 插件创建器实现
HierarchicalSequenceFusionPluginCreator::HierarchicalSequenceFusionPluginCreator()
    : GRPluginCreatorBase("HierarchicalSequenceFusion", "1.0") {
    mPluginAttributes.emplace_back(nvinfer1::PluginField("hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("num_levels", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("fusion_strength", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

// 通过PluginFieldCollection创建插件实例
nvinfer1::IPluginV2* HierarchicalSequenceFusionPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc
) noexcept {
    try {
        int hidden_dim = 1024;
        int num_levels = 3;
        float fusion_strength = 0.5f;

        for (int i = 0; i < fc->nbFields; ++i) {
            const nvinfer1::PluginField* field = &fc->fields[i];
            if (strcmp(field->name, "hidden_dim") == 0) {
                hidden_dim = *static_cast<const int*>(field->data);
            }
            if (strcmp(field->name, "num_levels") == 0) {
                num_levels = *static_cast<const int*>(field->data);
            }
            if (strcmp(field->name, "fusion_strength") == 0) {
                fusion_strength = *static_cast<const float*>(field->data);
            }
        }
        auto plugin = new HierarchicalSequenceFusionPlugin(hidden_dim, num_levels, fusion_strength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "Create HierarchicalSequenceFusionPlugin failed: " << e.what() << std::endl;
        return nullptr;
    }
}

// 通过序列化数据创建插件实例
nvinfer1::IPluginV2* HierarchicalSequenceFusionPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength
) noexcept {
    try {
        auto plugin = new HierarchicalSequenceFusionPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "Deserialize HierarchicalSequenceFusionPlugin failed: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra