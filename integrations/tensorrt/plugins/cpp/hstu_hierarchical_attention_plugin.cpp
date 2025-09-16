#include "hstu_hierarchical_attention_plugin.h"
#include "tensorrt_plugin_utils.h"
#include <cuda_runtime.h>

namespace gr {
namespace ml {
namespace infra {

extern "C" {
void hstu_hierarchical_attention_kernel_driver(const float* query, const float* key, const float* value, const float* level_mask, float* output, int B, int H, int S, int D, int num_levels, int level_size, cudaStream_t stream);
}

// HSTUHierarchicalAttentionPlugin
HSTUHierarchicalAttentionPlugin::HSTUHierarchicalAttentionPlugin(const std::string& name, int num_levels, int level_size)
    : layer_name_(name), num_levels_(num_levels), level_size_(level_size) {}

HSTUHierarchicalAttentionPlugin::HSTUHierarchicalAttentionPlugin(const std::string& name, const void* serial_data, size_t serial_length)
    : layer_name_(name) {
    deserialize_value(&serial_data, &serial_length, &num_levels_);
    deserialize_value(&serial_data, &serial_length, &level_size_);
}

const char* HSTUHierarchicalAttentionPlugin::getPluginType() const noexcept {
    return "HSTUHierarchicalAttention";
}

const char* HSTUHierarchicalAttentionPlugin::getPluginVersion() const noexcept {
    return "1";
}

int HSTUHierarchicalAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

size_t HSTUHierarchicalAttentionPlugin::getSerializationSize() const noexcept {
    return sizeof(num_levels_) + sizeof(level_size_);
}

void HSTUHierarchicalAttentionPlugin::serialize(void* buffer) const noexcept {
    serialize_value(&buffer, num_levels_);
    serialize_value(&buffer, level_size_);
}

IPluginV2DynamicExt* HSTUHierarchicalAttentionPlugin::clone() const noexcept {
    auto plugin = new HSTUHierarchicalAttentionPlugin(layer_name_, num_levels_, level_size_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs HSTUHierarchicalAttentionPlugin::getOutputDimensions(int output_index, const DimsExprs* inputs, int nb_inputs, IExprBuilder& expr_builder) noexcept {
    return inputs[0];
}

bool HSTUHierarchicalAttentionPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* in_out, int nb_inputs, int nb_outputs) noexcept {
    if (in_out[pos].format != TensorFormat::kLINEAR)
        return false;

    if (pos >= 0 && pos <= 3) { // query, key, value, level_mask
        return (in_out[pos].type == DataType::kFLOAT);
    }
    if (pos == 4) { // output
        return (in_out[pos].type == DataType::kFLOAT);
    }
    return false;
}

void HSTUHierarchicalAttentionPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nb_inputs, const DynamicPluginTensorDesc* out, int nb_outputs) noexcept {}

size_t HSTUHierarchicalAttentionPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nb_inputs, const PluginTensorDesc* outputs, int nb_outputs) const noexcept {
    return 0;
}

int HSTUHierarchicalAttentionPlugin::enqueue(const PluginTensorDesc* input_desc, const PluginTensorDesc* output_desc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    // Placeholder for kernel launch
    return 0;
}

// HSTUHierarchicalAttentionPluginCreator
HSTUHierarchicalAttentionPluginCreator::HSTUHierarchicalAttentionPluginCreator() {
    mPluginAttributes.emplace_back(PluginField("num_levels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("level_size", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* HSTUHierarchicalAttentionPluginCreator::getPluginName() const noexcept {
    return "HSTUHierarchicalAttention";
}

const char* HSTUHierarchicalAttentionPluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const PluginFieldCollection* HSTUHierarchicalAttentionPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* HSTUHierarchicalAttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    const PluginField* fields = fc->fields;
    int num_levels = 4;
    int level_size = 0;
    for (int i = 0; i < fc->nbFields; ++i) {
        if (strcmp(fields[i].name, "num_levels") == 0) {
            num_levels = *static_cast<const int*>(fields[i].data);
        }
        if (strcmp(fields[i].name, "level_size") == 0) {
            level_size = *static_cast<const int*>(fields[i].data);
        }
    }
    return new HSTUHierarchicalAttentionPlugin(name, num_levels, level_size);
}

IPluginV2* HSTUHierarchicalAttentionPluginCreator::deserializePlugin(const char* name, const void* serial_data, size_t serial_length) noexcept {
    return new HSTUHierarchicalAttentionPlugin(name, serial_data, serial_length);
}

} // namespace infra
} // namespace ml
} // namespace gr