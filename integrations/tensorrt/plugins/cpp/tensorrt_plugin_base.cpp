#include "tensorrt_plugin_base.h"
#include <cassert>
#include <iostream>
#include <sstream>

namespace gr_ml_infra {
namespace tensorrt_plugins {

// GRPluginBase 实现
GRPluginBase::GRPluginBase(const std::string& name, const std::string& version)
    : plugin_name_(name), plugin_version_(version), plugin_namespace_("GR_ML_INFRA") {
}

bool GRPluginBase::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    // 默认支持FP32和FP16，输入输出格式一致
    const auto& desc = inOut[pos];

    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }

    if (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF) {
        // 检查所有张量类型一致
        for (int32_t i = 0; i < nbInputs + nbOutputs; ++i) {
            if (inOut[i].type != desc.type) {
                return false;
            }
        }
        return true;
    }

    return false;
}

void GRPluginBase::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs
) noexcept {
    // 基类默认实现为空，子类可以重写
}

size_t GRPluginBase::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) const noexcept {
    // 默认不需要额外workspace
    return 0;
}

nvinfer1::DataType GRPluginBase::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs
) const noexcept {
    // 默认输出类型与第一个输入相同
    return nbInputs > 0 ? inputTypes[0] : nvinfer1::DataType::kFLOAT;
}

const char* GRPluginBase::getPluginType() const noexcept {
    return plugin_name_.c_str();
}

const char* GRPluginBase::getPluginVersion() const noexcept {
    return plugin_version_.c_str();
}

int32_t GRPluginBase::initialize() noexcept {
    return 0;
}

void GRPluginBase::terminate() noexcept {
    // 基类默认实现为空
}

size_t GRPluginBase::getSerializationSize() const noexcept {
    // 基类只序列化名称和版本
    return plugin_name_.size() + plugin_version_.size() + 2 * sizeof(size_t);
}

void GRPluginBase::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    size_t offset = 0;

    // 序列化插件名称
    size_t name_size = plugin_name_.size();
    std::memcpy(buf + offset, &name_size, sizeof(size_t));
    offset += sizeof(size_t);
    std::memcpy(buf + offset, plugin_name_.c_str(), name_size);
    offset += name_size;

    // 序列化插件版本
    size_t version_size = plugin_version_.size();
    std::memcpy(buf + offset, &version_size, sizeof(size_t));
    offset += sizeof(size_t);
    std::memcpy(buf + offset, plugin_version_.c_str(), version_size);
}

void GRPluginBase::destroy() noexcept {
    delete this;
}

void GRPluginBase::setPluginNamespace(const char* pluginNamespace) noexcept {
    plugin_namespace_ = pluginNamespace;
}

const char* GRPluginBase::getPluginNamespace() const noexcept {
    return plugin_namespace_.c_str();
}

size_t GRPluginBase::calculateTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType dtype) const {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }

    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return size * sizeof(float);
        case nvinfer1::DataType::kHALF:
            return size * sizeof(__half);
        case nvinfer1::DataType::kINT8:
            return size * sizeof(int8_t);
        case nvinfer1::DataType::kINT32:
            return size * sizeof(int32_t);
        default:
            return size * sizeof(float);
    }
}

void GRPluginBase::logError(const std::string& message) const {
    std::cerr << "[GR-ML-Infra Plugin Error] " << plugin_name_ << ": " << message << std::endl;
}

void GRPluginBase::logInfo(const std::string& message) const {
    std::cout << "[GR-ML-Infra Plugin Info] " << plugin_name_ << ": " << message << std::endl;
}

// GRPluginCreatorBase 实现
GRPluginCreatorBase::GRPluginCreatorBase(const std::string& name, const std::string& version)
    : plugin_name_(name), plugin_version_(version), plugin_namespace_("GR_ML_INFRA") {
    field_collection_.nbFields = 0;
    field_collection_.fields = nullptr;
}

const char* GRPluginCreatorBase::getPluginName() const noexcept {
    return plugin_name_.c_str();
}

const char* GRPluginCreatorBase::getPluginVersion() const noexcept {
    return plugin_version_.c_str();
}

const nvinfer1::PluginFieldCollection* GRPluginCreatorBase::getFieldNames() noexcept {
    return &field_collection_;
}

void GRPluginCreatorBase::setPluginNamespace(const char* pluginNamespace) noexcept {
    plugin_namespace_ = pluginNamespace;
}

const char* GRPluginCreatorBase::getPluginNamespace() const noexcept {
    return plugin_namespace_.c_str();
}

template<typename T>
T GRPluginCreatorBase::getPluginField(
    const nvinfer1::PluginFieldCollection* fc,
    const std::string& name,
    T defaultValue
) const {
    for (int i = 0; i < fc->nbFields; ++i) {
        if (fc->fields[i].name == name) {
            return *static_cast<const T*>(fc->fields[i].data);
        }
    }
    return defaultValue;
}

// 显式实例化模板
template int GRPluginCreatorBase::getPluginField<int>(
    const nvinfer1::PluginFieldCollection*, const std::string&, int) const;
template float GRPluginCreatorBase::getPluginField<float>(
    const nvinfer1::PluginFieldCollection*, const std::string&, float) const;
template bool GRPluginCreatorBase::getPluginField<bool>(
    const nvinfer1::PluginFieldCollection*, const std::string&, bool) const;

// PluginRegistry 实现
PluginRegistry& PluginRegistry::getInstance() {
    static PluginRegistry instance;
    return instance;
}

void PluginRegistry::registerPlugin(std::unique_ptr<GRPluginCreatorBase> creator) {
    auto* registry = getPluginRegistry();
    if (registry) {
        registry->registerCreator(*creator, creator->getPluginNamespace());
        creators_.push_back(std::move(creator));
        std::cout << "[GR-ML-Infra] 注册插件: " << creators_.back()->getPluginName() << std::endl;
    }
}

void PluginRegistry::registerAllPlugins() {
    std::cout << "[GR-ML-Infra] 开始注册所有TensorRT插件..." << std::endl;

    // 这里将注册所有插件
    // 具体插件注册将在实现各个插件时添加

    std::cout << "[GR-ML-Infra] 插件注册完成，共注册 " << creators_.size() << " 个插件" << std::endl;
}

std::vector<std::string> PluginRegistry::getRegisteredPlugins() const {
    std::vector<std::string> names;
    for (const auto& creator : creators_) {
        names.push_back(creator->getPluginName());
    }
    return names;
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra