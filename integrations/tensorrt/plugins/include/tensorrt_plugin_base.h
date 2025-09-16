#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>

namespace gr_ml_infra {
namespace tensorrt_plugins {

/**
 * GR-ML-infra TensorRT插件基类
 * 为所有自定义算子提供统一的插件接口
 */
class GRPluginBase : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit GRPluginBase(const std::string& name, const std::string& version = "1.0");
    virtual ~GRPluginBase() = default;

    // IPluginV2DynamicExt接口实现
    int32_t getNbOutputs() const noexcept override = 0;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override = 0;

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
    ) noexcept override = 0;

    // IPluginV2Ext接口
    nvinfer1::DataType getOutputDataType(
        int32_t index,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs
    ) const noexcept override;

    // IPluginV2接口
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override = 0;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    std::string plugin_name_;
    std::string plugin_version_;
    std::string plugin_namespace_;

    // 通用工具函数
    size_t calculateTensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType dtype) const;
    void logError(const std::string& message) const;
    void logInfo(const std::string& message) const;

    // CUDA错误检查宏
    #define CUDA_CHECK(call) do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            logError("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)
};

/**
 * 插件创建器基类
 */
class GRPluginCreatorBase : public nvinfer1::IPluginCreator {
public:
    explicit GRPluginCreatorBase(const std::string& name, const std::string& version = "1.0");
    virtual ~GRPluginCreatorBase() = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override = 0;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength
    ) noexcept override = 0;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    std::string plugin_name_;
    std::string plugin_version_;
    std::string plugin_namespace_;
    std::vector<nvinfer1::PluginField> plugin_fields_;
    nvinfer1::PluginFieldCollection field_collection_;

    // 参数解析工具
    template<typename T>
    T getPluginField(const nvinfer1::PluginFieldCollection* fc, const std::string& name, T defaultValue) const;
};

// 插件注册管理器
class PluginRegistry {
public:
    static PluginRegistry& getInstance();

    void registerPlugin(std::unique_ptr<GRPluginCreatorBase> creator);
    void registerAllPlugins();

    // 获取已注册的插件
    std::vector<std::string> getRegisteredPlugins() const;

private:
    PluginRegistry() = default;
    std::vector<std::unique_ptr<GRPluginCreatorBase>> creators_;
};

} // namespace tensorrt_plugins
} // namespace gr_ml_infra