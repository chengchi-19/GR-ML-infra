#include "fused_attention_layernorm_plugin.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cassert>

namespace gr_ml_infra {
namespace tensorrt_plugins {

// CUDA Kernel声明
extern "C" {
    // FP32版本
    void launch_fused_attention_layernorm_kernel_fp32(
        const float* input,
        const float* weight_q,
        const float* weight_k,
        const float* weight_v,
        const float* weight_o,
        const float* layer_norm_weight,
        const float* layer_norm_bias,
        float* output,
        void* workspace,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int num_heads,
        int head_dim,
        float dropout_rate,
        float layer_norm_eps,
        cudaStream_t stream
    );

    // FP16版本
    void launch_fused_attention_layernorm_kernel_fp16(
        const __half* input,
        const __half* weight_q,
        const __half* weight_k,
        const __half* weight_v,
        const __half* weight_o,
        const __half* layer_norm_weight,
        const __half* layer_norm_bias,
        __half* output,
        void* workspace,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int num_heads,
        int head_dim,
        float dropout_rate,
        float layer_norm_eps,
        cudaStream_t stream
    );
}

// FusedAttentionLayerNormPlugin 实现
FusedAttentionLayerNormPlugin::FusedAttentionLayerNormPlugin(
    int32_t hidden_dim,
    int32_t num_heads,
    float dropout_rate,
    float layer_norm_eps
) : GRPluginBase("FusedAttentionLayerNorm", "1.0"),
    hidden_dim_(hidden_dim),
    num_heads_(num_heads),
    dropout_rate_(dropout_rate),
    layer_norm_eps_(layer_norm_eps) {

    assert(hidden_dim_ % num_heads_ == 0);
    head_dim_ = hidden_dim_ / num_heads_;

    logInfo("初始化 - hidden_dim=" + std::to_string(hidden_dim_) +
           ", num_heads=" + std::to_string(num_heads_) +
           ", head_dim=" + std::to_string(head_dim_));
}

FusedAttentionLayerNormPlugin::FusedAttentionLayerNormPlugin(
    const void* serialData,
    size_t serialLength
) : GRPluginBase("FusedAttentionLayerNorm", "1.0") {
    deserialize(serialData, serialLength);
}

int32_t FusedAttentionLayerNormPlugin::getNbOutputs() const noexcept {
    return 1; // 只输出融合后的结果
}

nvinfer1::DimsExprs FusedAttentionLayerNormPlugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    assert(outputIndex == 0);
    assert(nbInputs >= 1);

    // 输出维度与输入相同 [batch_size, seq_len, hidden_dim]
    return inputs[0];
}

bool FusedAttentionLayerNormPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    // 支持FP32和FP16
    const auto& desc = inOut[pos];

    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }

    bool isValidType = (desc.type == nvinfer1::DataType::kFLOAT ||
                       desc.type == nvinfer1::DataType::kHALF);

    if (!isValidType) {
        return false;
    }

    // 确保所有张量类型一致
    for (int32_t i = 0; i < nbInputs + nbOutputs; ++i) {
        if (inOut[i].type != desc.type) {
            return false;
        }
    }

    return true;
}

void FusedAttentionLayerNormPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs
) noexcept {
    // 验证输入参数
    assert(nbInputs >= 6); // input, Wq, Wk, Wv, Wo, ln_weight, ln_bias
    assert(nbOutputs == 1);

    logInfo("配置插件 - 输入数量: " + std::to_string(nbInputs));
}

size_t FusedAttentionLayerNormPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) const noexcept {
    // 计算workspace大小
    // 需要存储中间的Q, K, V矩阵和注意力分数

    const auto& input_dims = inputs[0].dims;
    int32_t batch_size = input_dims.d[0];
    int32_t seq_len = input_dims.d[1];

    size_t element_size = (inputs[0].type == nvinfer1::DataType::kHALF) ?
                         sizeof(__half) : sizeof(float);

    // Q, K, V 矩阵空间: 3 * [batch_size, num_heads, seq_len, head_dim]
    size_t qkv_size = 3 * batch_size * num_heads_ * seq_len * head_dim_ * element_size;

    // 注意力分数矩阵: [batch_size, num_heads, seq_len, seq_len]
    size_t attn_scores_size = batch_size * num_heads_ * seq_len * seq_len * element_size;

    // 临时输出空间: [batch_size, seq_len, hidden_dim]
    size_t temp_output_size = batch_size * seq_len * hidden_dim_ * element_size;

    size_t total_workspace = qkv_size + attn_scores_size + temp_output_size;

    // 对齐到256字节边界
    total_workspace = ((total_workspace + 255) / 256) * 256;

    return total_workspace;
}

int32_t FusedAttentionLayerNormPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    const auto& input_dims = inputDesc[0].dims;
    int32_t batch_size = input_dims.d[0];
    int32_t seq_len = input_dims.d[1];

    try {
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
            // FP32版本
            launch_fused_attention_layernorm_kernel_fp32(
                static_cast<const float*>(inputs[0]),  // input
                static_cast<const float*>(inputs[1]),  // weight_q
                static_cast<const float*>(inputs[2]),  // weight_k
                static_cast<const float*>(inputs[3]),  // weight_v
                static_cast<const float*>(inputs[4]),  // weight_o
                static_cast<const float*>(inputs[5]),  // layer_norm_weight
                static_cast<const float*>(inputs[6]),  // layer_norm_bias
                static_cast<float*>(outputs[0]),       // output
                workspace,
                batch_size,
                seq_len,
                hidden_dim_,
                num_heads_,
                head_dim_,
                dropout_rate_,
                layer_norm_eps_,
                stream
            );
        } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
            // FP16版本
            launch_fused_attention_layernorm_kernel_fp16(
                static_cast<const __half*>(inputs[0]),  // input
                static_cast<const __half*>(inputs[1]),  // weight_q
                static_cast<const __half*>(inputs[2]),  // weight_k
                static_cast<const __half*>(inputs[3]),  // weight_v
                static_cast<const __half*>(inputs[4]),  // weight_o
                static_cast<const __half*>(inputs[5]),  // layer_norm_weight
                static_cast<const __half*>(inputs[6]),  // layer_norm_bias
                static_cast<__half*>(outputs[0]),       // output
                workspace,
                batch_size,
                seq_len,
                hidden_dim_,
                num_heads_,
                head_dim_,
                dropout_rate_,
                layer_norm_eps_,
                stream
            );
        } else {
            logError("不支持的数据类型");
            return -1;
        }

        CUDA_CHECK(cudaGetLastError());
        return 0;

    } catch (const std::exception& e) {
        logError("Kernel执行失败: " + std::string(e.what()));
        return -1;
    }
}

nvinfer1::IPluginV2DynamicExt* FusedAttentionLayerNormPlugin::clone() const noexcept {
    auto plugin = new FusedAttentionLayerNormPlugin(
        hidden_dim_, num_heads_, dropout_rate_, layer_norm_eps_);
    plugin->setPluginNamespace(plugin_namespace_.c_str());
    return plugin;
}

size_t FusedAttentionLayerNormPlugin::getSerializationSize() const noexcept {
    return GRPluginBase::getSerializationSize() +
           4 * sizeof(int32_t) + 2 * sizeof(float); // hidden_dim, num_heads, head_dim, 未使用, dropout_rate, layer_norm_eps
}

void FusedAttentionLayerNormPlugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    size_t offset = 0;

    // 调用基类序列化
    GRPluginBase::serialize(buf);
    offset += GRPluginBase::getSerializationSize();

    // 序列化插件特定参数
    std::memcpy(buf + offset, &hidden_dim_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(buf + offset, &num_heads_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(buf + offset, &head_dim_, sizeof(int32_t));
    offset += sizeof(int32_t);

    int32_t reserved = 0; // 保留字段
    std::memcpy(buf + offset, &reserved, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(buf + offset, &dropout_rate_, sizeof(float));
    offset += sizeof(float);

    std::memcpy(buf + offset, &layer_norm_eps_, sizeof(float));
}

void FusedAttentionLayerNormPlugin::deserialize(const void* serialData, size_t serialLength) {
    const char* buf = static_cast<const char*>(serialData);
    size_t offset = 0;

    // 跳过基类序列化数据
    offset += GRPluginBase::getSerializationSize();

    // 反序列化插件特定参数
    std::memcpy(&hidden_dim_, buf + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&num_heads_, buf + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&head_dim_, buf + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    int32_t reserved;
    std::memcpy(&reserved, buf + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&dropout_rate_, buf + offset, sizeof(float));
    offset += sizeof(float);

    std::memcpy(&layer_norm_eps_, buf + offset, sizeof(float));

    logInfo("反序列化完成 - hidden_dim=" + std::to_string(hidden_dim_) +
           ", num_heads=" + std::to_string(num_heads_));
}

// FusedAttentionLayerNormPluginCreator 实现
FusedAttentionLayerNormPluginCreator::FusedAttentionLayerNormPluginCreator()
    : GRPluginCreatorBase(PLUGIN_NAME, PLUGIN_VERSION) {

    // 定义插件字段
    plugin_fields_.emplace_back("hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_fields_.emplace_back("num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_fields_.emplace_back("dropout_rate", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    plugin_fields_.emplace_back("layer_norm_eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);

    field_collection_.nbFields = plugin_fields_.size();
    field_collection_.fields = plugin_fields_.data();
}

nvinfer1::IPluginV2* FusedAttentionLayerNormPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc
) noexcept {
    try {
        // 解析参数
        int32_t hidden_dim = getPluginField<int32_t>(fc, "hidden_dim", 1024);
        int32_t num_heads = getPluginField<int32_t>(fc, "num_heads", 16);
        float dropout_rate = getPluginField<float>(fc, "dropout_rate", 0.1f);
        float layer_norm_eps = getPluginField<float>(fc, "layer_norm_eps", 1e-5f);

        auto plugin = new FusedAttentionLayerNormPlugin(
            hidden_dim, num_heads, dropout_rate, layer_norm_eps);
        plugin->setPluginNamespace(plugin_namespace_.c_str());

        return plugin;

    } catch (const std::exception& e) {
        std::cerr << "创建FusedAttentionLayerNorm插件失败: " << e.what() << std::endl;
        return nullptr;
    }
}

nvinfer1::IPluginV2* FusedAttentionLayerNormPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength
) noexcept {
    try {
        auto plugin = new FusedAttentionLayerNormPlugin(serialData, serialLength);
        plugin->setPluginNamespace(plugin_namespace_.c_str());
        return plugin;

    } catch (const std::exception& e) {
        std::cerr << "反序列化FusedAttentionLayerNorm插件失败: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra