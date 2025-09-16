#include "sequence_recommendation_interaction_plugin.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cassert>
#include <cstring>

namespace gr_ml_infra {
namespace tensorrt_plugins {

// CUDA Kernel声明
extern "C" {
    // FP32版本
    void launch_sequence_recommendation_interaction_kernel_fp32(
        const float* item_embeddings,
        const float* user_embedding,
        const float* time_weights,
        const float* interaction_mask,
        float* user_item_scores,
        float* item_cooccur,
        float* short_term,
        float* long_term,
        float* cf_scores,
        float* neighbor_weights,
        void* workspace,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int short_window,
        int long_window,
        float decay_factor,
        int top_k,
        float min_cooccur,
        cudaStream_t stream
    );

    // FP16版本
    void launch_sequence_recommendation_interaction_kernel_fp16(
        const __half* item_embeddings,
        const __half* user_embedding,
        const __half* time_weights,
        const __half* interaction_mask,
        __half* user_item_scores,
        __half* item_cooccur,
        __half* short_term,
        __half* long_term,
        __half* cf_scores,
        __half* neighbor_weights,
        void* workspace,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int short_window,
        int long_window,
        float decay_factor,
        int top_k,
        float min_cooccur,
        cudaStream_t stream
    );
}

// SequenceRecommendationInteractionPlugin 实现
SequenceRecommendationInteractionPlugin::SequenceRecommendationInteractionPlugin(
    int32_t seq_len,
    int32_t hidden_dim,
    int32_t short_window,
    int32_t long_window,
    float decay_factor,
    int32_t top_k,
    float min_cooccur
) : GRPluginBase("SequenceRecommendationInteraction", "1.0"),
    seq_len_(seq_len),
    hidden_dim_(hidden_dim),
    short_window_(short_window),
    long_window_(long_window),
    decay_factor_(decay_factor),
    top_k_(top_k),
    min_cooccur_(min_cooccur) {

    logInfo("初始化序列推荐交互插件 - seq_len=" + std::to_string(seq_len_) +
           ", hidden_dim=" + std::to_string(hidden_dim_) +
           ", short_window=" + std::to_string(short_window_) +
           ", top_k=" + std::to_string(top_k_));
}

SequenceRecommendationInteractionPlugin::SequenceRecommendationInteractionPlugin(
    const void* serialData,
    size_t serialLength
) : GRPluginBase("SequenceRecommendationInteraction", "1.0") {
    deserialize(serialData, serialLength);
}

int32_t SequenceRecommendationInteractionPlugin::getNbOutputs() const noexcept {
    return 6; // 6个输出：user_item_scores, item_cooccur, short_term, long_term, cf_scores, neighbor_weights
}

nvinfer1::DimsExprs SequenceRecommendationInteractionPlugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    assert(nbInputs == 4); // item_embeddings, user_embedding, time_weights, interaction_mask
    assert(outputIndex >= 0 && outputIndex < 6);

    const auto& batch_size = inputs[0].d[0];
    const auto& seq_len = inputs[0].d[1];
    const auto& hidden_dim = inputs[0].d[2];

    nvinfer1::DimsExprs output;

    switch (outputIndex) {
    case 0: // user_item_scores: [B, S]
        output.nbDims = 2;
        output.d[0] = batch_size;
        output.d[1] = seq_len;
        break;
    case 1: // item_cooccur: [B, S, S]
        output.nbDims = 3;
        output.d[0] = batch_size;
        output.d[1] = seq_len;
        output.d[2] = seq_len;
        break;
    case 2: // short_term: [B, D]
    case 3: // long_term: [B, D]
        output.nbDims = 2;
        output.d[0] = batch_size;
        output.d[1] = hidden_dim;
        break;
    case 4: // cf_scores: [B, S]
        output.nbDims = 2;
        output.d[0] = batch_size;
        output.d[1] = seq_len;
        break;
    case 5: // neighbor_weights: [B, S, TOP_K]
        output.nbDims = 3;
        output.d[0] = batch_size;
        output.d[1] = seq_len;
        output.d[2] = exprBuilder.constant(top_k_);
        break;
    }

    return output;
}

bool SequenceRecommendationInteractionPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    assert(nbInputs == 4 && nbOutputs == 6);

    const nvinfer1::PluginTensorDesc& desc = inOut[pos];

    // 支持FP32和FP16
    if (desc.type != nvinfer1::DataType::kFLOAT && desc.type != nvinfer1::DataType::kHALF) {
        return false;
    }

    // 所有输入输出必须使用相同的数据类型
    if (pos > 0) {
        return desc.type == inOut[0].type && desc.format == inOut[0].format;
    }

    return desc.format == nvinfer1::TensorFormat::kLINEAR;
}

void SequenceRecommendationInteractionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs
) noexcept {
    assert(nbInputs == 4 && nbOutputs == 6);

    logInfo("配置插件 - 输入维度: [" +
           std::to_string(in[0].desc.dims.d[0]) + ", " +
           std::to_string(in[0].desc.dims.d[1]) + ", " +
           std::to_string(in[0].desc.dims.d[2]) + "]");
}

size_t SequenceRecommendationInteractionPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) const noexcept {
    assert(nbInputs == 4 && nbOutputs == 6);

    const auto& dims = inputs[0].dims;
    int32_t batch_size = dims.d[0];
    int32_t seq_len = dims.d[1];
    int32_t hidden_dim = dims.d[2];

    return calculateWorkspaceSize(batch_size, seq_len, hidden_dim);
}

int32_t SequenceRecommendationInteractionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    const auto& dims = inputDesc[0].dims;
    int32_t batch_size = dims.d[0];
    int32_t seq_len = dims.d[1];
    int32_t hidden_dim = dims.d[2];

    KernelParams params;
    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.short_window = short_window_;
    params.long_window = long_window_;
    params.decay_factor = decay_factor_;
    params.top_k = top_k_;
    params.min_cooccur = min_cooccur_;

    // 根据数据类型调用相应的kernel
    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        launch_sequence_recommendation_interaction_kernel_fp32(
            static_cast<const float*>(inputs[0]),          // item_embeddings
            static_cast<const float*>(inputs[1]),          // user_embedding
            static_cast<const float*>(inputs[2]),          // time_weights
            static_cast<const float*>(inputs[3]),          // interaction_mask
            static_cast<float*>(outputs[0]),               // user_item_scores
            static_cast<float*>(outputs[1]),               // item_cooccur
            static_cast<float*>(outputs[2]),               // short_term
            static_cast<float*>(outputs[3]),               // long_term
            static_cast<float*>(outputs[4]),               // cf_scores
            static_cast<float*>(outputs[5]),               // neighbor_weights
            workspace,
            params.batch_size,
            params.seq_len,
            params.hidden_dim,
            params.short_window,
            params.long_window,
            params.decay_factor,
            params.top_k,
            params.min_cooccur,
            stream
        );
    } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
        launch_sequence_recommendation_interaction_kernel_fp16(
            static_cast<const __half*>(inputs[0]),         // item_embeddings
            static_cast<const __half*>(inputs[1]),         // user_embedding
            static_cast<const __half*>(inputs[2]),         // time_weights
            static_cast<const __half*>(inputs[3]),         // interaction_mask
            static_cast<__half*>(outputs[0]),              // user_item_scores
            static_cast<__half*>(outputs[1]),              // item_cooccur
            static_cast<__half*>(outputs[2]),              // short_term
            static_cast<__half*>(outputs[3]),              // long_term
            static_cast<__half*>(outputs[4]),              // cf_scores
            static_cast<__half*>(outputs[5]),              // neighbor_weights
            workspace,
            params.batch_size,
            params.seq_len,
            params.hidden_dim,
            params.short_window,
            params.long_window,
            params.decay_factor,
            params.top_k,
            params.min_cooccur,
            stream
        );
    } else {
        logError("不支持的数据类型");
        return -1;
    }

    return 0;
}

nvinfer1::IPluginV2DynamicExt* SequenceRecommendationInteractionPlugin::clone() const noexcept {
    return new SequenceRecommendationInteractionPlugin(
        seq_len_, hidden_dim_, short_window_, long_window_,
        decay_factor_, top_k_, min_cooccur_
    );
}

size_t SequenceRecommendationInteractionPlugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) * 5 + sizeof(float) * 2; // 5个int32_t + 2个float
}

void SequenceRecommendationInteractionPlugin::serialize(void* buffer) const noexcept {
    char* data = static_cast<char*>(buffer);
    size_t offset = 0;

    std::memcpy(data + offset, &seq_len_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(data + offset, &hidden_dim_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(data + offset, &short_window_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(data + offset, &long_window_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(data + offset, &top_k_, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(data + offset, &decay_factor_, sizeof(float));
    offset += sizeof(float);

    std::memcpy(data + offset, &min_cooccur_, sizeof(float));
}

void SequenceRecommendationInteractionPlugin::deserialize(const void* serialData, size_t serialLength) {
    const char* data = static_cast<const char*>(serialData);
    size_t offset = 0;

    std::memcpy(&seq_len_, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&hidden_dim_, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&short_window_, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&long_window_, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&top_k_, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&decay_factor_, data + offset, sizeof(float));
    offset += sizeof(float);

    std::memcpy(&min_cooccur_, data + offset, sizeof(float));
}

size_t SequenceRecommendationInteractionPlugin::calculateWorkspaceSize(
    int32_t batch_size, int32_t seq_len, int32_t hidden_dim
) const {
    // 工作空间大小计算 - 主要用于临时数组和共享内存溢出
    size_t workspace_size = 0;

    // 临时缓冲区大小（如果需要）
    size_t temp_buffer_size = batch_size * seq_len * hidden_dim * sizeof(float);
    workspace_size += temp_buffer_size;

    // 对齐到32字节边界
    workspace_size = (workspace_size + 31) & ~31;

    return workspace_size;
}

// SequenceRecommendationInteractionPluginCreator 实现
SequenceRecommendationInteractionPluginCreator::SequenceRecommendationInteractionPluginCreator()
    : GRPluginCreatorBase(PLUGIN_NAME, PLUGIN_VERSION) {

    // 定义插件属性
    plugin_attributes_.emplace_back("seq_len", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_attributes_.emplace_back("hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_attributes_.emplace_back("short_window", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_attributes_.emplace_back("long_window", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_attributes_.emplace_back("decay_factor", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    plugin_attributes_.emplace_back("top_k", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_attributes_.emplace_back("min_cooccur", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);

    field_collection_.nbFields = plugin_attributes_.size();
    field_collection_.fields = plugin_attributes_.data();
}

nvinfer1::IPluginV2* SequenceRecommendationInteractionPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc
) noexcept {
    int32_t seq_len = 32;
    int32_t hidden_dim = 128;
    int32_t short_window = 8;
    int32_t long_window = 32;
    float decay_factor = 0.1f;
    int32_t top_k = 8;
    float min_cooccur = 0.1f;

    // 解析插件属性
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        const char* attr_name = fc->fields[i].name;

        if (std::strcmp(attr_name, "seq_len") == 0) {
            seq_len = *static_cast<const int32_t*>(fc->fields[i].data);
        } else if (std::strcmp(attr_name, "hidden_dim") == 0) {
            hidden_dim = *static_cast<const int32_t*>(fc->fields[i].data);
        } else if (std::strcmp(attr_name, "short_window") == 0) {
            short_window = *static_cast<const int32_t*>(fc->fields[i].data);
        } else if (std::strcmp(attr_name, "long_window") == 0) {
            long_window = *static_cast<const int32_t*>(fc->fields[i].data);
        } else if (std::strcmp(attr_name, "decay_factor") == 0) {
            decay_factor = *static_cast<const float*>(fc->fields[i].data);
        } else if (std::strcmp(attr_name, "top_k") == 0) {
            top_k = *static_cast<const int32_t*>(fc->fields[i].data);
        } else if (std::strcmp(attr_name, "min_cooccur") == 0) {
            min_cooccur = *static_cast<const float*>(fc->fields[i].data);
        }
    }

    return new SequenceRecommendationInteractionPlugin(
        seq_len, hidden_dim, short_window, long_window,
        decay_factor, top_k, min_cooccur
    );
}

nvinfer1::IPluginV2* SequenceRecommendationInteractionPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength
) noexcept {
    return new SequenceRecommendationInteractionPlugin(serialData, serialLength);
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra