#include "tensorrt_plugin_base.h"
#include "fused_attention_layernorm_plugin.h"
#include "hierarchical_sequence_fusion_plugin.h"
#include "sequence_recommendation_interaction_plugin.h"
#include "interaction_triton_fast_plugin.h"
#include "hstu_hierarchical_attention_plugin.h"
#include <memory>

namespace gr_ml_infra {
namespace tensorrt_plugins {

// 插件注册初始化函数
void initializeGRMLInfraPlugins() {
    auto& registry = PluginRegistry::getInstance();

    // 注册FusedAttentionLayerNorm插件
    registry.registerPlugin(std::make_unique<FusedAttentionLayerNormPluginCreator>());

    // 注册分层序列融合插件
    registry.registerPlugin(std::make_unique<HierarchicalSequenceFusionPluginCreator>());

    // 注册SequenceRecommendationInteraction插件
    registry.registerPlugin(std::make_unique<SequenceRecommendationInteractionPluginCreator>());

    // 注册InteractionTritonFast插件
    registry.registerPlugin(std::make_unique<InteractionTritonFastPluginCreator>());

    // 注册HSTUHierarchicalAttention插件
    registry.registerPlugin(std::make_unique<HSTUHierarchicalAttentionPluginCreator>());

    std::cout << "[GR-ML-Infra] 所有TensorRT插件注册完成" << std::endl;
}

// 获取所有已注册插件的列表
std::vector<std::string> getRegisteredGRMLInfraPlugins() {
    return PluginRegistry::getInstance().getRegisteredPlugins();
}

} // namespace tensorrt_plugins
} // namespace gr_ml_infra

// C接口，供Python调用
extern "C" {
    void initialize_gr_ml_infra_plugins() {
        gr_ml_infra::tensorrt_plugins::initializeGRMLInfraPlugins();
    }

    int get_num_registered_plugins() {
        auto plugins = gr_ml_infra::tensorrt_plugins::getRegisteredGRMLInfraPlugins();
        return static_cast<int>(plugins.size());
    }
}