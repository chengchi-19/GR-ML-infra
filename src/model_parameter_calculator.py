#!/usr/bin/env python3
"""
模型参数计算器
计算生成式推荐模型的参数数量和内存占用
"""

import torch
import torch.nn as nn
from src.export_onnx import GenerativeRecommendationModel

def calculate_model_parameters(model):
    """计算模型参数数量"""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return total_params, trainable_params

def calculate_memory_usage(model, precision='fp16'):
    """计算模型内存占用"""
    total_params, _ = calculate_model_parameters(model)
    
    # 不同精度的字节数
    precision_bytes = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1
    }
    
    memory_bytes = total_params * precision_bytes.get(precision, 2)
    memory_mb = memory_bytes / (1024 * 1024)
    
    return memory_mb, memory_bytes

def analyze_model_components(model):
    """分析模型各组件参数分布"""
    component_params = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                component_params[name] = param_count
    
    return component_params

def main():
    """主函数"""
    print("=" * 60)
    print("生成式推荐模型参数分析")
    print("=" * 60)
    
    # 创建不同配置的模型
    configs = {
        "企业级配置": {
            "vocab_size": 10000,
            "embedding_dim": 512,
            "hidden_dim": 1024,
            "num_features": 1024,
            "num_layers": 6,
            "max_seq_len": 2048
        },
        "高性能配置": {
            "vocab_size": 20000,
            "embedding_dim": 768,
            "hidden_dim": 1536,
            "num_features": 1536,
            "num_layers": 8,
            "max_seq_len": 3072
        },
        "轻量级配置": {
            "vocab_size": 5000,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_features": 512,
            "num_layers": 4,
            "max_seq_len": 1024
        }
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name}:")
        print("-" * 40)
        
        # 创建模型
        model = GenerativeRecommendationModel(**config)
        
        # 计算参数
        total_params, trainable_params = calculate_model_parameters(model)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 计算不同精度的内存占用
        for precision in ['fp32', 'fp16', 'int8']:
            memory_mb, memory_bytes = calculate_memory_usage(model, precision)
            print(f"{precision.upper()} 内存占用: {memory_mb:.1f}MB ({memory_bytes:,} bytes)")
        
        # 分析组件参数分布
        component_params = analyze_model_components(model)
        print("\n组件参数分布:")
        for component, params in sorted(component_params.items(), key=lambda x: x[1], reverse=True):
            percentage = (params / total_params) * 100
            print(f"  {component}: {params:,} ({percentage:.1f}%)")
    
    # 详细分析默认配置
    print(f"\n{'='*60}")
    print("默认配置详细分析")
    print("=" * 60)
    
    default_model = GenerativeRecommendationModel()
    total_params, _ = calculate_model_parameters(default_model)
    
    print(f"默认配置总参数量: {total_params:,}")
    
    # 计算理论参数量
    vocab_size = 10000
    embedding_dim = 512
    hidden_dim = 1024
    num_features = 1024
    num_layers = 6
    
    # 理论计算
    token_embedding_params = vocab_size * embedding_dim
    position_embedding_params = 512 * embedding_dim  # max_seq_len
    feature_projection_params = num_features * embedding_dim
    transformer_params = num_layers * (
        4 * embedding_dim * embedding_dim +  # 4个线性层 (Q, K, V, O)
        2 * embedding_dim * hidden_dim +     # 2个FFN层
        4 * embedding_dim +                  # 4个LayerNorm
        2 * hidden_dim                       # 2个FFN偏置
    )
    output_projection_params = embedding_dim * vocab_size
    feature_output_params = embedding_dim * 1
    
    theoretical_total = (
        token_embedding_params +
        position_embedding_params +
        feature_projection_params +
        transformer_params +
        output_projection_params +
        feature_output_params
    )
    
    print(f"理论参数量: {theoretical_total:,}")
    print(f"实际参数量: {total_params:,}")
    print(f"差异: {abs(theoretical_total - total_params):,}")
    
    # 内存占用分析
    print(f"\n内存占用分析:")
    for precision in ['fp32', 'fp16', 'int8']:
        memory_mb, _ = calculate_memory_usage(default_model, precision)
        print(f"  {precision.upper()}: {memory_mb:.1f}MB")
    
    # 外部输入参数分析
    print(f"\n外部输入参数分析:")
    print(f"  dense_features: [batch_size, {num_features}] - {num_features}维密集特征")
    print(f"  input_ids: [batch_size, seq_len] - 序列ID")
    print(f"  attention_mask: [batch_size, seq_len] - 注意力掩码")
    print(f"  past_key_value_states: [batch_size, past_len, {embedding_dim}] - KV缓存")
    
    # 计算输入数据大小 - 扩展版本
    batch_size = 4  # 增加批次大小
    seq_len = 2000  # 增加序列长度
    past_len = 1000  # 增加KV缓存长度
    user_profile_dim = 256
    video_features_dim = 512
    
    dense_features_size = batch_size * num_features * 4  # FP32
    input_ids_size = batch_size * seq_len * 4  # INT32
    attention_mask_size = batch_size * seq_len * 4  # INT32
    past_kv_size = batch_size * past_len * embedding_dim * 4  # FP32
    user_profile_size = batch_size * user_profile_dim * 4  # FP32
    video_features_size = batch_size * video_features_dim * 4  # FP32
    
    total_input_size = dense_features_size + input_ids_size + attention_mask_size + past_kv_size + user_profile_size + video_features_size
    total_input_mb = total_input_size / (1024 * 1024)
    
    print(f"\n输入数据大小 (batch_size=4, seq_len=2000, past_len=1000):")
    print(f"  dense_features: {dense_features_size:,} bytes ({dense_features_size/1024/1024:.1f}MB)")
    print(f"  input_ids: {input_ids_size:,} bytes ({input_ids_size/1024/1024:.1f}MB)")
    print(f"  attention_mask: {attention_mask_size:,} bytes ({attention_mask_size/1024/1024:.1f}MB)")
    print(f"  past_key_value_states: {past_kv_size:,} bytes ({past_kv_size/1024/1024:.1f}MB)")
    print(f"  user_profile: {user_profile_size:,} bytes ({user_profile_size/1024/1024:.1f}MB)")
    print(f"  video_features: {video_features_size:,} bytes ({video_features_size/1024/1024:.1f}MB)")
    print(f"  总输入大小: {total_input_size:,} bytes ({total_input_mb:.1f}MB)")
    
    # 计算不同批次大小的输入大小
    print(f"\n不同批次大小的输入数据大小:")
    for bs in [1, 4, 8, 16, 32]:
        bs_dense_features_size = bs * num_features * 4
        bs_input_ids_size = bs * seq_len * 4
        bs_attention_mask_size = bs * seq_len * 4
        bs_past_kv_size = bs * past_len * embedding_dim * 4
        bs_user_profile_size = bs * user_profile_dim * 4
        bs_video_features_size = bs * video_features_dim * 4
        bs_total_size = bs_dense_features_size + bs_input_ids_size + bs_attention_mask_size + bs_past_kv_size + bs_user_profile_size + bs_video_features_size
        bs_total_mb = bs_total_size / (1024 * 1024)
        print(f"  batch_size={bs}: {bs_total_size:,} bytes ({bs_total_mb:.1f}MB)")

if __name__ == "__main__":
    main()
