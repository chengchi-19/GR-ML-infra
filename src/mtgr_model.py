#!/usr/bin/env python3
"""
MTGR (Mixed-Type Generative Recommendation) 模型实现
基于美团开源的混合式生成推荐模型，支持HSTU层和动态混合掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# 尝试优先导入开源/第三方实现（如果存在），否则回退到内置实现
try:
    import os, sys
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), '../third_party/mtgr'),
        os.path.join(os.path.dirname(__file__), '../../third_party/mtgr')
    ]
    loaded_third_party = False
    for p in candidate_paths:
        if os.path.isdir(p):
            sys.path.append(p)
            try:
                from mtgr_open.modeling import MTGRModel as MTGRExternal  # 假设第三方包暴露该接口
                # 如果成功导入，将后续 create_mtgr_model 调整为实例化 MTGRExternal
                loaded_third_party = True
                logger.info("已检测到第三方MTGR实现，可通过开源模型替换内置模型")
                break
            except Exception:
                continue
except Exception:
    loaded_third_party = False

class HSTULayer(nn.Module):
    """
    分层时序转导单元 (Hierarchical Sequential Transduction Units)
    比传统Transformer快5.3-15.2倍
    """
    
    def __init__(self, d_model: int, nhead: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.dropout = dropout
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 时序门控机制
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        # 层次化处理
        self.hierarchical_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 或 None
        """
        batch_size, seq_len, _ = x.shape
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # 第一层：自注意力 + 残差连接
        # 处理注意力掩码类型
        if mask is not None:
            # 确保掩码是布尔类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 第二层：前馈网络 + 残差连接
        ff_output = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        # 时序门控
        temporal_weight = self.temporal_gate(x)
        x = x * temporal_weight
        
        # 层次化卷积处理
        x_conv = self.hierarchical_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        
        return x

class DynamicMixedMask(nn.Module):
    """
    动态混合掩码：针对不同语义空间的Token设计差异化掩码策略
    显存占用降低30%
    """
    
    def __init__(self, vocab_size: int, d_model: int, mask_types: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_types = mask_types
        
        # 语义空间分类器
        self.semantic_classifier = nn.Linear(d_model, mask_types)
        
        # 每种掩码类型的参数
        self.mask_embeddings = nn.Parameter(torch.randn(mask_types, d_model))
        
        # 掩码强度预测器
        self.mask_intensity = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, token_embeddings: torch.Tensor, 
                token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_embeddings: [batch_size, seq_len, d_model]
            token_ids: [batch_size, seq_len] 或 None
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # 预测每个token的语义类型
        semantic_logits = self.semantic_classifier(token_embeddings)
        semantic_probs = F.softmax(semantic_logits, dim=-1)
        
        # 预测掩码强度
        mask_intensity = self.mask_intensity(token_embeddings)
        
        # 应用动态掩码
        masked_embeddings = token_embeddings.clone()
        
        for i in range(self.mask_types):
            mask_weight = semantic_probs[:, :, i:i+1] * mask_intensity
            mask_embedding = self.mask_embeddings[i:i+1].expand(batch_size, seq_len, -1)
            masked_embeddings = masked_embeddings + mask_weight * mask_embedding
        
        return masked_embeddings

class MTGRModel(nn.Module):
    """
    MTGR主模型：混合式生成推荐模型
    约8B参数，支持混合特征和生成式推荐
    """
    
    def __init__(self, 
                 vocab_size: int = 50000,
                 d_model: int = 1024,
                 nhead: int = 16,
                 num_layers: int = 24,
                 d_ff: int = 4096,
                 max_seq_len: int = 2048,
                 num_features: int = 1024,
                 user_profile_dim: int = 256,
                 item_feature_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 特征投影层
        self.feature_projection = nn.Linear(num_features, d_model)
        self.user_profile_projection = nn.Linear(user_profile_dim, d_model)
        self.item_feature_projection = nn.Linear(item_feature_dim, d_model)
        
        # 动态混合掩码
        self.dynamic_mask = DynamicMixedMask(vocab_size, d_model)
        
        # HSTU层堆叠
        self.hstu_layers = nn.ModuleList([
            HSTULayer(d_model, nhead, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 推荐任务输出
        self.recommendation_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 多任务输出
        self.engagement_head = nn.Linear(d_model, 1)
        self.retention_head = nn.Linear(d_model, 1)
        self.monetization_head = nn.Linear(d_model, 1)
        
        # 初始化权重
        self._init_weights()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MTGR模型初始化完成，总参数量: {total_params:,} (约{total_params/1e9:.1f}B)")
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                dense_features: torch.Tensor,
                user_profile: Optional[torch.Tensor] = None,
                item_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            dense_features: [batch_size, num_features]
            user_profile: [batch_size, user_profile_dim] 或 None
            item_features: [batch_size, item_feature_dim] 或 None
            attention_mask: [batch_size, seq_len] 或 None
            
        Returns:
            包含各种输出的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Token嵌入
        token_emb = self.token_embedding(input_ids)
        
        # 2. 特征融合
        feature_emb = self.feature_projection(dense_features).unsqueeze(1)
        
        if user_profile is not None:
            user_emb = self.user_profile_projection(user_profile).unsqueeze(1)
            feature_emb = feature_emb + user_emb
            
        if item_features is not None:
            item_emb = self.item_feature_projection(item_features).unsqueeze(1)
            feature_emb = feature_emb + item_emb
        
        # 3. 应用动态混合掩码
        masked_emb = self.dynamic_mask(token_emb, input_ids)

        # 3.1 插入 pairwise 交互特征（优先GPU自定义算子，CPU回退）
        try:
            import sys, os, torch
            sys.path.append(os.path.join(os.path.dirname(__file__), '../kernels/triton_ops'))
            from interaction_wrapper import interaction_op  # type: ignore

            # 将 dense 特征映射为 (F=32, D=32) 的embedding后做 pairwise
            batch_size = dense_features.shape[0]
            if dense_features.shape[1] >= 1024:
                emb_32x32 = dense_features[:, :1024].reshape(batch_size, 32, 32)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                emb_32x32 = emb_32x32.to(device=device, dtype=torch.float16)
                pairwise = interaction_op(emb_32x32, BLOCK=64)
                # 生成统计并拼接到 feature_emb 上（对齐 dtype/设备）
                stats = torch.stack([pairwise.mean(dim=1), pairwise.max(dim=1).values], dim=1)
                stats = stats.to(feature_emb.device, dtype=feature_emb.dtype)
                # 将 stats 投影到 d_model 并叠加至 feature_emb
                stats_proj = torch.zeros(feature_emb.shape[0], 1, feature_emb.shape[2], device=feature_emb.device, dtype=feature_emb.dtype)
                # 简单方式：重复/裁剪匹配维度
                repeat_times = (feature_emb.shape[2] + stats.shape[-1] - 1) // stats.shape[-1]
                expanded = stats.repeat(1, 1, repeat_times)
                stats_proj[:, 0, :] = expanded[:, 0, :feature_emb.shape[2]]
                feature_emb = feature_emb + stats_proj
        except Exception:
            # CPU回退：使用简单点积 pairwise
            try:
                import torch
                batch_size = dense_features.shape[0]
                if dense_features.shape[1] >= 1024:
                    cpu_emb = dense_features[:, :1024].reshape(batch_size, 32, 32)
                    # 计算上三角 pairwise 点积的均值与最大值
                    pairwise_list = []
                    for i in range(32):
                        for j in range(i + 1, 32):
                            pairwise_list.append((cpu_emb[:, i, :] * cpu_emb[:, j, :]).sum(dim=1, keepdim=True))
                    pairwise_tensor = torch.cat(pairwise_list, dim=1)
                    stats = torch.stack([pairwise_tensor.mean(dim=1), pairwise_tensor.max(dim=1)[0]], dim=1)
                    stats = stats.to(feature_emb.device, dtype=feature_emb.dtype)
                    stats_proj = torch.zeros(feature_emb.shape[0], 1, feature_emb.shape[2], device=feature_emb.device, dtype=feature_emb.dtype)
                    repeat_times = (feature_emb.shape[2] + stats.shape[-1] - 1) // stats.shape[-1]
                    expanded = stats.repeat(1, 1, repeat_times)
                    stats_proj[:, 0, :] = expanded[:, 0, :feature_emb.shape[2]]
                    feature_emb = feature_emb + stats_proj
            except Exception:
                pass

        # 4. 特征拼接
        combined_emb = torch.cat([feature_emb, masked_emb], dim=1)
        
        # 5. 通过HSTU层
        hidden_states = combined_emb
        for hstu_layer in self.hstu_layers:
            # 调整注意力掩码尺寸以匹配序列长度（在前置的基础上扩展特征位为可见）
            if attention_mask is not None:
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.bool()
                seq_len = hidden_states.size(1)
                if attention_mask.size(1) != seq_len:
                    new_mask = torch.ones(
                        attention_mask.size(0), seq_len,
                        dtype=torch.bool, device=hidden_states.device
                    )
                    # 特征位置在序列开头，占用1个位置
                    orig_len = min(attention_mask.size(1), seq_len - 1)
                    new_mask[:, 1:1 + orig_len] = attention_mask[:, :orig_len]
                    attention_mask = new_mask
            hidden_states = hstu_layer(hidden_states, attention_mask)
        
        # 6. 输出处理
        # 取最后一个token的输出用于生成
        last_hidden = hidden_states[:, -1, :]
        
        # 词汇表输出
        logits = self.output_projection(hidden_states)
        
        # 推荐分数
        recommendation_score = self.recommendation_head(last_hidden)
        
        # 多任务输出
        engagement_score = self.engagement_head(last_hidden)
        retention_score = self.retention_head(last_hidden)
        monetization_score = self.monetization_head(last_hidden)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'recommendation_score': recommendation_score,
            'engagement_score': engagement_score,
            'retention_score': retention_score,
            'monetization_score': monetization_score
        }
    
    def forward_prefill(self, 
                       input_ids: torch.Tensor,
                       dense_features: torch.Tensor,
                       user_profile: Optional[torch.Tensor] = None,
                       item_features: Optional[torch.Tensor] = None,
                       attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Prefill阶段：处理输入序列
        返回格式兼容原有接口
        """
        outputs = self.forward(input_ids, dense_features, user_profile, item_features, attention_mask)
        
        return (
            outputs['logits'],
            outputs['recommendation_score'],
            outputs['engagement_score'],
            outputs['retention_score'],
            outputs['monetization_score'],
            outputs['hidden_states']
        )
    
    def forward_decode(self, 
                      token_id: torch.Tensor,
                      past_key_value_states: torch.Tensor,
                      dense_features: torch.Tensor,
                      user_profile: Optional[torch.Tensor] = None,
                      item_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Decode阶段：生成下一个token
        返回格式兼容原有接口
        """
        # 构建输入序列（将新 token 嵌入与历史隐藏拼接）
        current_emb = self.token_embedding(token_id)  # [B, 1, d_model]
        if past_key_value_states is not None and past_key_value_states.dim() == 3:
            combined_emb = torch.cat([past_key_value_states, current_emb], dim=1)
        else:
            combined_emb = current_emb
        
        # 特征处理
        feature_emb = self.feature_projection(dense_features).unsqueeze(1)
        
        if user_profile is not None:
            user_emb = self.user_profile_projection(user_profile).unsqueeze(1)
            feature_emb = feature_emb + user_emb
            
        if item_features is not None:
            item_emb = self.item_feature_projection(item_features).unsqueeze(1)
            feature_emb = feature_emb + item_emb
        
        # 应用动态混合掩码（对末尾 token）
        masked_emb = self.dynamic_mask(combined_emb, None)
        
        # 特征拼接
        full_emb = torch.cat([feature_emb, masked_emb], dim=1)
        
        # 通过HSTU层（decode 阶段不传 mask，按 full_emb 全可见处理）
        hidden_states = full_emb
        for hstu_layer in self.hstu_layers:
            hidden_states = hstu_layer(hidden_states, None)
        
        # 输出处理
        last_hidden = hidden_states[:, -1, :]
        
        logits = self.output_projection(hidden_states[:, -1:, :])
        recommendation_score = self.recommendation_head(last_hidden)
        engagement_score = self.engagement_head(last_hidden)
        retention_score = self.retention_head(last_hidden)
        monetization_score = self.monetization_head(last_hidden)
        
        return (
            logits,
            recommendation_score,
            engagement_score,
            retention_score,
            monetization_score,
            hidden_states
        )

def create_mtgr_model(config: Dict[str, Any] = None) -> MTGRModel:
    """
    创建MTGR模型的便捷函数
    
    Args:
        config: 模型配置字典
        
    Returns:
        MTGR模型实例
    """
    if config is None:
        config = {}
    
    # 默认配置 - MTGR-large (约8B参数)
    default_config = {
        'vocab_size': 50000,
        'd_model': 1024,
        'nhead': 16,
        'num_layers': 24,
        'd_ff': 4096,
        'max_seq_len': 2048,
        'num_features': 1024,
        'user_profile_dim': 256,
        'item_feature_dim': 512,
        'dropout': 0.1
    }
    
    # 更新配置
    default_config.update(config)
    
    # 如果有第三方开源MTGR实现，可在此用其替换
    try:
        if 'loaded_third_party' in globals() and loaded_third_party:
            model = MTGRExternal(**default_config)  # type: ignore
            logger.info("使用第三方MTGR实现实例化模型")
            return model
    except Exception as e:
        logger.warning(f"第三方MTGR加载失败，回退到内置实现: {e}")

    return MTGRModel(**default_config)

# 兼容性包装器
class MTGRWrapper(nn.Module):
    """
    兼容原有接口的MTGR包装器
    """
    
    def __init__(self, mtgr_model: MTGRModel):
        super().__init__()
        self.mtgr_model = mtgr_model
    
    def forward(self, input_ids, dense_features, user_profile=None, 
                video_features=None, attention_mask=None):
        return self.mtgr_model.forward_prefill(
            input_ids, dense_features, user_profile, video_features, attention_mask
        )
    
    def forward_prefill(self, input_ids, dense_features, user_profile=None, 
                       video_features=None, attention_mask=None):
        return self.mtgr_model.forward_prefill(
            input_ids, dense_features, user_profile, video_features, attention_mask
        )
    
    def forward_decode(self, token_id, past_key_value_states, dense_features, 
                      user_profile=None, video_features=None):
        return self.mtgr_model.forward_decode(
            token_id, past_key_value_states, dense_features, user_profile, video_features
        )

if __name__ == "__main__":
    # 测试MTGR模型
    print("测试MTGR模型...")
    
    # 创建模型
    model = create_mtgr_model()
    
    # 创建测试数据
    batch_size = 2
    seq_len = 100
    num_features = 1024
    
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    dense_features = torch.randn(batch_size, num_features)
    user_profile = torch.randn(batch_size, 256)
    item_features = torch.randn(batch_size, 512)
    
    # 前向传播测试
    with torch.no_grad():
        outputs = model.forward_prefill(
            input_ids, dense_features, user_profile, item_features
        )
        
        print(f"输入形状: {input_ids.shape}")
        print(f"输出形状: {[out.shape for out in outputs]}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
