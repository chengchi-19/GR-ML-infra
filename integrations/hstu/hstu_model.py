#!/usr/bin/env python3
"""
Meta HSTU (Hierarchical Sequential Transduction Unit) 模型集成

集成Meta开源的HSTU模型，支持生成式推荐和多任务学习。
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn

# 添加Meta HSTU模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
hstu_path = os.path.join(project_root, "external", "meta-hstu")
sys.path.append(hstu_path)

logger = logging.getLogger(__name__)

try:
    # 导入Meta HSTU核心模块
    from generative_recommenders.research.modeling.sequential.hstu import (
        HSTUModule,
        HSTUModuleV2,
        HSTUConfig,
    )
    from generative_recommenders.research.modeling.sequential.embedding_modules import (
        EmbeddingModule,
    )
    from generative_recommenders.research.modeling.sequential.features import (
        SequentialFeatures,
    )
    from generative_recommenders.research.modeling.sequential.utils import (
        get_current_embeddings,
    )
    
    HSTU_AVAILABLE = True
    logger.info("✅ Meta HSTU模块导入成功")
    
except ImportError as e:
    HSTU_AVAILABLE = False
    logger.warning(f"⚠️ Meta HSTU模块导入失败: {e}")


class HSTUModelConfig:
    """HSTU模型配置"""
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        use_bias: bool = True,
        activation_fn: str = "gelu",
        # HSTU特定参数
        hstu_expansion_factor: int = 4,
        hstu_gate_type: str = "sigmoid",
        enable_hierarchical_attention: bool = True,
        similarity_dim: int = 256,
        temperature: float = 0.1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers  
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.activation_fn = activation_fn
        self.hstu_expansion_factor = hstu_expansion_factor
        self.hstu_gate_type = hstu_gate_type
        self.enable_hierarchical_attention = enable_hierarchical_attention
        self.similarity_dim = similarity_dim
        self.temperature = temperature
        
        # 更新其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)


class HSTUGenerativeRecommender(nn.Module):
    """
    基于Meta HSTU的生成式推荐模型
    
    集成了HSTU架构和我们的自定义优化功能
    """
    
    def __init__(self, config: HSTUModelConfig):
        super().__init__()
        self.config = config
        
        if not HSTU_AVAILABLE:
            logger.warning("Meta HSTU不可用，使用简化实现")
            self._init_fallback_model()
        else:
            self._init_hstu_model()
    
    def _init_hstu_model(self):
        """初始化真正的Meta HSTU模型"""
        try:
            # 创建HSTU配置
            hstu_config = HSTUConfig(
                embedding_dim=self.config.d_model,
                num_layers=self.config.num_layers,
                num_attention_heads=self.config.num_heads,
                intermediate_size=self.config.d_ff,
                max_position_embeddings=self.config.max_seq_len,
                hidden_dropout_prob=self.config.dropout,
                attention_probs_dropout_prob=self.config.dropout,
                layer_norm_eps=self.config.layer_norm_eps,
                pad_token_id=self.config.pad_token_id,
                vocab_size=self.config.vocab_size,
                use_bias=self.config.use_bias,
                activation_function=self.config.activation_fn,
                similarity_dim=self.config.similarity_dim,
                temperature=self.config.temperature,
            )
            
            # 初始化嵌入模块
            self.embedding_module = EmbeddingModule(
                embedding_dim=self.config.d_model,
                vocab_size=self.config.vocab_size,
                pad_token_id=self.config.pad_token_id,
            )
            
            # 初始化HSTU核心模块 
            if self.config.enable_hierarchical_attention:
                self.hstu_encoder = HSTUModuleV2(hstu_config)
            else:
                self.hstu_encoder = HSTUModule(hstu_config)
            
            # 输出层
            self.output_projection = nn.Linear(self.config.d_model, self.config.vocab_size)
            
            # 多任务输出头
            self.engagement_head = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.d_model // 2, 1),
                nn.Sigmoid()
            )
            
            self.retention_head = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model // 2), 
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.d_model // 2, 1),
                nn.Sigmoid()
            )
            
            self.monetization_head = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model // 2),
                nn.GELU(), 
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.d_model // 2, 1),
                nn.Sigmoid()
            )
            
            self.hstu_available = True
            logger.info("✅ Meta HSTU模型初始化成功")
            
        except Exception as e:
            logger.error(f"❌ Meta HSTU模型初始化失败: {e}")
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """初始化简化回退模型"""
        self.hstu_available = False
        
        # 简化的Transformer实现
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, 
            self.config.d_model,
            padding_idx=self.config.pad_token_id
        )
        
        self.position_embedding = nn.Embedding(
            self.config.max_seq_len,
            self.config.d_model
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation=self.config.activation_fn,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config.num_layers
        )
        
        self.output_projection = nn.Linear(self.config.d_model, self.config.vocab_size)
        
        # 多任务输出头
        self.engagement_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.retention_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(), 
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.monetization_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout), 
            nn.Linear(self.config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info("✅ 回退模型初始化成功")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        dense_features: Optional[torch.Tensor] = None,
        user_profile: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.hstu_available:
            return self._forward_hstu(
                input_ids, attention_mask, position_ids, dense_features,
                user_profile, item_features, past_key_values, return_dict, **kwargs
            )
        else:
            return self._forward_fallback(
                input_ids, attention_mask, position_ids, dense_features,
                user_profile, item_features, past_key_values, return_dict, **kwargs
            )
    
    def _forward_hstu(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        dense_features: Optional[torch.Tensor] = None,
        user_profile: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None, 
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """使用Meta HSTU的前向传播，适配优化的特征处理"""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 处理时间戳 (来自kwargs)
        timestamps = kwargs.get('timestamps', torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0))
        
        # 创建适配的SequentialFeatures对象
        features = SequentialFeatures(
            user_id=input_ids[:, 0].unsqueeze(1),  # 使用第一个token作为user_id
            item_id=input_ids,
            timestamps=timestamps.long(),  # HSTU期望长整型时间戳
            weights=attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.float),
        )
        
        # 获取嵌入
        embeddings = self.embedding_module(features)
        
        # 如果有密集特征，融合到嵌入中
        if dense_features is not None:
            # 通过线性层将密集特征投影到嵌入维度
            if not hasattr(self, 'dense_feature_projector'):
                self.dense_feature_projector = nn.Linear(
                    dense_features.shape[-1], 
                    self.config.d_model,
                    device=device
                )
            
            dense_embeddings = self.dense_feature_projector(dense_features)
            # 加权融合原始嵌入和密集特征嵌入
            embeddings = embeddings + 0.1 * dense_embeddings  # 0.1为融合权重
        
        # 如果有用户画像特征，进一步增强嵌入
        if user_profile is not None:
            if not hasattr(self, 'user_profile_projector'):
                self.user_profile_projector = nn.Linear(
                    user_profile.shape[-1],
                    self.config.d_model,
                    device=device
                )
            
            user_embeddings = self.user_profile_projector(user_profile)
            embeddings = embeddings + 0.05 * user_embeddings  # 更小的权重避免覆盖主特征
        
        # HSTU编码 - 使用增强的特征
        encoded_output = self.hstu_encoder(
            embeddings=embeddings,
            features=features,
            attention_mask=attention_mask,
        )
        
        # 输出投影
        logits = self.output_projection(encoded_output)
        
        # 多任务预测 - 使用更智能的池化策略
        # 使用注意力权重进行加权平均池化
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded_output)
            masked_output = encoded_output * mask_expanded.float()
            pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        else:
            pooled_output = encoded_output.mean(dim=1)  # 简单平均池化
        
        # 多任务预测头
        engagement_scores = self.engagement_head(pooled_output)
        retention_scores = self.retention_head(pooled_output)
        monetization_scores = self.monetization_head(pooled_output)
        
        results = {
            'logits': logits,
            'hidden_states': encoded_output,
            'engagement_scores': engagement_scores,
            'retention_scores': retention_scores, 
            'monetization_scores': monetization_scores,
            'pooled_output': pooled_output,  # 添加池化输出供后续使用
        }
        
        return results
    
    def _forward_fallback(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        dense_features: Optional[torch.Tensor] = None,
        user_profile: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None, 
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """使用简化Transformer的前向传播"""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token嵌入
        token_embeddings = self.token_embedding(input_ids)
        
        # 位置嵌入
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        
        # 组合嵌入
        embeddings = token_embeddings + position_embeddings
        
        # 注意力掩码处理
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer编码
        encoded_output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # 输出投影
        logits = self.output_projection(encoded_output)
        
        # 多任务预测
        pooled_output = encoded_output.mean(dim=1)  # 简单平均池化
        engagement_scores = self.engagement_head(pooled_output)
        retention_scores = self.retention_head(pooled_output)
        monetization_scores = self.monetization_head(pooled_output)
        
        results = {
            'logits': logits,
            'hidden_states': encoded_output,
            'engagement_scores': engagement_scores,
            'retention_scores': retention_scores,
            'monetization_scores': monetization_scores,
        }
        
        return results
    
    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        dense_features: Optional[torch.Tensor] = None,
        user_profile: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """兼容现有代码的前向传播接口"""
        
        results = self.forward(
            input_ids=input_ids,
            dense_features=dense_features,
            user_profile=user_profile,
            item_features=item_features,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 返回元组格式以保持兼容性
        return (
            results['logits'],
            results['hidden_states'],
            results['engagement_scores'],
            results['retention_scores'], 
            results['monetization_scores']
        )
    
    def generate_recommendations(
        self,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """生成推荐结果"""
        
        # 构建输入序列
        input_ids = self._build_input_sequence(user_behaviors)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :]  # 取最后一个时间步
            
            # 应用temperature
            logits = logits / temperature
            
            # Top-k和top-p采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits_filtered = torch.full_like(logits, -float('inf'))
                logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # 采样
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, num_recommendations)
            sampled_probs = probs.gather(-1, sampled_indices)
        
        # 构建推荐结果
        recommendations = []
        for i in range(num_recommendations):
            item_id = sampled_indices[0, i].item()
            score = sampled_probs[0, i].item()
            
            recommendations.append({
                'video_id': f'item_{item_id}',
                'score': float(score),
                'position': i + 1,
                'reason': 'HSTU生成式推荐'
            })
        
        return recommendations
    
    def _build_input_sequence(self, user_behaviors: List[Dict[str, Any]]) -> torch.Tensor:
        """从用户行为构建输入序列"""
        
        # 简化实现：使用行为的哈希值作为token ID
        sequence = []
        for behavior in user_behaviors:
            video_id = behavior.get('video_id', 'unknown')
            # 将视频ID转换为vocab范围内的ID
            token_id = abs(hash(video_id)) % self.config.vocab_size
            sequence.append(token_id)
        
        # 填充或截断到固定长度
        max_len = min(len(sequence), self.config.max_seq_len)
        if len(sequence) < max_len:
            sequence.extend([self.config.pad_token_id] * (max_len - len(sequence)))
        else:
            sequence = sequence[:max_len]
        
        return torch.tensor([sequence], dtype=torch.long)


def create_hstu_model(config_dict: Dict[str, Any]) -> HSTUGenerativeRecommender:
    """创建HSTU模型实例"""
    config = HSTUModelConfig(**config_dict)
    model = HSTUGenerativeRecommender(config)
    
    logger.info(f"✅ HSTU模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    # 测试HSTU模型
    config = HSTUModelConfig(
        vocab_size=10000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=1024
    )
    
    model = HSTUGenerativeRecommender(config)
    
    # 测试前向传播
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model.forward(input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    print(f"Engagement scores shape: {outputs['engagement_scores'].shape}")
    
    # 测试推荐生成
    user_behaviors = [
        {'video_id': 'video_1', 'watch_duration': 120},
        {'video_id': 'video_2', 'watch_duration': 90},
        {'video_id': 'video_3', 'watch_duration': 200},
    ]
    
    recommendations = model.generate_recommendations(user_behaviors, num_recommendations=5)
    print(f"Generated {len(recommendations)} recommendations")
    for rec in recommendations:
        print(f"  {rec['video_id']}: {rec['score']:.4f}")