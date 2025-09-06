#!/usr/bin/env python3
"""
Triton算子统一管理器

集成所有Triton自定义算子，提供统一的接口和策略选择
"""

import torch
import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入所有Triton算子
try:
    from .fused_attention_layernorm import FusedMultiHeadAttentionLayerNorm, create_fused_attention_layer
    from .hierarchical_sequence_fusion import HierarchicalSequenceFusion, create_hierarchical_fusion
    from .hstu_hierarchical_attention import HSTUHierarchicalAttention, create_hstu_attention
    from .sequence_recommendation_interaction import SequenceRecommendationInteraction, create_sequence_interaction
    from .interaction_wrapper import InteractionOperator
    
    TRITON_OPS_AVAILABLE = True
    logger.info("✅ 所有Triton算子导入成功")
except ImportError as e:
    TRITON_OPS_AVAILABLE = False
    logger.warning(f"⚠️ Triton算子导入失败: {e}")


class TritonOperatorManager:
    """
    Triton算子统一管理器
    
    负责管理所有Triton自定义算子的初始化和使用
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.operators = {}
        self.availability = {}
        
        # 初始化所有算子
        self._initialize_all_operators()
    
    def _initialize_all_operators(self):
        """初始化所有Triton算子"""
        
        if not TRITON_OPS_AVAILABLE:
            logger.warning("Triton算子不可用，跳过初始化")
            return
        
        # 1. 融合注意力+LayerNorm算子
        self._init_fused_attention_layernorm()
        
        # 2. 分层序列融合算子
        self._init_hierarchical_sequence_fusion()
        
        # 3. HSTU分层注意力算子
        self._init_hstu_hierarchical_attention()
        
        # 4. 序列推荐交互算子
        self._init_sequence_recommendation_interaction()
        
        # 5. 交互算子（已存在）
        self._init_interaction_operator()
        
        logger.info(f"Triton算子初始化完成: {sum(self.availability.values())}/{len(self.availability)}")
    
    def _init_fused_attention_layernorm(self):
        """初始化融合注意力+LayerNorm算子"""
        try:
            hstu_config = self.config.get('hstu', {})
            hidden_dim = hstu_config.get('d_model', 1024)
            num_heads = hstu_config.get('num_heads', 16)
            dropout_prob = hstu_config.get('dropout', 0.1)
            
            self.operators['fused_attention_layernorm'] = create_fused_attention_layer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                use_triton=True
            )
            self.availability['fused_attention_layernorm'] = True
            logger.info("✅ 融合注意力+LayerNorm算子初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 融合注意力+LayerNorm算子初始化失败: {e}")
            self.availability['fused_attention_layernorm'] = False
    
    def _init_hierarchical_sequence_fusion(self):
        """初始化分层序列融合算子"""
        try:
            fusion_config = self.config.get('hierarchical_fusion', {})
            
            self.operators['hierarchical_sequence_fusion'] = create_hierarchical_fusion(
                config=fusion_config
            )
            self.availability['hierarchical_sequence_fusion'] = True
            logger.info("✅ 分层序列融合算子初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 分层序列融合算子初始化失败: {e}")
            self.availability['hierarchical_sequence_fusion'] = False
    
    def _init_hstu_hierarchical_attention(self):
        """初始化HSTU分层注意力算子"""
        try:
            hstu_config = self.config.get('hstu', {})
            
            self.operators['hstu_hierarchical_attention'] = create_hstu_attention(
                config=hstu_config
            )
            self.availability['hstu_hierarchical_attention'] = True
            logger.info("✅ HSTU分层注意力算子初始化成功")
            
        except Exception as e:
            logger.error(f"❌ HSTU分层注意力算子初始化失败: {e}")
            self.availability['hstu_hierarchical_attention'] = False
    
    def _init_sequence_recommendation_interaction(self):
        """初始化序列推荐交互算子"""
        try:
            interaction_config = self.config.get('sequence_interaction', {})
            
            self.operators['sequence_recommendation_interaction'] = create_sequence_interaction(
                config=interaction_config
            )
            self.availability['sequence_recommendation_interaction'] = True
            logger.info("✅ 序列推荐交互算子初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 序列推荐交互算子初始化失败: {e}")
            self.availability['sequence_recommendation_interaction'] = False
    
    def _init_interaction_operator(self):
        """初始化交互算子（已存在）"""
        try:
            ops_config = self.config.get('custom_operators', {})
            
            self.operators['interaction_operator'] = InteractionOperator(
                cache_size=ops_config.get('cache_size', 1000),
                enable_benchmarking=ops_config.get('enable_benchmarking', True),
            )
            self.availability['interaction_operator'] = True
            logger.info("✅ 交互算子初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 交互算子初始化失败: {e}")
            self.availability['interaction_operator'] = False
    
    def apply_fused_attention_layernorm(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """应用融合注意力+LayerNorm算子"""
        
        if not self.availability.get('fused_attention_layernorm', False):
            logger.warning("融合注意力+LayerNorm算子不可用")
            return hidden_states
        
        try:
            operator = self.operators['fused_attention_layernorm']
            return operator(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights
            )
        except Exception as e:
            logger.error(f"融合注意力+LayerNorm算子执行失败: {e}")
            return hidden_states
    
    def apply_hierarchical_sequence_fusion(
        self,
        sequence_features: torch.Tensor,
        fusion_levels: List[int] = None
    ) -> torch.Tensor:
        """应用分层序列融合算子"""
        
        if not self.availability.get('hierarchical_sequence_fusion', False):
            logger.warning("分层序列融合算子不可用")
            return sequence_features
        
        try:
            operator = self.operators['hierarchical_sequence_fusion']
            return operator(
                sequence_features=sequence_features,
                fusion_levels=fusion_levels
            )
        except Exception as e:
            logger.error(f"分层序列融合算子执行失败: {e}")
            return sequence_features
    
    def apply_hstu_hierarchical_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hierarchical_levels: List[int] = None
    ) -> torch.Tensor:
        """应用HSTU分层注意力算子"""
        
        if not self.availability.get('hstu_hierarchical_attention', False):
            logger.warning("HSTU分层注意力算子不可用")
            return query
        
        try:
            operator = self.operators['hstu_hierarchical_attention']
            return operator(
                query=query,
                key=key,
                value=value,
                hierarchical_levels=hierarchical_levels
            )
        except Exception as e:
            logger.error(f"HSTU分层注意力算子执行失败: {e}")
            return query
    
    def apply_sequence_recommendation_interaction(
        self,
        user_sequences: List[List[Dict[str, Any]]],
        candidate_items: List[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """应用序列推荐交互算子"""
        
        if not self.availability.get('sequence_recommendation_interaction', False):
            logger.warning("序列推荐交互算子不可用")
            return torch.randn(len(user_sequences), 1024)
        
        try:
            operator = self.operators['sequence_recommendation_interaction']
            return operator(
                user_sequences=user_sequences,
                candidate_items=candidate_items
            )
        except Exception as e:
            logger.error(f"序列推荐交互算子执行失败: {e}")
            return torch.randn(len(user_sequences), 1024)
    
    def apply_interaction_operator(
        self,
        features: torch.Tensor,
        mode: str = 'advanced',
        return_stats: bool = False
    ) -> torch.Tensor:
        """应用交互算子（已存在）"""
        
        if not self.availability.get('interaction_operator', False):
            logger.warning("交互算子不可用")
            return features
        
        try:
            operator = self.operators['interaction_operator']
            return operator(
                features=features,
                mode=mode,
                return_stats=return_stats
            )
        except Exception as e:
            logger.error(f"交互算子执行失败: {e}")
            return features
    
    def get_operator_availability(self) -> Dict[str, bool]:
        """获取算子可用性"""
        return self.availability.copy()
    
    def get_operator_stats(self) -> Dict[str, Any]:
        """获取算子统计信息"""
        stats = {
            'availability': self.availability,
            'total_operators': len(self.availability),
            'available_operators': sum(self.availability.values()),
        }
        
        # 添加各算子的详细统计
        for name, operator in self.operators.items():
            if hasattr(operator, 'get_performance_stats'):
                stats[name] = operator.get_performance_stats()
            else:
                stats[name] = {'status': 'initialized'}
        
        return stats
    
    def benchmark_all_operators(
        self,
        test_data: Dict[str, Any],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """对所有算子进行基准测试"""
        
        results = {'benchmark_results': {}}
        
        for name, available in self.availability.items():
            if not available:
                continue
            
            logger.info(f"开始基准测试算子: {name}")
            
            try:
                operator = self.operators[name]
                times = []
                
                for i in range(num_iterations):
                    start_time = time.time()
                    
                    # 根据算子类型执行相应操作
                    if name == 'fused_attention_layernorm':
                        output = operator(test_data['hidden_states'])
                    elif name == 'hierarchical_sequence_fusion':
                        output = operator(test_data['sequence_features'])
                    elif name == 'hstu_hierarchical_attention':
                        output = operator(
                            test_data['query'],
                            test_data['key'],
                            test_data['value']
                        )
                    elif name == 'sequence_recommendation_interaction':
                        output = operator(test_data['user_sequences'])
                    elif name == 'interaction_operator':
                        output = operator(test_data['features'])
                    
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                if times:
                    results['benchmark_results'][name] = {
                        'avg_latency_ms': np.mean(times),
                        'min_latency_ms': np.min(times),
                        'max_latency_ms': np.max(times),
                        'std_latency_ms': np.std(times),
                        'successful_runs': len(times),
                        'total_runs': num_iterations,
                    }
                    
            except Exception as e:
                logger.error(f"算子 {name} 基准测试失败: {e}")
                results['benchmark_results'][name] = {'error': str(e)}
        
        return results


def create_triton_operator_manager(config: Dict[str, Any]) -> TritonOperatorManager:
    """创建Triton算子管理器"""
    
    return TritonOperatorManager(config)


if __name__ == "__main__":
    # 测试配置
    test_config = {
        'hstu': {
            'd_model': 1024,
            'num_heads': 16,
            'dropout': 0.1,
        },
        'custom_operators': {
            'cache_size': 1000,
            'enable_benchmarking': True,
        }
    }
    
    # 创建管理器
    manager = create_triton_operator_manager(test_config)
    
    # 获取可用性
    availability = manager.get_operator_availability()
    print("Triton算子可用性:")
    for name, available in availability.items():
        print(f"  {name}: {'✅' if available else '❌'}")
    
    # 获取统计信息
    stats = manager.get_operator_stats()
    print(f"\n统计信息: {stats}")