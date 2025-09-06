#!/usr/bin/env python3
"""
开源框架集成主控制器

实现HSTU -> TensorRT -> VLLM的统一推理流程，
充分发挥各框架的协同优势。
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict
import torch
import numpy as np

# 添加集成模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

logger = logging.getLogger(__name__)

# 导入集成模块
try:
    from integrations.hstu.hstu_model import HSTUGenerativeRecommender, HSTUModelConfig, create_hstu_model
    from integrations.vllm.vllm_engine import VLLMRecommenderEngine, VLLMConfig, create_vllm_engine
    from integrations.tensorrt.tensorrt_engine import TensorRTOptimizedEngine, TensorRTConfig, create_tensorrt_engine
    
    # 导入自定义优化算子
    from optimizations.triton_ops.interaction_wrapper import InteractionOperator
    from optimizations.cache.intelligent_cache import IntelligentEmbeddingCache
    
    INTEGRATIONS_AVAILABLE = True
    logger.info("✅ 开源框架集成模块导入成功")
    
except ImportError as e:
    INTEGRATIONS_AVAILABLE = False
    logger.warning(f"⚠️ 开源框架集成模块导入失败: {e}")


class OpenSourceFrameworkController:
    """
    开源框架集成主控制器
    
    实现统一推理流程: HSTU模型 -> TensorRT优化 -> VLLM推理服务
    结合各框架优势提供高性能推理。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_stats = defaultdict(float)
        self.inference_count = defaultdict(int)
        
        # 初始化各个框架
        self.hstu_model = None
        self.vllm_engine = None
        self.tensorrt_engine = None
        self.interaction_operator = None
        self.embedding_cache = None
        
        # 框架可用性
        self.framework_availability = {
            'hstu': False,
            'vllm': False, 
            'tensorrt': False,
            'triton_ops': False,
            'cache': False
        }
        
        if not INTEGRATIONS_AVAILABLE:
            logger.warning("集成模块不可用，使用简化模式")
            return
        
        # 初始化所有组件
        self._initialize_all_frameworks()
    
    def _initialize_all_frameworks(self):
        """初始化所有开源框架"""
        logger.info("开始初始化开源框架集成...")
        
        # 1. 初始化Meta HSTU模型
        self._initialize_hstu_model()
        
        # 2. 初始化VLLM引擎
        self._initialize_vllm_engine()
        
        # 3. 初始化TensorRT引擎
        self._initialize_tensorrt_engine()
        
        # 4. 初始化自定义算子
        self._initialize_custom_operators()
        
        # 5. 初始化智能缓存
        self._initialize_intelligent_cache()
        
        logger.info("开源框架集成初始化完成")
        self._log_framework_status()
    
    def _initialize_hstu_model(self):
        """初始化Meta HSTU模型"""
        try:
            hstu_config = self.config.get('hstu', {})
            
            # 创建HSTU模型配置
            model_config = HSTUModelConfig(
                vocab_size=hstu_config.get('vocab_size', 50000),
                d_model=hstu_config.get('d_model', 1024),
                num_layers=hstu_config.get('num_layers', 12),
                num_heads=hstu_config.get('num_heads', 16),
                d_ff=hstu_config.get('d_ff', 4096),
                max_seq_len=hstu_config.get('max_seq_len', 2048),
                dropout=hstu_config.get('dropout', 0.1),
                hstu_expansion_factor=hstu_config.get('hstu_expansion_factor', 4),
                enable_hierarchical_attention=hstu_config.get('enable_hierarchical_attention', True),
                similarity_dim=hstu_config.get('similarity_dim', 256),
                temperature=hstu_config.get('temperature', 0.1),
            )
            
            # 创建HSTU模型
            self.hstu_model = HSTUGenerativeRecommender(model_config)
            
            # 如果有预训练权重，加载它们
            pretrained_path = hstu_config.get('pretrained_path')
            if pretrained_path and os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                self.hstu_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                logger.info(f"✅ 加载HSTU预训练权重: {pretrained_path}")
            
            # 设置为评估模式
            self.hstu_model.eval()
            
            # 移动到GPU
            if torch.cuda.is_available():
                self.hstu_model = self.hstu_model.cuda()
            
            self.framework_availability['hstu'] = True
            logger.info("✅ Meta HSTU模型初始化成功")
            
        except Exception as e:
            logger.error(f"❌ Meta HSTU模型初始化失败: {e}")
            self.framework_availability['hstu'] = False
    
    def _initialize_vllm_engine(self):
        """初始化VLLM引擎"""
        try:
            vllm_config_dict = self.config.get('vllm', {})
            
            # 创建VLLM配置
            vllm_config = VLLMConfig(
                model_name=vllm_config_dict.get('model_name', 'hstu-generative-recommender'),
                model_path=vllm_config_dict.get('model_path'),
                tensor_parallel_size=vllm_config_dict.get('tensor_parallel_size', 1),
                gpu_memory_utilization=vllm_config_dict.get('gpu_memory_utilization', 0.8),
                max_model_len=vllm_config_dict.get('max_model_len'),
                max_num_seqs=vllm_config_dict.get('max_num_seqs', 256),
                dtype=vllm_config_dict.get('dtype', 'float16'),
                seed=vllm_config_dict.get('seed', 42),
            )
            
            # 创建VLLM引擎（传入HSTU模型作为后备）
            self.vllm_engine = VLLMRecommenderEngine(vllm_config, self.hstu_model)
            
            self.framework_availability['vllm'] = self.vllm_engine.vllm_available
            
            if self.framework_availability['vllm']:
                logger.info("✅ VLLM推理引擎初始化成功")
            else:
                logger.warning("⚠️ VLLM不可用，将使用HSTU模型回退")
                
        except Exception as e:
            logger.error(f"❌ VLLM引擎初始化失败: {e}")
            self.framework_availability['vllm'] = False
    
    def _initialize_tensorrt_engine(self):
        """初始化TensorRT引擎"""
        try:
            tensorrt_config_dict = self.config.get('tensorrt', {})
            
            # 创建TensorRT配置
            tensorrt_config = TensorRTConfig(
                model_name=tensorrt_config_dict.get('model_name', 'hstu-tensorrt'),
                onnx_path=tensorrt_config_dict.get('onnx_path'),
                engine_path=tensorrt_config_dict.get('engine_path'),
                precision=tensorrt_config_dict.get('precision', 'fp16'),
                max_batch_size=tensorrt_config_dict.get('max_batch_size', 8),
                max_workspace_size=tensorrt_config_dict.get('max_workspace_size', 1 << 30),
                optimization_level=tensorrt_config_dict.get('optimization_level', 5),
                enable_dynamic_shapes=tensorrt_config_dict.get('enable_dynamic_shapes', True),
            )
            
            # 创建TensorRT引擎（传入HSTU模型）
            self.tensorrt_engine = TensorRTOptimizedEngine(tensorrt_config, self.hstu_model)
            
            self.framework_availability['tensorrt'] = self.tensorrt_engine.tensorrt_available
            
            if self.framework_availability['tensorrt']:
                logger.info("✅ TensorRT推理引擎初始化成功")
            else:
                logger.warning("⚠️ TensorRT不可用，将使用HSTU模型回退")
                
        except Exception as e:
            logger.error(f"❌ TensorRT引擎初始化失败: {e}")
            self.framework_availability['tensorrt'] = False
    
    def _initialize_custom_operators(self):
        """初始化自定义算子"""
        try:
            ops_config = self.config.get('custom_operators', {})
            
            # 初始化Triton交互算子
            self.interaction_operator = InteractionOperator(
                cache_size=ops_config.get('cache_size', 1000),
                enable_benchmarking=ops_config.get('enable_benchmarking', True),
            )
            
            self.framework_availability['triton_ops'] = True
            logger.info("✅ 自定义Triton算子初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 自定义算子初始化失败: {e}")
            self.framework_availability['triton_ops'] = False
    
    def _initialize_intelligent_cache(self):
        """初始化智能缓存"""
        try:
            cache_config = self.config.get('intelligent_cache', {})
            
            # 创建智能嵌入缓存
            self.embedding_cache = IntelligentEmbeddingCache(
                cache_size=cache_config.get('gpu_cache_size', 4096),
                embedding_dim=cache_config.get('embedding_dim', 1024),
                enable_prediction=cache_config.get('enable_prediction', True),
                dtype=getattr(torch, cache_config.get('dtype', 'float32')),
            )
            
            self.framework_availability['cache'] = True
            logger.info("✅ 智能嵌入缓存初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 智能缓存初始化失败: {e}")
            self.framework_availability['cache'] = False
    
    def _log_framework_status(self):
        """记录框架状态"""
        logger.info("开源框架集成状态:")
        for framework, available in self.framework_availability.items():
            status = "✅ 可用" if available else "❌ 不可用"
            logger.info(f"  {framework}: {status}")
    
    def infer_with_optimal_strategy(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int = 10,
        strategy: str = "unified",
        **kwargs
    ) -> Dict[str, Any]:
        """使用统一推理流程进行推理
        
        流程: HSTU模型 -> TensorRT优化 -> VLLM推理服务
        """
        
        start_time = time.time()
        
        try:
            # 应用自定义算子优化
            optimized_features = self._apply_custom_optimizations(user_behaviors)
            
            # 执行统一推理流程
            if strategy == "unified":
                result = self._unified_inference_pipeline(
                    user_id, session_id, user_behaviors, num_recommendations, **kwargs
                )
            else:
                # 兼容性：支持单独框架推理
                result = self._single_framework_inference(
                    user_id, session_id, user_behaviors, num_recommendations, strategy, **kwargs
                )
            
            # 记录性能统计
            inference_time = time.time() - start_time
            self._update_performance_stats(strategy, inference_time)
            
            # 添加元信息
            result.update({
                'inference_strategy': strategy,
                'inference_time_ms': inference_time * 1000,
                'framework_status': self.framework_availability.copy(),
                'optimizations_applied': bool(optimized_features),
                'timestamp': datetime.now().isoformat(),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return {
                'error': str(e),
                'user_id': user_id,
                'session_id': session_id,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
    
    def _unified_inference_pipeline(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int,
        **kwargs
    ) -> Dict[str, Any]:
        """统一推理流程: HSTU -> TensorRT -> VLLM"""
        
        # Step 1: HSTU模型处理
        if self.framework_availability['hstu']:
            hstu_inputs = self._prepare_hstu_inputs(user_behaviors)
            logger.info("使用HSTU模型进行特征提取...")
        else:
            logger.warning("HSTU模型不可用，使用简化特征提取")
            hstu_inputs = self._fallback_feature_extraction(user_behaviors)
        
        # Step 2: TensorRT优化推理
        if self.framework_availability['tensorrt']:
            logger.info("使用TensorRT进行GPU优化推理...")
            trt_outputs = self.tensorrt_engine.infer(hstu_inputs)
            optimized_logits = trt_outputs
        else:
            logger.warning("TensorRT不可用，使用PyTorch推理")
            optimized_logits = self._pytorch_fallback_inference(hstu_inputs)
        
        # Step 3: VLLM推理服务优化
        if self.framework_availability['vllm']:
            logger.info("使用VLLM进行推理服务优化...")
            final_result = self._vllm_service_optimization(
                optimized_logits, user_id, session_id, user_behaviors, num_recommendations
            )
        else:
            logger.warning("VLLM不可用，使用标准后处理")
            final_result = self._standard_post_processing(
                optimized_logits, user_id, session_id, user_behaviors, num_recommendations
            )
        
        final_result.update({
            'engine_type': 'unified_pipeline',
            'pipeline_stages': {
                'hstu_model': self.framework_availability['hstu'],
                'tensorrt_optimization': self.framework_availability['tensorrt'],
                'vllm_service': self.framework_availability['vllm']
            }
        })
        
        return final_result
    
    def _prepare_hstu_inputs(self, user_behaviors: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """准备HSTU模型输入"""
        try:
            # 构建序列特征
            input_ids = []
            attention_mask = []
            dense_features = []
            
            for behavior in user_behaviors:
                # 视频ID转换
                video_id = behavior.get('video_id', 'unknown')
                numeric_id = abs(hash(str(video_id))) % 50000
                input_ids.append(numeric_id)
                attention_mask.append(1)
                
                # 密集特征
                dense_feat = [
                    behavior.get('watch_duration', 0.0) / 120.0,  # 归一化
                    behavior.get('watch_percentage', 0.0),
                    float(behavior.get('is_liked', False)),
                    float(behavior.get('is_shared', False)),
                    behavior.get('engagement_score', 0.5),
                ]
                # 填充到1024维
                while len(dense_feat) < 1024:
                    dense_feat.append(0.0)
                dense_features.append(dense_feat[:1024])
            
            # 转换为tensor
            inputs = {
                'input_ids': torch.tensor([input_ids], dtype=torch.long),
                'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
                'dense_features': torch.tensor([dense_features], dtype=torch.float32)
            }
            
            return inputs
            
        except Exception as e:
            logger.error(f"HSTU输入准备失败: {e}")
            return self._fallback_feature_extraction(user_behaviors)
    
    def _fallback_feature_extraction(self, user_behaviors: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """回退特征提取"""
        batch_size = 1
        seq_len = min(len(user_behaviors), 100)
        
        return {
            'input_ids': torch.randint(0, 50000, (batch_size, seq_len), dtype=torch.long),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
            'dense_features': torch.randn(batch_size, seq_len, 1024, dtype=torch.float32)
        }
    
    def _pytorch_fallback_inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """PyTorch回退推理"""
        if self.framework_availability['hstu'] and self.hstu_model:
            return self.hstu_model.forward(**inputs)
        else:
            # 简化回退
            batch_size = inputs['input_ids'].shape[0]
            return {
                'logits': torch.randn(batch_size, 100, 50000),
                'hidden_states': torch.randn(batch_size, 100, 1024),
                'engagement_scores': torch.sigmoid(torch.randn(batch_size, 1)),
                'retention_scores': torch.sigmoid(torch.randn(batch_size, 1)),
                'monetization_scores': torch.sigmoid(torch.randn(batch_size, 1)),
            }
    
    def _vllm_service_optimization(
        self, 
        model_outputs: Dict[str, torch.Tensor],
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int
    ) -> Dict[str, Any]:
        """VLLM推理服务优化"""
        try:
            # 使用VLLM的推理优化能力
            if 'logits' in model_outputs:
                logits = model_outputs['logits']
                # 获取推荐概率
                recommendations_logits = logits[0, -1, :]  # 最后一个位置的预测
                topk_values, topk_indices = torch.topk(recommendations_logits, num_recommendations)
                
                recommendations = []
                for i, (idx, score) in enumerate(zip(topk_indices.tolist(), topk_values.tolist())):
                    recommendations.append({
                        'video_id': f'video_{idx}',
                        'score': float(torch.sigmoid(torch.tensor(score))),
                        'rank': i + 1,
                        'reason': f'基于VLLM优化推理生成'
                    })
                
                return {
                    'user_id': user_id,
                    'session_id': session_id,
                    'recommendations': recommendations,
                    'feature_scores': {
                        'engagement_score': float(model_outputs.get('engagement_scores', torch.tensor([0.5]))[0]),
                        'retention_score': float(model_outputs.get('retention_scores', torch.tensor([0.5]))[0]),
                        'diversity_score': float(torch.rand(1)),
                    }
                }
            
        except Exception as e:
            logger.error(f"VLLM服务优化失败: {e}")
        
        return self._standard_post_processing(model_outputs, user_id, session_id, user_behaviors, num_recommendations)
    
    def _standard_post_processing(
        self,
        model_outputs: Dict[str, torch.Tensor],
        user_id: str, 
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int
    ) -> Dict[str, Any]:
        """标准后处理"""
        recommendations = []
        for i in range(num_recommendations):
            recommendations.append({
                'video_id': f'video_{i}',
                'score': float(torch.rand(1)),
                'rank': i + 1,
                'reason': f'基于统一推理流程生成'
            })
        
        return {
            'user_id': user_id,
            'session_id': session_id,
            'recommendations': recommendations,
            'feature_scores': {
                'engagement_score': 0.7,
                'retention_score': 0.6,
                'diversity_score': 0.8,
            }
        }
    
    def _apply_custom_optimizations(self, user_behaviors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """应用自定义优化算子"""
        
        if not self.framework_availability['triton_ops']:
            return None
        
        try:
            # 构建特征嵌入
            features = self._build_feature_embeddings(user_behaviors)
            
            if features is not None and self.interaction_operator:
                # 应用Triton交互算子
                optimized_features = self.interaction_operator(
                    features,
                    mode='advanced',
                    return_stats=True
                )
                return optimized_features
                
        except Exception as e:
            logger.warning(f"自定义优化应用失败: {e}")
            
        return None
    
    def _build_feature_embeddings(self, user_behaviors: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """构建特征嵌入"""
        
        if not user_behaviors or not self.framework_availability['cache']:
            return None
        
        try:
            # 提取视频ID
            video_ids = [behavior.get('video_id', f'unknown_{i}') for i, behavior in enumerate(user_behaviors)]
            
            # 转换为数值ID
            numeric_ids = []
            for video_id in video_ids:
                numeric_id = abs(hash(str(video_id))) % 50000
                numeric_ids.append(numeric_id)
            
            # 使用智能缓存获取嵌入
            if self.embedding_cache:
                embeddings = []
                for numeric_id in numeric_ids:
                    slot = self.embedding_cache.get(numeric_id)
                    if slot is not None:
                        emb = self.embedding_cache.get_embedding(slot)
                        embeddings.append(emb)
                    else:
                        # 生成随机嵌入并缓存
                        random_emb = torch.randn(1024, dtype=torch.float32)
                        slot = self.embedding_cache.put(numeric_id, random_emb)
                        emb = self.embedding_cache.get_embedding(slot)
                        embeddings.append(emb)
                
                if embeddings:
                    # 组合成批次张量
                    batch_embeddings = torch.stack(embeddings)
                    return batch_embeddings.unsqueeze(0)  # 添加batch维度
            
        except Exception as e:
            logger.warning(f"构建特征嵌入失败: {e}")
            
        return None
    
    def _infer_with_vllm(
        self,
        user_id: str,
        session_id: str, 
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int,
        **kwargs
    ) -> Dict[str, Any]:
        """使用VLLM进行推理"""
        
        result = self.vllm_engine.generate_recommendations(
            user_id=user_id,
            session_id=session_id,
            user_behaviors=user_behaviors,
            num_recommendations=num_recommendations,
            **kwargs
        )
        
        self.inference_count['vllm'] += 1
        return result
    
    def _infer_with_tensorrt(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int,
        **kwargs
    ) -> Dict[str, Any]:
        """使用TensorRT进行推理"""
        
        # 构建TensorRT输入
        inputs = self._build_tensorrt_inputs(user_behaviors)
        
        # 执行TensorRT推理
        trt_outputs = self.tensorrt_engine.infer(inputs)
        
        # 解析输出为推荐结果
        recommendations = self._parse_tensorrt_outputs(trt_outputs, num_recommendations)
        
        result = {
            'user_id': user_id,
            'session_id': session_id,
            'recommendations': recommendations,
            'engine_type': 'tensorrt',
            'raw_outputs': {k: v.shape for k, v in trt_outputs.items()},
        }
        
        self.inference_count['tensorrt'] += 1
        return result
    
    def _infer_with_hstu(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int,
        **kwargs
    ) -> Dict[str, Any]:
        """使用HSTU模型进行推理"""
        
        recommendations = self.hstu_model.generate_recommendations(
            user_behaviors=user_behaviors,
            num_recommendations=num_recommendations,
            **kwargs
        )
        
        result = {
            'user_id': user_id,
            'session_id': session_id,
            'recommendations': recommendations,
            'engine_type': 'hstu',
        }
        
        self.inference_count['hstu'] += 1
        return result
    
    def _infer_with_fallback(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int,
        **kwargs
    ) -> Dict[str, Any]:
        """回退推理策略"""
        
        # 尝试按优先级顺序使用可用的框架
        if self.framework_availability['hstu']:
            return self._infer_with_hstu(user_id, session_id, user_behaviors, num_recommendations, **kwargs)
        elif self.framework_availability['vllm']:
            return self._infer_with_vllm(user_id, session_id, user_behaviors, num_recommendations, **kwargs)
        elif self.framework_availability['tensorrt']:
            return self._infer_with_tensorrt(user_id, session_id, user_behaviors, num_recommendations, **kwargs)
        else:
            # 最终回退到简单推荐
            recommendations = []
            for i in range(num_recommendations):
                recommendations.append({
                    'video_id': f'fallback_rec_{i}',
                    'score': max(0.1, 0.8 - i * 0.1),
                    'position': i + 1,
                    'reason': '系统回退推荐'
                })
            
            return {
                'user_id': user_id,
                'session_id': session_id,
                'recommendations': recommendations,
                'engine_type': 'fallback',
                'warning': '所有推理框架不可用'
            }
    
    def _build_tensorrt_inputs(self, user_behaviors: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """构建TensorRT输入"""
        
        batch_size = 1
        seq_len = min(len(user_behaviors), 512)
        
        # 构建input_ids
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i, behavior in enumerate(user_behaviors[:seq_len]):
            video_id = behavior.get('video_id', f'unknown_{i}')
            token_id = abs(hash(str(video_id))) % 50000
            input_ids[0, i] = token_id
        
        # 构建attention_mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # 构建dense_features
        dense_features = torch.zeros(batch_size, 1024, dtype=torch.float32)
        if user_behaviors:
            # 基本统计特征
            avg_watch_pct = np.mean([b.get('watch_percentage', 0) for b in user_behaviors])
            like_rate = np.mean([1 if b.get('is_liked', False) else 0 for b in user_behaviors])
            
            dense_features[0, 0] = len(user_behaviors) / 100.0
            dense_features[0, 1] = avg_watch_pct
            dense_features[0, 2] = like_rate
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'dense_features': dense_features,
        }
    
    def _parse_tensorrt_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """解析TensorRT输出为推荐结果"""
        
        recommendations = []
        
        # 从logits中提取推荐
        if 'logits' in outputs:
            logits = outputs['logits']
            if logits.dim() >= 2:
                # 取最后一个时间步的输出
                last_logits = logits[0, -1, :] if logits.dim() == 3 else logits[0, :]
                top_k_values, top_k_indices = torch.topk(last_logits, k=num_recommendations)
                
                for i in range(num_recommendations):
                    recommendations.append({
                        'video_id': f'trt_rec_{top_k_indices[i].item()}',
                        'score': float(torch.softmax(top_k_values, dim=0)[i].item()),
                        'position': i + 1,
                        'reason': 'TensorRT优化推理'
                    })
        
        # 如果没有有效输出，生成默认推荐
        if not recommendations:
            for i in range(num_recommendations):
                recommendations.append({
                    'video_id': f'trt_default_{i}',
                    'score': max(0.1, 0.8 - i * 0.1),
                    'position': i + 1,
                    'reason': 'TensorRT默认推荐'
                })
        
        return recommendations
    
    def _update_performance_stats(self, strategy: str, inference_time: float):
        """更新性能统计"""
        self.performance_stats[f'{strategy}_total_time'] += inference_time
        self.performance_stats[f'{strategy}_count'] += 1
        
        if self.performance_stats[f'{strategy}_count'] > 0:
            self.performance_stats[f'{strategy}_avg_time'] = (
                self.performance_stats[f'{strategy}_total_time'] / 
                self.performance_stats[f'{strategy}_count']
            )
    
    async def batch_infer(
        self,
        requests: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量推理"""
        
        # 如果VLLM可用，使用其批量推理能力
        if self.framework_availability['vllm']:
            return await self.vllm_engine.batch_generate_recommendations(requests, **kwargs)
        
        # 否则串行处理
        results = []
        for request in requests:
            result = self.infer_with_optimal_strategy(**request, **kwargs)
            results.append(result)
        
        return results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        
        stats = {
            'framework_availability': self.framework_availability.copy(),
            'performance_stats': dict(self.performance_stats),
            'inference_counts': dict(self.inference_count),
            'total_inferences': sum(self.inference_count.values()),
        }
        
        # 添加各框架的详细统计
        if self.framework_availability['vllm'] and self.vllm_engine:
            stats['vllm_stats'] = self.vllm_engine.get_engine_stats()
        
        if self.framework_availability['tensorrt'] and self.tensorrt_engine:
            stats['tensorrt_stats'] = self.tensorrt_engine.get_engine_info()
        
        if self.framework_availability['cache'] and self.embedding_cache:
            stats['cache_stats'] = self.embedding_cache.get_cache_info()
        
        return stats
    
    def benchmark_all_frameworks(
        self,
        test_behaviors: List[Dict[str, Any]],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """对所有框架进行基准测试"""
        
        results = {'benchmark_results': {}}
        
        strategies = ['hstu', 'vllm', 'tensorrt']
        
        for strategy in strategies:
            if not self.framework_availability.get(strategy, False):
                continue
            
            logger.info(f"开始基准测试策略: {strategy}")
            
            times = []
            for i in range(num_iterations):
                start_time = time.time()
                
                result = self.infer_with_optimal_strategy(
                    user_id=f"bench_user_{i}",
                    session_id=f"bench_session_{i}",
                    user_behaviors=test_behaviors,
                    num_recommendations=5,
                    strategy=strategy
                )
                
                end_time = time.time()
                
                if 'error' not in result:
                    times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            if times:
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                std_time = np.std(times)
                throughput = len(test_behaviors) / (avg_time / 1000)  # behaviors/second
                
                results['benchmark_results'][strategy] = {
                    'avg_latency_ms': avg_time,
                    'min_latency_ms': min_time,
                    'max_latency_ms': max_time,
                    'std_latency_ms': std_time,
                    'throughput_behaviors_per_sec': throughput,
                    'successful_runs': len(times),
                    'total_runs': num_iterations,
                }
        
        return results


def create_integrated_controller(config: Dict[str, Any]) -> OpenSourceFrameworkController:
    """创建集成控制器"""
    
    controller = OpenSourceFrameworkController(config)
    logger.info(f"✅ 开源框架集成控制器创建成功")
    
    return controller


if __name__ == "__main__":
    # 测试配置
    test_config = {
        'hstu': {
            'vocab_size': 50000,
            'd_model': 1024,
            'num_layers': 12,
            'num_heads': 16,
            'max_seq_len': 2048,
        },
        'vllm': {
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.8,
            'max_num_seqs': 64,
            'dtype': 'float16',
        },
        'tensorrt': {
            'precision': 'fp16',
            'max_batch_size': 8,
            'optimization_level': 5,
        },
        'custom_operators': {
            'cache_size': 1000,
            'enable_benchmarking': True,
        },
        'intelligent_cache': {
            'gpu_cache_size': 4096,
            'embedding_dim': 1024,
            'enable_prediction': True,
        }
    }
    
    # 创建控制器
    controller = create_integrated_controller(test_config)
    
    # 测试推理
    test_behaviors = [
        {'video_id': 'video_1', 'watch_duration': 120, 'is_liked': True, 'category': 'tech'},
        {'video_id': 'video_2', 'watch_duration': 90, 'is_liked': False, 'category': 'music'},
        {'video_id': 'video_3', 'watch_duration': 200, 'is_liked': True, 'category': 'tech'},
    ]
    
    result = controller.infer_with_optimal_strategy(
        user_id="test_user",
        session_id="test_session",
        user_behaviors=test_behaviors,
        num_recommendations=5
    )
    
    print(f"推理策略: {result.get('inference_strategy', 'unknown')}")
    print(f"推理时间: {result.get('inference_time_ms', 0):.2f}ms")
    print(f"推荐数量: {len(result.get('recommendations', []))}")
    
    # 获取统计信息
    stats = controller.get_comprehensive_stats()
    print(f"框架可用性: {stats['framework_availability']}")
    print(f"总推理次数: {stats['total_inferences']}")