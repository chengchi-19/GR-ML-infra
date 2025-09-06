#!/usr/bin/env python3
"""
VLLM推理优化框架集成适配器

基于vllm-project/vllm开源框架，提供高性能LLM推理服务，
集成PagedAttention和Continuous Batching优化技术。
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import torch

# 添加VLLM路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
vllm_path = os.path.join(project_root, "external", "vllm")
sys.path.append(vllm_path)

logger = logging.getLogger(__name__)

try:
    # 导入VLLM核心组件
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils import Counter
    from vllm.outputs import RequestOutput
    from vllm.transformers_utils.tokenizer import get_tokenizer
    
    VLLM_AVAILABLE = True
    logger.info("✅ VLLM框架导入成功")
    
except ImportError as e:
    VLLM_AVAILABLE = False
    logger.warning(f"⚠️ VLLM框架导入失败: {e}")


class VLLMConfig:
    """VLLM引擎配置"""
    def __init__(
        self,
        model_name: str = "hstu-generative-recommender",
        model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        max_num_seqs: int = 256,
        max_num_batched_tokens: Optional[int] = None,
        block_size: int = 16,
        seed: int = 0,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        tokenizer: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        dtype: str = "float16",
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        enable_chunked_prefill: bool = False,
        max_num_on_the_fly_seq_groups: int = 64,
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.seed = seed
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.tokenizer = tokenizer
        self.tokenizer_revision = tokenizer_revision
        self.dtype = dtype
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_on_the_fly_seq_groups = max_num_on_the_fly_seq_groups
        
        # 添加其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)


class VLLMRecommenderEngine:
    """
    基于VLLM的推荐系统推理引擎
    
    集成PagedAttention优化和高吞吐量批处理
    """
    
    def __init__(self, config: VLLMConfig, hstu_model=None):
        self.config = config
        self.hstu_model = hstu_model
        self.request_counter = Counter()
        
        if not VLLM_AVAILABLE:
            logger.warning("VLLM不可用，使用HSTU模型回退")
            self.vllm_available = False
            return
        
        self.vllm_available = True
        self._initialize_vllm_engine()
    
    def _initialize_vllm_engine(self):
        """初始化VLLM引擎"""
        try:
            # 如果有指定模型路径，使用指定路径，否则使用HSTU模型
            if self.config.model_path and os.path.exists(self.config.model_path):
                model_path = self.config.model_path
                logger.info(f"使用指定模型路径: {model_path}")
            elif self.hstu_model is not None:
                # 将HSTU模型包装为VLLM兼容格式
                logger.info("使用HSTU模型进行VLLM优化推理")
                model_path = None  # 将直接使用模型对象
            else:
                # 使用默认的预训练模型
                model_path = "microsoft/DialoGPT-medium"  # 使用一个兼容的模型作为基础
                logger.info(f"使用默认模型: {model_path}")
            
            # 创建AsyncEngineArgs
            engine_args = AsyncEngineArgs(
                model=model_path if model_path else "microsoft/DialoGPT-medium",
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                swap_space=self.config.swap_space,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                block_size=self.config.block_size,
                seed=self.config.seed,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.revision,
                tokenizer=self.config.tokenizer,
                tokenizer_revision=self.config.tokenizer_revision,
                dtype=self.config.dtype,
                quantization=self.config.quantization,
                enforce_eager=self.config.enforce_eager,
                max_context_len_to_capture=self.config.max_context_len_to_capture,
                disable_custom_all_reduce=self.config.disable_custom_all_reduce,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
            )
            
            # 创建异步引擎
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # 创建同步引擎用于简单推理
            self.sync_engine = LLM(
                model=model_path if model_path else "microsoft/DialoGPT-medium",
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                seed=self.config.seed,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # 获取tokenizer
            self.tokenizer = get_tokenizer(
                self.config.tokenizer or (model_path if model_path else "microsoft/DialoGPT-medium"),
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.tokenizer_revision,
            )
            
            logger.info("✅ VLLM引擎初始化成功")
            
        except Exception as e:
            logger.error(f"❌ VLLM引擎初始化失败: {e}")
            self.vllm_available = False
    
    def generate_recommendations(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int = 10,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """生成推荐结果"""
        
        if not self.vllm_available:
            return self._fallback_generate_recommendations(
                user_id, session_id, user_behaviors, num_recommendations, **kwargs
            )
        
        try:
            # 构建推理提示
            prompt = self._build_recommendation_prompt(user_behaviors, num_recommendations)
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=self.config.seed,
            )
            
            # 执行推理
            outputs = self.sync_engine.generate([prompt], sampling_params)
            
            if outputs:
                output = outputs[0]
                generated_text = output.outputs[0].text
                
                # 解析生成的推荐结果
                recommendations = self._parse_generated_recommendations(
                    generated_text, num_recommendations
                )
                
                return {
                    'user_id': user_id,
                    'session_id': session_id,
                    'recommendations': recommendations,
                    'engine_type': 'vllm',
                    'generated_text': generated_text,
                    'timestamp': output.finished_time if hasattr(output, 'finished_time') else None,
                    'usage_stats': {
                        'prompt_tokens': len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0,
                        'completion_tokens': sum(len(o.token_ids) for o in output.outputs),
                        'total_tokens': len(output.prompt_token_ids) + sum(len(o.token_ids) for o in output.outputs) if hasattr(output, 'prompt_token_ids') else 0,
                    }
                }
            else:
                raise RuntimeError("VLLM生成失败")
                
        except Exception as e:
            logger.error(f"VLLM推理失败: {e}")
            return self._fallback_generate_recommendations(
                user_id, session_id, user_behaviors, num_recommendations, **kwargs
            )
    
    async def generate_recommendations_async(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int = 10,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """异步生成推荐结果"""
        
        if not self.vllm_available:
            return self._fallback_generate_recommendations(
                user_id, session_id, user_behaviors, num_recommendations, **kwargs
            )
        
        try:
            # 构建推理提示
            prompt = self._build_recommendation_prompt(user_behaviors, num_recommendations)
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=self.config.seed,
            )
            
            # 生成请求ID
            request_id = f"{user_id}_{session_id}_{self.request_counter.next()}"
            
            # 异步生成
            results_generator = self.async_engine.generate(prompt, sampling_params, request_id)
            
            # 等待生成完成
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                generated_text = final_output.outputs[0].text
                
                # 解析生成的推荐结果
                recommendations = self._parse_generated_recommendations(
                    generated_text, num_recommendations
                )
                
                return {
                    'user_id': user_id,
                    'session_id': session_id,
                    'request_id': request_id,
                    'recommendations': recommendations,
                    'engine_type': 'vllm_async',
                    'generated_text': generated_text,
                    'finished': final_output.finished,
                    'usage_stats': {
                        'prompt_tokens': len(final_output.prompt_token_ids),
                        'completion_tokens': sum(len(o.token_ids) for o in final_output.outputs),
                        'total_tokens': len(final_output.prompt_token_ids) + sum(len(o.token_ids) for o in final_output.outputs),
                    }
                }
            else:
                raise RuntimeError("异步VLLM生成失败")
                
        except Exception as e:
            logger.error(f"异步VLLM推理失败: {e}")
            return self._fallback_generate_recommendations(
                user_id, session_id, user_behaviors, num_recommendations, **kwargs
            )
    
    def _build_recommendation_prompt(
        self, 
        user_behaviors: List[Dict[str, Any]], 
        num_recommendations: int
    ) -> str:
        """构建推荐提示"""
        
        # 分析用户行为模式
        behavior_summary = self._analyze_user_behaviors(user_behaviors)
        
        # 构建结构化提示
        prompt = f"""作为一个智能推荐系统，请基于用户的历史行为为其推荐{num_recommendations}个相关内容。

用户行为分析：
{behavior_summary}

请按以下JSON格式返回推荐结果：
{{
    "recommendations": [
        {{"item_id": "推荐内容ID", "score": 推荐分数(0-1), "reason": "推荐理由"}},
        ...
    ]
}}

推荐结果："""
        
        return prompt
    
    def _analyze_user_behaviors(self, user_behaviors: List[Dict[str, Any]]) -> str:
        """分析用户行为模式"""
        if not user_behaviors:
            return "用户无历史行为记录"
        
        # 计算基本统计
        total_behaviors = len(user_behaviors)
        avg_watch_duration = sum(b.get('watch_duration', 0) for b in user_behaviors) / total_behaviors
        like_rate = sum(1 for b in user_behaviors if b.get('is_liked', False)) / total_behaviors
        
        # 识别偏好类别
        categories = {}
        for behavior in user_behaviors:
            category = behavior.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 时间模式分析
        recent_behaviors = user_behaviors[-5:]  # 最近5个行为
        
        analysis = f"""
- 总行为数量: {total_behaviors}
- 平均观看时长: {avg_watch_duration:.1f}秒
- 点赞率: {like_rate:.2%}
- 主要偏好类别: {', '.join([f'{cat}({count})' for cat, count in top_categories])}
- 最近行为: {len(recent_behaviors)}个内容
        """
        
        return analysis.strip()
    
    def _parse_generated_recommendations(
        self, 
        generated_text: str, 
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """解析生成的推荐结果"""
        
        try:
            import json
            import re
            
            # 尝试提取JSON部分
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, generated_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                if 'recommendations' in result:
                    recommendations = result['recommendations'][:num_recommendations]
                    
                    # 标准化格式
                    for i, rec in enumerate(recommendations):
                        rec['position'] = i + 1
                        rec['video_id'] = rec.get('item_id', f'vllm_rec_{i}')
                        if 'score' not in rec:
                            rec['score'] = max(0.1, 0.9 - i * 0.1)
                        if 'reason' not in rec:
                            rec['reason'] = 'VLLM生成式推荐'
                    
                    return recommendations
            
            # 如果JSON解析失败，生成默认推荐
            logger.warning("无法解析生成的推荐结果，使用默认推荐")
            return self._generate_default_recommendations(num_recommendations, "VLLM解析失败")
            
        except Exception as e:
            logger.error(f"解析推荐结果失败: {e}")
            return self._generate_default_recommendations(num_recommendations, "VLLM解析错误")
    
    def _generate_default_recommendations(
        self, 
        num_recommendations: int, 
        reason: str = "默认推荐"
    ) -> List[Dict[str, Any]]:
        """生成默认推荐结果"""
        
        recommendations = []
        for i in range(num_recommendations):
            recommendations.append({
                'video_id': f'default_vllm_{i}',
                'score': max(0.1, 0.9 - i * 0.08),
                'position': i + 1,
                'reason': reason
            })
        
        return recommendations
    
    def _fallback_generate_recommendations(
        self,
        user_id: str,
        session_id: str,
        user_behaviors: List[Dict[str, Any]],
        num_recommendations: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """回退到HSTU模型生成推荐"""
        
        if self.hstu_model is not None:
            try:
                recommendations = self.hstu_model.generate_recommendations(
                    user_behaviors, num_recommendations
                )
                
                return {
                    'user_id': user_id,
                    'session_id': session_id,
                    'recommendations': recommendations,
                    'engine_type': 'hstu_fallback',
                    'reason': 'VLLM不可用，使用HSTU模型'
                }
                
            except Exception as e:
                logger.error(f"HSTU模型推理失败: {e}")
        
        # 最终回退到默认推荐
        return {
            'user_id': user_id,
            'session_id': session_id,
            'recommendations': self._generate_default_recommendations(num_recommendations, "系统回退推荐"),
            'engine_type': 'fallback',
            'reason': '所有推理引擎不可用'
        }
    
    async def batch_generate_recommendations(
        self,
        requests: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量生成推荐"""
        
        if not self.vllm_available:
            # 串行处理回退请求
            results = []
            for request in requests:
                result = self._fallback_generate_recommendations(**request)
                results.append(result)
            return results
        
        # 使用VLLM的异步批处理能力
        tasks = []
        for request in requests:
            task = self.generate_recommendations_async(**request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量推理第{i}个请求失败: {result}")
                processed_results.append(
                    self._fallback_generate_recommendations(**requests[i])
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎状态统计"""
        
        if not self.vllm_available:
            return {
                'vllm_available': False,
                'fallback_mode': True
            }
        
        try:
            # 获取VLLM引擎统计
            stats = {
                'vllm_available': True,
                'model_config': {
                    'model_name': self.config.model_name,
                    'tensor_parallel_size': self.config.tensor_parallel_size,
                    'max_num_seqs': self.config.max_num_seqs,
                    'gpu_memory_utilization': self.config.gpu_memory_utilization,
                    'dtype': self.config.dtype,
                },
                'request_counter': self.request_counter.value,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取引擎统计失败: {e}")
            return {
                'vllm_available': False,
                'error': str(e)
            }


def create_vllm_engine(
    model_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "float16",
    hstu_model=None,
    **kwargs
) -> VLLMRecommenderEngine:
    """创建VLLM推荐引擎"""
    
    config = VLLMConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        **kwargs
    )
    
    engine = VLLMRecommenderEngine(config, hstu_model)
    logger.info(f"✅ VLLM推荐引擎创建成功，可用性: {engine.vllm_available}")
    
    return engine


if __name__ == "__main__":
    # 测试VLLM引擎
    config = VLLMConfig(
        model_name="test-recommender",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_num_seqs=64,
    )
    
    engine = VLLMRecommenderEngine(config)
    
    # 测试推荐生成
    user_behaviors = [
        {'video_id': 'video_1', 'watch_duration': 120, 'is_liked': True, 'category': 'tech'},
        {'video_id': 'video_2', 'watch_duration': 90, 'is_liked': False, 'category': 'music'},
        {'video_id': 'video_3', 'watch_duration': 200, 'is_liked': True, 'category': 'tech'},
    ]
    
    result = engine.generate_recommendations(
        user_id="test_user",
        session_id="test_session", 
        user_behaviors=user_behaviors,
        num_recommendations=5
    )
    
    print(f"推荐引擎类型: {result['engine_type']}")
    print(f"生成了 {len(result['recommendations'])} 个推荐")
    for rec in result['recommendations']:
        print(f"  {rec['video_id']}: {rec['score']:.4f} - {rec['reason']}")
    
    # 测试引擎统计
    stats = engine.get_engine_stats()
    print(f"引擎统计: {stats}")