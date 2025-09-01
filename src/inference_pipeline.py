#!/usr/bin/env python3
"""
集成用户行为序列的推理流水线
将用户行为序列处理集成到正常的推理流程中
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from src.user_behavior_schema import (
    VideoMetadata, UserBehavior, UserBehaviorSequence, UserBehaviorProcessor
)
from src.mtgr_model import create_mtgr_model, MTGRWrapper
from src.vllm_engine import create_vllm_engine, VLLMInferenceEngine
from src.embedding_service import EmbeddingService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserBehaviorInferencePipeline:
    """集成用户行为序列的推理流水线"""
    
    def __init__(self, 
                 model_config: Dict[str, Any] = None,
                 max_sequence_length: int = 50,
                 embedding_cache_size: int = 10000):
        """
        初始化推理流水线
        
        Args:
            model_config: 模型配置
            max_sequence_length: 最大序列长度
            embedding_cache_size: 嵌入缓存大小
        """
        self.max_sequence_length = max_sequence_length
        
        # 默认模型配置 - MTGR-large (约8B参数)
        if model_config is None:
            model_config = {
                "vocab_size": 50000,
                "d_model": 1024,
                "nhead": 16,
                "num_layers": 24,
                "d_ff": 4096,
                "max_seq_len": 2048,
                "num_features": 1024,
                "user_profile_dim": 256,
                "item_feature_dim": 512,
                "dropout": 0.1
            }
        
        # 初始化MTGR模型
        self.model = create_mtgr_model(model_config)
        self.model.eval()
        
        # 初始化VLLM推理引擎
        self.vllm_engine = create_vllm_engine(
            model_path="mtgr_model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048
        )
        
        # 记录模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"MTGR模型初始化完成，总参数量: {total_params:,} (约{total_params/1e9:.1f}B)")
        
        # 初始化用户行为处理器
        self.behavior_processor = UserBehaviorProcessor(max_sequence_length=max_sequence_length)
        
        # 初始化嵌入服务 (CPU模式)
        try:
            self.embedding_service = EmbeddingService(
                num_items=model_config["vocab_size"],
                emb_dim=model_config["d_model"],  # 使用d_model而不是embedding_dim
                gpu_cache_size=embedding_cache_size // 2,
                host_cache_size=embedding_cache_size // 2
            )
        except AssertionError:
            # 如果没有CUDA，使用CPU模式
            logger.warning("CUDA不可用，使用CPU模式")
            self.embedding_service = None
        
        # 视频元数据缓存
        self.video_metadata_cache = {}
        
        logger.info(f"推理流水线初始化完成: 模型参数={sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_video_metadata(self, video_id: str, **kwargs) -> VideoMetadata:
        """创建视频元数据"""
        if video_id in self.video_metadata_cache:
            return self.video_metadata_cache[video_id]
        
        # 默认视频元数据
        default_metadata = {
            "title": f"视频_{video_id}",
            "category": "entertainment",
            "tags": ["默认"],
            "duration": 30,
            "upload_time": datetime.now(),
            "creator_id": "default_creator",
            "creator_name": "默认创作者",
            "creator_followers": 1000,
            "video_quality": "hd",
            "language": "zh",
            "region": "china",
            "content_rating": "general"
        }
        
        # 更新用户提供的参数
        default_metadata.update(kwargs)
        
        metadata = VideoMetadata(video_id=video_id, **default_metadata)
        self.video_metadata_cache[video_id] = metadata
        
        return metadata
    
    def create_user_behavior(self, 
                           user_id: str,
                           session_id: str,
                           video_id: str,
                           timestamp: datetime,
                           **kwargs) -> UserBehavior:
        """创建用户行为记录"""
        
        # 获取或创建视频元数据
        video_metadata = self.create_video_metadata(video_id)
        
        # 默认行为参数
        default_behavior = {
            "watch_duration": 25,
            "watch_percentage": 0.83,
            "watch_complete": False,
            "rewatch_count": 0,
            "skip_duration": 0,
            "is_liked": True,
            "is_favorited": False,
            "is_shared": True,
            "is_commented": False,
            "is_followed_creator": True,
            "is_blocked_creator": False,
            "is_reported": False,
            "time_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_weekend": timestamp.weekday() >= 5,
            "is_holiday": False,
            "device_type": "mobile",
            "os_type": "ios",
            "network_type": "wifi",
            "language": "zh",
            "timezone": "UTC+8",
            "screen_size": "large",
            "user_level": 2,
            "user_vip": False,
            "session_duration": 120,
            "session_video_count": 5,
            "session_start_time": timestamp,
            "recommendation_source": "home",
            "recommendation_algorithm": "collaborative_filtering",
            "recommendation_position": 3
        }
        
        # 更新用户提供的参数
        default_behavior.update(kwargs)
        
        return UserBehavior(
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp,
            video_id=video_id,
            video_metadata=video_metadata,
            **default_behavior
        )
    
    def process_user_behavior_sequence(self, 
                                     user_id: str,
                                     session_id: str,
                                     behaviors: List[Dict[str, Any]]) -> UserBehaviorSequence:
        """处理用户行为序列"""
        
        # 转换行为数据为UserBehavior对象
        user_behaviors = []
        for behavior_data in behaviors:
            # 提取时间戳
            if isinstance(behavior_data.get("timestamp"), str):
                timestamp = datetime.fromisoformat(behavior_data["timestamp"])
            else:
                timestamp = behavior_data.get("timestamp", datetime.now())
            
            # 创建用户行为
            behavior = self.create_user_behavior(
                user_id=user_id,
                session_id=session_id,
                video_id=behavior_data["video_id"],
                timestamp=timestamp,
                **{k: v for k, v in behavior_data.items() if k not in ["user_id", "session_id", "video_id", "timestamp"]}
            )
            user_behaviors.append(behavior)
        
        # 按时间排序
        user_behaviors.sort(key=lambda x: x.timestamp)
        
        # 创建行为序列
        if user_behaviors:
            sequence_start_time = user_behaviors[0].timestamp
            sequence_end_time = user_behaviors[-1].timestamp
            total_duration = int((sequence_end_time - sequence_start_time).total_seconds())
        else:
            sequence_start_time = datetime.now()
            sequence_end_time = datetime.now()
            total_duration = 0
        
        sequence = UserBehaviorSequence(
            user_id=user_id,
            session_id=session_id,
            behaviors=user_behaviors,
            sequence_start_time=sequence_start_time,
            sequence_end_time=sequence_end_time,
            total_duration=total_duration,
            total_videos=len(user_behaviors)
        )
        
        return sequence
    
    def extract_features_from_sequence(self, sequence: UserBehaviorSequence) -> Dict[str, torch.Tensor]:
        """从行为序列中提取特征"""
        
        # 使用行为处理器提取特征
        features = self.behavior_processor.process_behavior_sequence(sequence)
        
        # 转换为张量
        batch_size = 1
        
        # 密集特征 (1024维)
        dense_features = torch.zeros(batch_size, 1024, dtype=torch.float32)
        
        # 填充密集特征 - 扩展到1024维
        if features["sequence_length"] > 0:
            # 基础用户行为特征 (0-99)
            # 观看时长特征 (0-19)
            watch_durations = features["watch_durations"]
            for i, duration in enumerate(watch_durations[:20]):
                dense_features[0, i] = duration / 100.0
            
            # 观看百分比特征 (20-39)
            watch_percentages = features["watch_percentages"]
            for i, percentage in enumerate(watch_percentages[:20]):
                dense_features[0, 20 + i] = percentage
            
            # 交互标志特征 (40-59)
            interaction_flags = features["interaction_flags"]
            for i, flag_type in enumerate(["likes", "favorites", "shares", "comments", "follows"]):
                if interaction_flags[flag_type]:
                    dense_features[0, 40 + i] = sum(interaction_flags[flag_type]) / len(interaction_flags[flag_type])
            
            # 时间特征 (60-79)
            time_features = features["time_features"]
            if time_features["time_of_days"]:
                for i, time_of_day in enumerate(time_features["time_of_days"][:20]):
                    dense_features[0, 60 + i] = time_of_day / 24.0  # 归一化
            if time_features["day_of_weeks"]:
                for i, day_of_week in enumerate(time_features["day_of_weeks"][:20]):
                    dense_features[0, 80 + i] = day_of_week / 7.0  # 归一化
            
            # 设备特征 (100-199)
            device_features = features["device_features"]
            if device_features["device_types"]:
                for i, device_type in enumerate(device_features["device_types"][:100]):
                    dense_features[0, 100 + i] = device_type / 4.0  # 归一化
            if device_features["network_types"]:
                for i, network_type in enumerate(device_features["network_types"][:100]):
                    dense_features[0, 200 + i] = network_type / 4.0  # 归一化
            
            # 推荐特征 (300-399)
            recommendation_features = features["recommendation_features"]
            if recommendation_features["sources"]:
                for i, source in enumerate(recommendation_features["sources"][:100]):
                    dense_features[0, 300 + i] = source / 6.0  # 归一化
            if recommendation_features["positions"]:
                for i, position in enumerate(recommendation_features["positions"][:100]):
                    dense_features[0, 400 + i] = position / 10.0  # 归一化
            
            # 统计特征 (500-599)
            statistical_features = features["statistical_features"]
            dense_features[0, 500] = statistical_features["avg_watch_duration"] / 100.0
            dense_features[0, 501] = statistical_features["avg_watch_percentage"]
            dense_features[0, 502] = statistical_features["like_rate"]
            dense_features[0, 503] = statistical_features["favorite_rate"]
            dense_features[0, 504] = statistical_features["share_rate"]
            
            # 用户画像特征 (600-799) - 模拟用户画像数据
            for i in range(200):
                dense_features[0, 600 + i] = np.random.uniform(0, 1)
            
            # 视频特征 (800-999) - 模拟视频特征数据
            for i in range(200):
                dense_features[0, 800 + i] = np.random.uniform(0, 1)
            
            # 序列特征 (1000-1023) - 序列长度、多样性等
            dense_features[0, 1000] = features["sequence_length"] / 100.0
            dense_features[0, 1001] = len(set(features["video_ids"])) / len(features["video_ids"]) if features["video_ids"] else 0
            dense_features[0, 1002] = np.std(watch_durations) if watch_durations else 0
            dense_features[0, 1003] = np.std(watch_percentages) if watch_percentages else 0
        
        # 输入ID序列 (使用视频ID的哈希值)
        input_ids = torch.zeros(batch_size, min(len(features["video_ids"]), 50), dtype=torch.long)
        for i, video_id in enumerate(features["video_ids"][:50]):
            # 简单的哈希函数
            hash_value = hash(video_id) % 10000
            input_ids[0, i] = hash_value
        
        # 注意力掩码
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "dense_features": dense_features,
            "attention_mask": attention_mask,
            "raw_features": features
        }
    
    def infer_recommendations(self, 
                            user_id: str,
                            session_id: str,
                            behaviors: List[Dict[str, Any]],
                            num_recommendations: int = 10,
                            use_vllm: bool = True) -> Dict[str, Any]:
        """
        基于用户行为序列进行推荐推理
        支持MTGR模型和VLLM推理优化
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            behaviors: 用户行为列表
            num_recommendations: 推荐数量
            use_vllm: 是否使用VLLM推理优化
            
        Returns:
            推荐结果字典
        """
        
        try:
            # 1. 处理用户行为序列
            logger.info(f"处理用户 {user_id} 的行为序列，共 {len(behaviors)} 个行为")
            sequence = self.process_user_behavior_sequence(user_id, session_id, behaviors)
            
            # 2. 提取特征
            logger.info("提取行为特征")
            features = self.extract_features_from_sequence(sequence)
            
            # 3. 选择推理方式
            if use_vllm and hasattr(self, 'vllm_engine'):
                logger.info("使用VLLM推理优化引擎")
                # 对于同步调用，使用MTGR推理
                logger.info("同步调用模式，使用MTGR推理")
                return self._infer_with_mtgr(features, sequence, num_recommendations)
            else:
                logger.info("使用MTGR模型推理")
                return self._infer_with_mtgr(features, sequence, num_recommendations)
            
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def infer_recommendations_async(self, 
                                        user_id: str,
                                        session_id: str,
                                        behaviors: List[Dict[str, Any]],
                                        num_recommendations: int = 10,
                                        use_vllm: bool = True) -> Dict[str, Any]:
        """
        异步推荐推理接口
        支持VLLM推理优化
        """
        try:
            # 1. 处理用户行为序列
            logger.info(f"处理用户 {user_id} 的行为序列，共 {len(behaviors)} 个行为")
            sequence = self.process_user_behavior_sequence(user_id, session_id, behaviors)
            
            # 2. 提取特征
            logger.info("提取行为特征")
            features = self.extract_features_from_sequence(sequence)
            
            # 3. 选择推理方式
            if use_vllm and hasattr(self, 'vllm_engine'):
                logger.info("使用VLLM推理优化引擎")
                return await self._infer_with_vllm(user_id, session_id, behaviors, num_recommendations)
            else:
                logger.info("使用MTGR模型推理")
                return self._infer_with_mtgr(features, sequence, num_recommendations)
            
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _infer_with_vllm(self, 
                              user_id: str,
                              session_id: str,
                              behaviors: List[Dict[str, Any]],
                              num_recommendations: int) -> Dict[str, Any]:
        """使用VLLM推理优化"""
        try:
            start_time = time.time()
            
            # 调用VLLM引擎
            result = await self.vllm_engine.generate_recommendations(
                user_id=user_id,
                session_id=session_id,
                user_behaviors=behaviors,
                num_recommendations=num_recommendations
            )
            
            # 添加额外信息
            result.update({
                "timestamp": datetime.now().isoformat(),
                "sequence_length": len(behaviors),
                "engine_type": "vllm_optimized",
                "processing_time_ms": result.get("latency_ms", 0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"VLLM推理失败: {e}")
            # 回退到MTGR推理
            return self._infer_with_mtgr(
                self.extract_features_from_sequence(
                    self.process_user_behavior_sequence(user_id, session_id, behaviors)
                ),
                self.process_user_behavior_sequence(user_id, session_id, behaviors),
                num_recommendations
            )
    
    def _infer_with_mtgr(self, 
                         features: Dict[str, torch.Tensor],
                         sequence: UserBehaviorSequence,
                         num_recommendations: int) -> Dict[str, Any]:
        """使用MTGR模型推理"""
        logger.info("执行MTGR模型推理")
        
        with torch.no_grad():
            # Prefill阶段
            logits, feature_scores, engagement_scores, retention_scores, monetization_scores, hidden_states = self.model.forward_prefill(
                features["input_ids"],
                features["dense_features"],
                None,  # user_profile - 暂时为None
                None,  # item_features - 暂时为None
                features["attention_mask"]
            )
            
            # 生成推荐
            recommendations = []
            current_input_ids = features["input_ids"]
            current_past_states = hidden_states
            
            for i in range(num_recommendations):
                # Decode阶段
                last_token = current_input_ids[:, -1:]
                logits, scores, engagement_scores, retention_scores, monetization_scores, new_past_states = self.model.forward_decode(
                    last_token,
                    current_past_states,
                    features["dense_features"],
                    None,  # user_profile - 暂时为None
                    None   # item_features - 暂时为None
                )
                
                # 选择下一个推荐
                next_token = logits.argmax(dim=-1)
                recommendation_score = scores.item()
                
                # 转换为视频ID
                video_id = f"video_{next_token.item()}"
                
                recommendations.append({
                    "video_id": video_id,
                    "score": float(recommendation_score),
                    "position": i + 1
                })
                
                # 更新状态
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                current_past_states = new_past_states
        
        # 构建结果
        result = {
            "user_id": sequence.user_id,
            "session_id": sequence.session_id,
            "timestamp": datetime.now().isoformat(),
            "sequence_length": len(sequence.behaviors),
            "recommendations": recommendations,
            "feature_scores": {
                "engagement_score": float(features["raw_features"]["statistical_features"]["like_rate"]),
                "retention_score": float(features["raw_features"]["statistical_features"]["avg_watch_percentage"]),
                "diversity_score": len(set(features["raw_features"]["video_ids"])) / len(features["raw_features"]["video_ids"]) if features["raw_features"]["video_ids"] else 0
            },
            "processing_time_ms": 0,
            "engine_type": "mtgr_model"
        }
        
        logger.info(f"MTGR推荐完成，生成了 {len(recommendations)} 个推荐")
        return result
    
    def batch_infer(self, 
                   user_requests: List[Dict[str, Any]],
                   num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """批量推理"""
        
        results = []
        for request in user_requests:
            user_id = request["user_id"]
            session_id = request["session_id"]
            behaviors = request["behaviors"]
            
            result = self.infer_recommendations(
                user_id=user_id,
                session_id=session_id,
                behaviors=behaviors,
                num_recommendations=num_recommendations
            )
            results.append(result)
        
        return results

def create_sample_inference_request():
    """创建示例推理请求"""
    
    # 示例用户行为数据
    behaviors = [
        {
            "video_id": "video_001",
            "timestamp": "2024-01-15T14:30:00Z",
            "watch_duration": 25,
            "watch_percentage": 0.83,
            "is_liked": True,
            "is_favorited": False,
            "is_shared": True,
            "device_type": "mobile",
            "network_type": "wifi"
        },
        {
            "video_id": "video_015",
            "timestamp": "2024-01-15T14:30:30Z",
            "watch_duration": 30,
            "watch_percentage": 1.0,
            "is_liked": True,
            "is_favorited": True,
            "is_shared": False,
            "device_type": "mobile",
            "network_type": "wifi"
        },
        {
            "video_id": "video_089",
            "timestamp": "2024-01-15T14:31:00Z",
            "watch_duration": 15,
            "watch_percentage": 0.5,
            "is_liked": False,
            "is_favorited": False,
            "is_shared": False,
            "device_type": "mobile",
            "network_type": "wifi"
        }
    ]
    
    return {
        "user_id": "user_12345",
        "session_id": "session_67890",
        "behaviors": behaviors
    }

def main():
    """主函数 - 演示推理流水线"""
    
    print("=" * 60)
    print("用户行为序列推理流水线演示")
    print("=" * 60)
    
    # 创建推理流水线
    pipeline = UserBehaviorInferencePipeline()
    
    # 创建示例请求
    request = create_sample_inference_request()
    
    print(f"用户ID: {request['user_id']}")
    print(f"会话ID: {request['session_id']}")
    print(f"行为数量: {len(request['behaviors'])}")
    
    # 执行推理
    print("\n执行推理...")
    result = pipeline.infer_recommendations(
        user_id=request["user_id"],
        session_id=request["session_id"],
        behaviors=request["behaviors"],
        num_recommendations=5
    )
    
    # 输出结果
    print("\n推理结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 批量推理演示
    print("\n" + "=" * 60)
    print("批量推理演示")
    print("=" * 60)
    
    batch_requests = [
        request,
        {
            "user_id": "user_67890",
            "session_id": "session_12345",
            "behaviors": [
                {
                    "video_id": "video_234",
                    "timestamp": "2024-01-15T15:00:00Z",
                    "watch_duration": 20,
                    "watch_percentage": 0.67,
                    "is_liked": True,
                    "is_favorited": False,
                    "is_shared": False,
                    "device_type": "tablet",
                    "network_type": "wifi"
                }
            ]
        }
    ]
    
    batch_results = pipeline.batch_infer(batch_requests, num_recommendations=3)
    
    for i, result in enumerate(batch_results):
        print(f"\n用户 {i+1} 推荐结果:")
        print(f"  推荐数量: {len(result['recommendations'])}")
        print(f"  序列长度: {result['sequence_length']}")
        print(f"  参与度分数: {result['feature_scores']['engagement_score']:.3f}")

if __name__ == "__main__":
    main()
