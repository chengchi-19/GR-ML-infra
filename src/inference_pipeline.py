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
from src.export_onnx import GenerativeRecommendationModel
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
        
        # 默认模型配置
        if model_config is None:
            model_config = {
                "vocab_size": 10000,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_features": 32,  # 扩展特征维度
                "num_layers": 4,
                "max_seq_len": 512
            }
        
        # 初始化模型
        self.model = GenerativeRecommendationModel(**model_config)
        self.model.eval()
        
        # 初始化用户行为处理器
        self.behavior_processor = UserBehaviorProcessor(max_sequence_length=max_sequence_length)
        
        # 初始化嵌入服务 (CPU模式)
        try:
            self.embedding_service = EmbeddingService(
                num_items=model_config["vocab_size"],
                emb_dim=model_config["embedding_dim"],
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
        
        # 密集特征 (32维)
        dense_features = torch.zeros(batch_size, 32, dtype=torch.float32)
        
        # 填充密集特征
        if features["sequence_length"] > 0:
            # 观看时长特征 (0-9)
            watch_durations = features["watch_durations"]
            for i, duration in enumerate(watch_durations[:10]):
                dense_features[0, i] = duration / 100.0  # 归一化
            
            # 观看百分比特征 (10-14)
            watch_percentages = features["watch_percentages"]
            for i, percentage in enumerate(watch_percentages[:5]):
                dense_features[0, 10 + i] = percentage
            
            # 交互标志特征 (15-19)
            interaction_flags = features["interaction_flags"]
            dense_features[0, 15] = sum(interaction_flags["likes"]) / len(interaction_flags["likes"]) if interaction_flags["likes"] else 0
            dense_features[0, 16] = sum(interaction_flags["favorites"]) / len(interaction_flags["favorites"]) if interaction_flags["favorites"] else 0
            dense_features[0, 17] = sum(interaction_flags["shares"]) / len(interaction_flags["shares"]) if interaction_flags["shares"] else 0
            dense_features[0, 18] = sum(interaction_flags["comments"]) / len(interaction_flags["comments"]) if interaction_flags["comments"] else 0
            dense_features[0, 19] = sum(interaction_flags["follows"]) / len(interaction_flags["follows"]) if interaction_flags["follows"] else 0
            
            # 时间特征 (20-24)
            time_features = features["time_features"]
            if time_features["time_of_days"]:
                dense_features[0, 20] = np.mean(time_features["time_of_days"]) / 24.0  # 归一化
            if time_features["day_of_weeks"]:
                dense_features[0, 21] = np.mean(time_features["day_of_weeks"]) / 7.0  # 归一化
            
            # 设备特征 (25-29)
            device_features = features["device_features"]
            if device_features["device_types"]:
                dense_features[0, 25] = np.mean(device_features["device_types"]) / 4.0  # 归一化
            if device_features["network_types"]:
                dense_features[0, 26] = np.mean(device_features["network_types"]) / 4.0  # 归一化
            
            # 推荐特征 (30-31)
            recommendation_features = features["recommendation_features"]
            if recommendation_features["sources"]:
                dense_features[0, 30] = np.mean(recommendation_features["sources"]) / 6.0  # 归一化
            if recommendation_features["positions"]:
                dense_features[0, 31] = np.mean(recommendation_features["positions"]) / 10.0  # 归一化
        
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
                            num_recommendations: int = 10) -> Dict[str, Any]:
        """
        基于用户行为序列进行推荐推理
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            behaviors: 用户行为列表
            num_recommendations: 推荐数量
            
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
            
            # 3. 模型推理
            logger.info("执行模型推理")
            with torch.no_grad():
                # Prefill阶段
                logits, feature_scores, hidden_states = self.model.forward_prefill(
                    features["input_ids"],
                    features["dense_features"],
                    features["attention_mask"]
                )
                
                # 生成推荐
                recommendations = []
                current_input_ids = features["input_ids"]
                current_past_states = hidden_states
                
                for i in range(num_recommendations):
                    # Decode阶段
                    last_token = current_input_ids[:, -1:]
                    logits, scores, new_past_states = self.model.forward_decode(
                        last_token,
                        current_past_states,
                        features["dense_features"]
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
            
            # 4. 构建结果
            result = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "sequence_length": len(sequence.behaviors),
                "recommendations": recommendations,
                "feature_scores": {
                    "engagement_score": float(features["raw_features"]["statistical_features"]["like_rate"]),
                    "retention_score": float(features["raw_features"]["statistical_features"]["avg_watch_percentage"]),
                    "diversity_score": len(set(features["raw_features"]["video_ids"])) / len(features["raw_features"]["video_ids"]) if features["raw_features"]["video_ids"] else 0
                },
                "processing_time_ms": 0  # 可以添加实际的时间计算
            }
            
            logger.info(f"推荐完成，生成了 {len(recommendations)} 个推荐")
            return result
            
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
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
