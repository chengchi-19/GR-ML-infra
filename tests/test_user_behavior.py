#!/usr/bin/env python3
"""
用户行为序列扩展功能测试
"""

import sys
import os
import unittest
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.user_behavior_schema import (
    VideoMetadata, UserBehavior, UserBehaviorSequence, UserBehaviorProcessor
)

class TestUserBehaviorSchema(unittest.TestCase):
    """测试用户行为序列扩展功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试用的视频元数据
        self.video_metadata = VideoMetadata(
            video_id="test_video_001",
            title="测试视频",
            category="entertainment",
            tags=["测试", "娱乐"],
            duration=30,
            upload_time=datetime(2024, 1, 10, 10, 0, 0),
            creator_id="creator_001",
            creator_name="测试创作者",
            creator_followers=1000,
            video_quality="hd",
            language="zh",
            region="china",
            content_rating="general",
            music_genre="pop"
        )
        
        # 创建测试用的用户行为
        self.user_behavior = UserBehavior(
            user_id="user_001",
            session_id="session_001",
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
            video_id="test_video_001",
            video_metadata=self.video_metadata,
            watch_duration=25,
            watch_percentage=0.83,
            watch_complete=False,
            rewatch_count=0,
            skip_duration=0,
            is_liked=True,
            is_favorited=False,
            is_shared=True,
            is_commented=False,
            is_followed_creator=True,
            is_blocked_creator=False,
            is_reported=False,
            share_platform="wechat",
            share_method="link",
            time_of_day=14,
            day_of_week=3,
            is_weekend=False,
            is_holiday=False,
            device_type="mobile",
            device_model="iPhone 14",
            os_type="ios",
            os_version="17.0",
            app_version="1.2.3",
            network_type="wifi",
            network_speed=100.0,
            location_country="china",
            location_province="beijing",
            location_city="beijing",
            language="zh",
            timezone="UTC+8",
            screen_size="large",
            screen_resolution="1170x2532",
            user_level=2,
            user_vip=False,
            user_age_group="young",
            user_gender="male",
            session_duration=120,
            session_video_count=5,
            session_start_time=datetime(2024, 1, 15, 14, 25, 0),
            recommendation_source="home",
            recommendation_algorithm="collaborative_filtering",
            recommendation_position=3,
            recommendation_score=0.85,
            engagement_score=0.75,
            retention_score=0.80,
            monetization_score=0.60
        )
        
        # 创建测试用的行为序列
        self.behavior_sequence = UserBehaviorSequence(
            user_id="user_001",
            session_id="session_001",
            behaviors=[self.user_behavior],
            sequence_start_time=datetime(2024, 1, 15, 14, 25, 0),
            sequence_end_time=datetime(2024, 1, 15, 14, 30, 0),
            total_duration=300,
            total_videos=5
        )
        
        # 创建处理器
        self.processor = UserBehaviorProcessor(max_sequence_length=50)
    
    def test_video_metadata_creation(self):
        """测试视频元数据创建"""
        self.assertEqual(self.video_metadata.video_id, "test_video_001")
        self.assertEqual(self.video_metadata.title, "测试视频")
        self.assertEqual(self.video_metadata.category, "entertainment")
        self.assertEqual(self.video_metadata.duration, 30)
        self.assertEqual(self.video_metadata.creator_followers, 1000)
    
    def test_user_behavior_creation(self):
        """测试用户行为创建"""
        self.assertEqual(self.user_behavior.user_id, "user_001")
        self.assertEqual(self.user_behavior.video_id, "test_video_001")
        self.assertEqual(self.user_behavior.watch_duration, 25)
        self.assertEqual(self.user_behavior.watch_percentage, 0.83)
        self.assertTrue(self.user_behavior.is_liked)
        self.assertFalse(self.user_behavior.is_favorited)
        self.assertTrue(self.user_behavior.is_shared)
        self.assertEqual(self.user_behavior.device_type, "mobile")
        self.assertEqual(self.user_behavior.network_type, "wifi")
        self.assertEqual(self.user_behavior.recommendation_source, "home")
    
    def test_behavior_sequence_creation(self):
        """测试行为序列创建"""
        self.assertEqual(self.behavior_sequence.user_id, "user_001")
        self.assertEqual(self.behavior_sequence.session_id, "session_001")
        self.assertEqual(len(self.behavior_sequence.behaviors), 1)
        self.assertEqual(self.behavior_sequence.total_duration, 300)
        self.assertEqual(self.behavior_sequence.total_videos, 5)
    
    def test_behavior_sequence_to_dict(self):
        """测试行为序列转换为字典"""
        sequence_dict = self.behavior_sequence.to_dict()
        
        self.assertIn("user_id", sequence_dict)
        self.assertIn("session_id", sequence_dict)
        self.assertIn("behaviors", sequence_dict)
        self.assertIn("total_duration", sequence_dict)
        self.assertIn("total_videos", sequence_dict)
        
        self.assertEqual(sequence_dict["user_id"], "user_001")
        self.assertEqual(sequence_dict["total_duration"], 300)
        self.assertEqual(len(sequence_dict["behaviors"]), 1)
    
    def test_processor_initialization(self):
        """测试处理器初始化"""
        self.assertEqual(self.processor.max_sequence_length, 50)
    
    def test_feature_extraction(self):
        """测试特征提取"""
        features = self.processor.process_behavior_sequence(self.behavior_sequence)
        
        # 检查基本特征
        self.assertIn("sequence_length", features)
        self.assertIn("video_ids", features)
        self.assertIn("watch_durations", features)
        self.assertIn("watch_percentages", features)
        
        # 检查交互特征
        self.assertIn("interaction_flags", features)
        interaction_flags = features["interaction_flags"]
        self.assertIn("likes", interaction_flags)
        self.assertIn("favorites", interaction_flags)
        self.assertIn("shares", interaction_flags)
        self.assertIn("comments", interaction_flags)
        self.assertIn("follows", interaction_flags)
        
        # 检查时间特征
        self.assertIn("time_features", features)
        time_features = features["time_features"]
        self.assertIn("time_of_days", time_features)
        self.assertIn("day_of_weeks", time_features)
        
        # 检查设备特征
        self.assertIn("device_features", features)
        device_features = features["device_features"]
        self.assertIn("device_types", device_features)
        self.assertIn("network_types", device_features)
        
        # 检查推荐特征
        self.assertIn("recommendation_features", features)
        recommendation_features = features["recommendation_features"]
        self.assertIn("sources", recommendation_features)
        self.assertIn("positions", recommendation_features)
        
        # 检查统计特征
        self.assertIn("statistical_features", features)
        statistical_features = features["statistical_features"]
        self.assertIn("avg_watch_duration", statistical_features)
        self.assertIn("avg_watch_percentage", statistical_features)
        self.assertIn("like_rate", statistical_features)
        self.assertIn("favorite_rate", statistical_features)
        self.assertIn("share_rate", statistical_features)
    
    def test_feature_values(self):
        """测试特征值"""
        features = self.processor.process_behavior_sequence(self.behavior_sequence)
        
        # 检查序列长度
        self.assertEqual(features["sequence_length"], 1)
        
        # 检查视频ID
        self.assertEqual(features["video_ids"], ["test_video_001"])
        
        # 检查观看时长
        self.assertEqual(features["watch_durations"], [25])
        
        # 检查观看百分比
        self.assertEqual(features["watch_percentages"], [0.83])
        
        # 检查交互标志
        self.assertEqual(features["interaction_flags"]["likes"], [1])
        self.assertEqual(features["interaction_flags"]["favorites"], [0])
        self.assertEqual(features["interaction_flags"]["shares"], [1])
        self.assertEqual(features["interaction_flags"]["comments"], [0])
        self.assertEqual(features["interaction_flags"]["follows"], [1])
        
        # 检查时间特征
        self.assertEqual(features["time_features"]["time_of_days"], [14])
        self.assertEqual(features["time_features"]["day_of_weeks"], [3])
        
        # 检查设备特征
        self.assertEqual(features["device_features"]["device_types"], [0])  # mobile
        self.assertEqual(features["device_features"]["network_types"], [0])  # wifi
        
        # 检查推荐特征
        self.assertEqual(features["recommendation_features"]["sources"], [0])  # home
        self.assertEqual(features["recommendation_features"]["positions"], [3])
        
        # 检查统计特征
        self.assertEqual(features["statistical_features"]["avg_watch_duration"], 25.0)
        self.assertEqual(features["statistical_features"]["avg_watch_percentage"], 0.83)
        self.assertEqual(features["statistical_features"]["like_rate"], 1.0)
        self.assertEqual(features["statistical_features"]["favorite_rate"], 0.0)
        self.assertEqual(features["statistical_features"]["share_rate"], 1.0)
    
    def test_encoding_functions(self):
        """测试编码函数"""
        # 测试设备类型编码
        self.assertEqual(self.processor._encode_device_type("mobile"), 0)
        self.assertEqual(self.processor._encode_device_type("tablet"), 1)
        self.assertEqual(self.processor._encode_device_type("desktop"), 2)
        self.assertEqual(self.processor._encode_device_type("tv"), 3)
        self.assertEqual(self.processor._encode_device_type("unknown"), 0)  # 默认值
        
        # 测试网络类型编码
        self.assertEqual(self.processor._encode_network_type("wifi"), 0)
        self.assertEqual(self.processor._encode_network_type("4g"), 1)
        self.assertEqual(self.processor._encode_network_type("5g"), 2)
        self.assertEqual(self.processor._encode_network_type("3g"), 3)
        self.assertEqual(self.processor._encode_network_type("unknown"), 0)  # 默认值
        
        # 测试推荐来源编码
        self.assertEqual(self.processor._encode_recommendation_source("home"), 0)
        self.assertEqual(self.processor._encode_recommendation_source("search"), 1)
        self.assertEqual(self.processor._encode_recommendation_source("following"), 2)
        self.assertEqual(self.processor._encode_recommendation_source("hashtag"), 3)
        self.assertEqual(self.processor._encode_recommendation_source("trending"), 4)
        self.assertEqual(self.processor._encode_recommendation_source("nearby"), 5)
        self.assertEqual(self.processor._encode_recommendation_source("unknown"), 0)  # 默认值
    
    def test_sequence_truncation(self):
        """测试序列截断"""
        # 创建超过最大长度的序列
        long_behaviors = []
        for i in range(60):  # 超过max_sequence_length=50
            behavior = UserBehavior(
                user_id=f"user_{i}",
                session_id="session_001",
                timestamp=datetime(2024, 1, 15, 14, 30, 0),
                video_id=f"video_{i}",
                video_metadata=self.video_metadata,
                watch_duration=25,
                watch_percentage=0.83,
                watch_complete=False,
                rewatch_count=0,
                skip_duration=0,
                is_liked=True,
                is_favorited=False,
                is_shared=True,
                is_commented=False,
                is_followed_creator=True,
                is_blocked_creator=False,
                is_reported=False,
                time_of_day=14,
                day_of_week=3,
                is_weekend=False,
                is_holiday=False,
                device_type="mobile",
                os_type="ios",
                network_type="wifi",
                language="zh",
                timezone="UTC+8",
                screen_size="large",
                user_level=2,
                user_vip=False,
                session_duration=120,
                session_video_count=5,
                session_start_time=datetime(2024, 1, 15, 14, 25, 0),
                recommendation_source="home",
                recommendation_algorithm="collaborative_filtering",
                recommendation_position=3
            )
            long_behaviors.append(behavior)
        
        long_sequence = UserBehaviorSequence(
            user_id="user_001",
            session_id="session_001",
            behaviors=long_behaviors,
            sequence_start_time=datetime(2024, 1, 15, 14, 25, 0),
            sequence_end_time=datetime(2024, 1, 15, 14, 30, 0),
            total_duration=300,
            total_videos=60
        )
        
        features = self.processor.process_behavior_sequence(long_sequence)
        
        # 检查序列被截断到最大长度
        self.assertEqual(features["sequence_length"], 50)
        self.assertEqual(len(features["video_ids"]), 50)
        self.assertEqual(len(features["watch_durations"]), 50)
        
        # 检查只保留最后50个行为
        self.assertEqual(features["video_ids"][0], "video_10")  # 第11个行为
        self.assertEqual(features["video_ids"][-1], "video_59")  # 最后一个行为

def run_performance_test():
    """运行性能测试"""
    print("运行用户行为序列处理性能测试...")
    
    # 创建大量测试数据
    processor = UserBehaviorProcessor(max_sequence_length=50)
    
    # 创建1000个行为序列
    sequences = []
    for i in range(1000):
        behaviors = []
        for j in range(30):  # 每个序列30个行为
            behavior = UserBehavior(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                timestamp=datetime(2024, 1, 15, 14, 30, 0),
                video_id=f"video_{i}_{j}",
                video_metadata=VideoMetadata(
                    video_id=f"video_{i}_{j}",
                    title=f"视频_{i}_{j}",
                    category="entertainment",
                    tags=["测试"],
                    duration=30,
                    upload_time=datetime(2024, 1, 10, 10, 0, 0),
                    creator_id="creator_001",
                    creator_name="测试创作者",
                    creator_followers=1000,
                    video_quality="hd",
                    language="zh",
                    region="china",
                    content_rating="general"
                ),
                watch_duration=25,
                watch_percentage=0.83,
                watch_complete=False,
                rewatch_count=0,
                skip_duration=0,
                is_liked=True,
                is_favorited=False,
                is_shared=True,
                is_commented=False,
                is_followed_creator=True,
                is_blocked_creator=False,
                is_reported=False,
                time_of_day=14,
                day_of_week=3,
                is_weekend=False,
                is_holiday=False,
                device_type="mobile",
                os_type="ios",
                network_type="wifi",
                language="zh",
                timezone="UTC+8",
                screen_size="large",
                user_level=2,
                user_vip=False,
                session_duration=120,
                session_video_count=5,
                session_start_time=datetime(2024, 1, 15, 14, 25, 0),
                recommendation_source="home",
                recommendation_algorithm="collaborative_filtering",
                recommendation_position=3
            )
            behaviors.append(behavior)
        
        sequence = UserBehaviorSequence(
            user_id=f"user_{i}",
            session_id=f"session_{i}",
            behaviors=behaviors,
            sequence_start_time=datetime(2024, 1, 15, 14, 25, 0),
            sequence_end_time=datetime(2024, 1, 15, 14, 30, 0),
            total_duration=300,
            total_videos=30
        )
        sequences.append(sequence)
    
    # 性能测试
    import time
    start_time = time.time()
    
    for sequence in sequences:
        features = processor.process_behavior_sequence(sequence)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"处理了 {len(sequences)} 个行为序列")
    print(f"总处理时间: {processing_time:.2f} 秒")
    print(f"平均每个序列处理时间: {processing_time/len(sequences)*1000:.2f} 毫秒")
    print(f"处理速度: {len(sequences)/processing_time:.2f} 序列/秒")

if __name__ == "__main__":
    # 运行单元测试
    print("运行用户行为序列扩展功能单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    print("\n" + "="*50)
    run_performance_test()
