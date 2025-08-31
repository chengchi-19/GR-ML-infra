#!/usr/bin/env python3
"""
用户行为序列字段定义
包含短视频推荐系统中常见的用户行为字段
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

@dataclass
class VideoMetadata:
    """视频元数据"""
    video_id: str
    title: str
    category: str
    tags: List[str]
    duration: int  # 秒
    upload_time: datetime
    creator_id: str
    creator_name: str
    creator_followers: int
    video_quality: str  # hd, sd, 4k
    language: str
    region: str
    content_rating: str  # general, mature, kids
    music_genre: Optional[str] = None
    hashtags: Optional[List[str]] = None
    description: Optional[str] = None

@dataclass
class UserBehavior:
    """用户行为记录"""
    # 基础信息
    user_id: str
    session_id: str
    timestamp: datetime
    
    # 视频信息
    video_id: str
    video_metadata: VideoMetadata
    
    # 观看行为
    watch_duration: int  # 实际观看时长(秒)
    watch_percentage: float  # 观看百分比 0.0-1.0
    watch_complete: bool  # 是否完整观看
    rewatch_count: int  # 重看次数
    skip_duration: int  # 跳过时长(秒)
    
    # 交互行为
    is_liked: bool  # 是否点赞
    is_favorited: bool  # 是否收藏
    is_shared: bool  # 是否分享
    is_commented: bool  # 是否评论
    is_followed_creator: bool  # 是否关注创作者
    is_blocked_creator: bool  # 是否屏蔽创作者
    is_reported: bool  # 是否举报
    
    # 时间信息
    time_of_day: int  # 小时 0-23
    day_of_week: int  # 星期几 0-6
    is_weekend: bool
    is_holiday: bool
    
    # 上下文信息
    device_type: str  # mobile, tablet, desktop, tv
    os_type: str  # ios, android, windows, macos
    network_type: str  # wifi, 4g, 5g, 3g
    language: str
    timezone: str
    screen_size: str  # small, medium, large, xlarge
    
    # 用户状态
    user_level: int  # 用户等级
    user_vip: bool  # 是否VIP用户
    
    # 会话信息
    session_duration: int  # 会话时长(秒)
    session_video_count: int  # 会话中观看视频数
    session_start_time: datetime
    
    # 推荐相关
    recommendation_source: str  # home, search, following, hashtag, etc.
    recommendation_algorithm: str  # collaborative_filtering, content_based, etc.
    recommendation_position: int  # 推荐位置
    
    # 可选字段
    share_platform: Optional[str] = None  # wechat, weibo, qq等
    share_method: Optional[str] = None  # link, video, screenshot等
    comment_text: Optional[str] = None
    comment_sentiment: Optional[str] = None  # positive, negative, neutral
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    network_speed: Optional[float] = None  # Mbps
    location_country: Optional[str] = None
    location_province: Optional[str] = None
    location_city: Optional[str] = None
    screen_resolution: Optional[str] = None
    user_age_group: Optional[str] = None  # teen, young, adult, senior
    user_gender: Optional[str] = None  # male, female, other
    recommendation_score: Optional[float] = None  # 推荐分数
    engagement_score: Optional[float] = None  # 参与度分数
    retention_score: Optional[float] = None  # 留存分数
    monetization_score: Optional[float] = None  # 商业化分数

@dataclass
class UserBehaviorSequence:
    """用户行为序列"""
    user_id: str
    session_id: str
    behaviors: List[UserBehavior]
    sequence_start_time: datetime
    sequence_end_time: datetime
    total_duration: int  # 总时长(秒)
    total_videos: int  # 总视频数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "behaviors": [self._behavior_to_dict(b) for b in self.behaviors],
            "sequence_start_time": self.sequence_start_time.isoformat(),
            "sequence_end_time": self.sequence_end_time.isoformat(),
            "total_duration": self.total_duration,
            "total_videos": self.total_videos
        }
    
    def _behavior_to_dict(self, behavior: UserBehavior) -> Dict[str, Any]:
        """转换单个行为为字典"""
        return {
            "timestamp": behavior.timestamp.isoformat(),
            "video_id": behavior.video_id,
            "video_metadata": {
                "title": behavior.video_metadata.title,
                "category": behavior.video_metadata.category,
                "duration": behavior.video_metadata.duration,
                "creator_id": behavior.video_metadata.creator_id
            },
            "watch_duration": behavior.watch_duration,
            "watch_percentage": behavior.watch_percentage,
            "watch_complete": behavior.watch_complete,
            "is_liked": behavior.is_liked,
            "is_favorited": behavior.is_favorited,
            "is_shared": behavior.is_shared,
            "is_commented": behavior.is_commented,
            "is_followed_creator": behavior.is_followed_creator,
            "time_of_day": behavior.time_of_day,
            "day_of_week": behavior.day_of_week,
            "device_type": behavior.device_type,
            "network_type": behavior.network_type,
            "recommendation_source": behavior.recommendation_source,
            "recommendation_position": behavior.recommendation_position
        }

class UserBehaviorProcessor:
    """用户行为处理器"""
    
    def __init__(self, max_sequence_length: int = 50):
        self.max_sequence_length = max_sequence_length
    
    def process_behavior_sequence(self, sequence: UserBehaviorSequence) -> Dict[str, Any]:
        """处理用户行为序列，提取特征"""
        
        # 1. 序列截断
        if len(sequence.behaviors) > self.max_sequence_length:
            behaviors = sequence.behaviors[-self.max_sequence_length:]
        else:
            behaviors = sequence.behaviors
        
        # 2. 提取序列特征
        video_ids = [b.video_id for b in behaviors]
        watch_durations = [b.watch_duration for b in behaviors]
        watch_percentages = [b.watch_percentage for b in behaviors]
        
        # 3. 提取交互特征
        like_flags = [1 if b.is_liked else 0 for b in behaviors]
        favorite_flags = [1 if b.is_favorited else 0 for b in behaviors]
        share_flags = [1 if b.is_shared else 0 for b in behaviors]
        comment_flags = [1 if b.is_commented else 0 for b in behaviors]
        follow_flags = [1 if b.is_followed_creator else 0 for b in behaviors]
        
        # 4. 提取时间特征
        time_of_days = [b.time_of_day for b in behaviors]
        day_of_weeks = [b.day_of_week for b in behaviors]
        
        # 5. 提取设备特征
        device_types = [self._encode_device_type(b.device_type) for b in behaviors]
        network_types = [self._encode_network_type(b.network_type) for b in behaviors]
        
        # 6. 提取推荐特征
        recommendation_sources = [self._encode_recommendation_source(b.recommendation_source) for b in behaviors]
        recommendation_positions = [b.recommendation_position for b in behaviors]
        
        # 7. 计算统计特征
        avg_watch_duration = sum(watch_durations) / len(watch_durations) if watch_durations else 0
        avg_watch_percentage = sum(watch_percentages) / len(watch_percentages) if watch_percentages else 0
        like_rate = sum(like_flags) / len(like_flags) if like_flags else 0
        favorite_rate = sum(favorite_flags) / len(favorite_flags) if favorite_flags else 0
        share_rate = sum(share_flags) / len(share_flags) if share_flags else 0
        
        return {
            "sequence_length": len(behaviors),
            "video_ids": video_ids,
            "watch_durations": watch_durations,
            "watch_percentages": watch_percentages,
            "interaction_flags": {
                "likes": like_flags,
                "favorites": favorite_flags,
                "shares": share_flags,
                "comments": comment_flags,
                "follows": follow_flags
            },
            "time_features": {
                "time_of_days": time_of_days,
                "day_of_weeks": day_of_weeks
            },
            "device_features": {
                "device_types": device_types,
                "network_types": network_types
            },
            "recommendation_features": {
                "sources": recommendation_sources,
                "positions": recommendation_positions
            },
            "statistical_features": {
                "avg_watch_duration": avg_watch_duration,
                "avg_watch_percentage": avg_watch_percentage,
                "like_rate": like_rate,
                "favorite_rate": favorite_rate,
                "share_rate": share_rate
            }
        }
    
    def _encode_device_type(self, device_type: str) -> int:
        """编码设备类型"""
        device_mapping = {
            "mobile": 0,
            "tablet": 1,
            "desktop": 2,
            "tv": 3
        }
        return device_mapping.get(device_type, 0)
    
    def _encode_network_type(self, network_type: str) -> int:
        """编码网络类型"""
        network_mapping = {
            "wifi": 0,
            "4g": 1,
            "5g": 2,
            "3g": 3
        }
        return network_mapping.get(network_type, 0)
    
    def _encode_recommendation_source(self, source: str) -> int:
        """编码推荐来源"""
        source_mapping = {
            "home": 0,
            "search": 1,
            "following": 2,
            "hashtag": 3,
            "trending": 4,
            "nearby": 5
        }
        return source_mapping.get(source, 0)

# 示例使用
def create_sample_behavior_sequence() -> UserBehaviorSequence:
    """创建示例用户行为序列"""
    
    # 创建视频元数据
    video_metadata = VideoMetadata(
        video_id="video_001",
        title="搞笑猫咪视频",
        category="entertainment",
        tags=["搞笑", "猫咪", "萌宠"],
        duration=30,
        upload_time=datetime(2024, 1, 10, 10, 0, 0),
        creator_id="creator_123",
        creator_name="搞笑达人",
        creator_followers=50000,
        video_quality="hd",
        language="zh",
        region="china",
        content_rating="general",
        music_genre="pop"
    )
    
    # 创建用户行为
    behavior = UserBehavior(
        user_id="user_12345",
        session_id="session_67890",
        timestamp=datetime(2024, 1, 15, 14, 30, 0),
        video_id="video_001",
        video_metadata=video_metadata,
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
    
    # 创建行为序列
    sequence = UserBehaviorSequence(
        user_id="user_12345",
        session_id="session_67890",
        behaviors=[behavior],
        sequence_start_time=datetime(2024, 1, 15, 14, 25, 0),
        sequence_end_time=datetime(2024, 1, 15, 14, 30, 0),
        total_duration=300,
        total_videos=5
    )
    
    return sequence

if __name__ == "__main__":
    # 创建示例序列
    sequence = create_sample_behavior_sequence()
    
    # 处理序列
    processor = UserBehaviorProcessor()
    features = processor.process_behavior_sequence(sequence)
    
    # 输出结果
    print("用户行为序列特征:")
    print(json.dumps(features, indent=2, ensure_ascii=False))
