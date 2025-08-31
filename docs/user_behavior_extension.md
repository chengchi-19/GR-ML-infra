# 用户行为序列扩展说明

## 概述

本文档详细说明了项目中用户行为序列的扩展字段定义，包含企业级短视频推荐系统中常见的用户行为字段。

## 1. 扩展字段说明

### 1.1 视频元数据 (VideoMetadata)

```python
@dataclass
class VideoMetadata:
    video_id: str              # 视频ID
    title: str                 # 视频标题
    category: str              # 视频类别
    tags: List[str]            # 视频标签
    duration: int              # 视频时长(秒)
    upload_time: datetime      # 上传时间
    creator_id: str            # 创作者ID
    creator_name: str          # 创作者名称
    creator_followers: int     # 创作者粉丝数
    video_quality: str         # 视频质量 (hd, sd, 4k)
    language: str              # 语言
    region: str                # 地区
    content_rating: str        # 内容评级 (general, mature, kids)
    music_genre: Optional[str] # 音乐类型
    hashtags: Optional[List[str]] # 话题标签
    description: Optional[str] # 视频描述
```

### 1.2 用户行为记录 (UserBehavior)

#### 基础信息
- `user_id`: 用户ID
- `session_id`: 会话ID
- `timestamp`: 行为时间戳

#### 观看行为
- `watch_duration`: 实际观看时长(秒)
- `watch_percentage`: 观看百分比 (0.0-1.0)
- `watch_complete`: 是否完整观看
- `rewatch_count`: 重看次数
- `skip_duration`: 跳过时长(秒)

#### 交互行为
- `is_liked`: 是否点赞
- `is_favorited`: 是否收藏
- `is_shared`: 是否分享
- `is_commented`: 是否评论
- `is_followed_creator`: 是否关注创作者
- `is_blocked_creator`: 是否屏蔽创作者
- `is_reported`: 是否举报

#### 分享行为
- `share_platform`: 分享平台 (wechat, weibo, qq等)
- `share_method`: 分享方式 (link, video, screenshot等)

#### 评论行为
- `comment_text`: 评论内容
- `comment_sentiment`: 评论情感 (positive, negative, neutral)

#### 时间信息
- `time_of_day`: 小时 (0-23)
- `day_of_week`: 星期几 (0-6)
- `is_weekend`: 是否周末
- `is_holiday`: 是否节假日

#### 上下文信息
- `device_type`: 设备类型 (mobile, tablet, desktop, tv)
- `device_model`: 设备型号
- `os_type`: 操作系统 (ios, android, windows, macos)
- `os_version`: 系统版本
- `app_version`: 应用版本
- `network_type`: 网络类型 (wifi, 4g, 5g, 3g)
- `network_speed`: 网络速度 (Mbps)
- `location_country`: 国家
- `location_province`: 省份
- `location_city`: 城市
- `language`: 语言
- `timezone`: 时区
- `screen_size`: 屏幕尺寸 (small, medium, large, xlarge)
- `screen_resolution`: 屏幕分辨率

#### 用户状态
- `user_level`: 用户等级
- `user_vip`: 是否VIP用户
- `user_age_group`: 年龄组 (teen, young, adult, senior)
- `user_gender`: 性别 (male, female, other)

#### 会话信息
- `session_duration`: 会话时长(秒)
- `session_video_count`: 会话中观看视频数
- `session_start_time`: 会话开始时间

#### 推荐相关
- `recommendation_source`: 推荐来源 (home, search, following, hashtag等)
- `recommendation_algorithm`: 推荐算法 (collaborative_filtering, content_based等)
- `recommendation_position`: 推荐位置
- `recommendation_score`: 推荐分数

#### 业务指标
- `engagement_score`: 参与度分数
- `retention_score`: 留存分数
- `monetization_score`: 商业化分数

## 2. 特征处理流程

### 2.1 序列处理
```python
class UserBehaviorProcessor:
    def process_behavior_sequence(self, sequence: UserBehaviorSequence) -> Dict[str, Any]:
        # 1. 序列截断到最大长度
        # 2. 提取序列特征 (视频ID, 观看时长, 观看百分比)
        # 3. 提取交互特征 (点赞, 收藏, 分享, 评论, 关注)
        # 4. 提取时间特征 (时间, 星期)
        # 5. 提取设备特征 (设备类型, 网络类型)
        # 6. 提取推荐特征 (推荐来源, 推荐位置)
        # 7. 计算统计特征 (平均观看时长, 交互率等)
```

### 2.2 特征编码
```python
# 设备类型编码
device_mapping = {
    "mobile": 0,
    "tablet": 1,
    "desktop": 2,
    "tv": 3
}

# 网络类型编码
network_mapping = {
    "wifi": 0,
    "4g": 1,
    "5g": 2,
    "3g": 3
}

# 推荐来源编码
source_mapping = {
    "home": 0,
    "search": 1,
    "following": 2,
    "hashtag": 3,
    "trending": 4,
    "nearby": 5
}
```

## 3. 使用示例

### 3.1 创建用户行为序列
```python
from src.user_behavior_schema import (
    VideoMetadata, UserBehavior, UserBehaviorSequence, UserBehaviorProcessor
)

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
    content_rating="general"
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
    is_liked=True,
    is_favorited=False,
    is_shared=True,
    is_commented=False,
    is_followed_creator=True,
    time_of_day=14,
    day_of_week=3,
    device_type="mobile",
    network_type="wifi",
    recommendation_source="home",
    recommendation_position=3
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
```

### 3.2 处理行为序列
```python
# 处理序列
processor = UserBehaviorProcessor(max_sequence_length=50)
features = processor.process_behavior_sequence(sequence)

# 输出特征
print("序列长度:", features["sequence_length"])
print("视频ID:", features["video_ids"])
print("观看时长:", features["watch_durations"])
print("点赞率:", features["statistical_features"]["like_rate"])
```

## 4. 配置文件更新

### 4.1 Triton模型配置
```protobuf
# triton_model_repo/preprocess_py/config.pbtxt
parameters {
  key: "max_sequence_length"
  value: { string_value: "50" }
}
parameters {
  key: "feature_dim"
  value: { string_value: "32" }  # 扩展特征维度
}
```

### 4.2 模型参数配置
```python
# src/export_onnx.py
model = GenerativeRecommendationModel(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    num_features=32,  # 扩展特征数量
    num_layers=4,
    max_seq_len=512
)
```

## 5. 性能考虑

### 5.1 内存使用
- 每个行为记录约占用 1KB 内存
- 50个行为的序列约占用 50KB 内存
- 建议使用流式处理减少内存占用

### 5.2 计算复杂度
- 特征提取: O(n) 其中n为序列长度
- 交互计算: O(n²) 其中n为特征数量
- 建议使用批处理优化

### 5.3 存储优化
- 使用压缩存储减少磁盘占用
- 使用缓存减少重复计算
- 使用索引加速查询

## 6. 扩展建议

### 6.1 实时特征
- 添加实时用户状态特征
- 添加实时环境特征
- 添加实时业务特征

### 6.2 多模态特征
- 添加视频内容特征
- 添加音频特征
- 添加文本特征

### 6.3 时序特征
- 添加时间衰减特征
- 添加周期性特征
- 添加趋势特征

## 总结

扩展后的用户行为序列包含企业级推荐系统所需的所有关键字段，能够更好地捕捉用户行为模式和偏好，为推荐算法提供更丰富的特征信息。
