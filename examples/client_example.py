#!/usr/bin/env python3
"""
客户端示例 - 演示如何使用用户行为序列推理流水线
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
import random

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference_pipeline import UserBehaviorInferencePipeline

def create_realistic_user_behaviors(user_id: str, num_behaviors: int = 10) -> list:
    """创建真实的用户行为数据"""
    
    behaviors = []
    base_time = datetime.now() - timedelta(hours=2)
    
    # 视频类别和标签
    video_categories = ["entertainment", "education", "sports", "music", "gaming", "food", "travel"]
    video_tags = {
        "entertainment": ["搞笑", "娱乐", "综艺", "明星"],
        "education": ["教程", "学习", "知识", "技能"],
        "sports": ["运动", "健身", "比赛", "训练"],
        "music": ["音乐", "歌曲", "舞蹈", "乐器"],
        "gaming": ["游戏", "电竞", "攻略", "直播"],
        "food": ["美食", "烹饪", "探店", "食谱"],
        "travel": ["旅行", "风景", "攻略", "vlog"]
    }
    
    # 设备类型
    device_types = ["mobile", "tablet", "desktop"]
    network_types = ["wifi", "4g", "5g"]
    
    for i in range(num_behaviors):
        # 随机选择视频类别
        category = random.choice(video_categories)
        tags = random.sample(video_tags[category], random.randint(1, 3))
        
        # 生成时间戳
        timestamp = base_time + timedelta(minutes=i * random.randint(5, 15))
        
        # 生成观看行为
        video_duration = random.randint(15, 120)  # 15秒到2分钟
        watch_duration = random.randint(5, video_duration)
        watch_percentage = watch_duration / video_duration
        
        # 生成交互行为
        is_liked = random.random() < 0.6  # 60%概率点赞
        is_favorited = random.random() < 0.2  # 20%概率收藏
        is_shared = random.random() < 0.1  # 10%概率分享
        is_commented = random.random() < 0.05  # 5%概率评论
        is_followed_creator = random.random() < 0.15  # 15%概率关注创作者
        
        behavior = {
            "video_id": f"video_{random.randint(1000, 9999)}",
            "timestamp": timestamp.isoformat(),
            "watch_duration": watch_duration,
            "watch_percentage": watch_percentage,
            "watch_complete": watch_percentage > 0.8,
            "rewatch_count": random.randint(0, 3),
            "skip_duration": random.randint(0, 10),
            "is_liked": is_liked,
            "is_favorited": is_favorited,
            "is_shared": is_shared,
            "is_commented": is_commented,
            "is_followed_creator": is_followed_creator,
            "is_blocked_creator": False,
            "is_reported": False,
            "share_platform": random.choice(["wechat", "weibo", "qq", None]) if is_shared else None,
            "share_method": random.choice(["link", "video", "screenshot"]) if is_shared else None,
            "time_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_weekend": timestamp.weekday() >= 5,
            "is_holiday": False,
            "device_type": random.choice(device_types),
            "device_model": f"Device_{random.randint(1, 10)}",
            "os_type": random.choice(["ios", "android", "windows", "macos"]),
            "os_version": f"{random.randint(10, 20)}.{random.randint(0, 9)}",
            "app_version": f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "network_type": random.choice(network_types),
            "network_speed": random.uniform(10, 100),
            "location_country": "china",
            "location_province": random.choice(["beijing", "shanghai", "guangdong", "sichuan"]),
            "location_city": random.choice(["beijing", "shanghai", "guangzhou", "chengdu"]),
            "language": "zh",
            "timezone": "UTC+8",
            "screen_size": random.choice(["small", "medium", "large", "xlarge"]),
            "screen_resolution": f"{random.randint(1000, 2000)}x{random.randint(2000, 3000)}",
            "user_level": random.randint(1, 10),
            "user_vip": random.random() < 0.1,  # 10%概率是VIP
            "user_age_group": random.choice(["teen", "young", "adult", "senior"]),
            "user_gender": random.choice(["male", "female", "other"]),
            "session_duration": random.randint(60, 600),
            "session_video_count": random.randint(1, 20),
            "session_start_time": (timestamp - timedelta(minutes=random.randint(10, 60))).isoformat(),
            "recommendation_source": random.choice(["home", "search", "following", "hashtag", "trending", "nearby"]),
            "recommendation_algorithm": random.choice(["collaborative_filtering", "content_based", "deep_learning"]),
            "recommendation_position": random.randint(1, 20),
            "recommendation_score": random.uniform(0.5, 1.0),
            "engagement_score": random.uniform(0.3, 0.9),
            "retention_score": random.uniform(0.4, 0.95),
            "monetization_score": random.uniform(0.1, 0.8)
        }
        
        behaviors.append(behavior)
    
    return behaviors

def create_batch_requests(num_users: int = 5) -> list:
    """创建批量请求"""
    
    requests = []
    for i in range(num_users):
        user_id = f"user_{random.randint(10000, 99999)}"
        session_id = f"session_{random.randint(100000, 999999)}"
        
        # 为每个用户创建不同数量的行为
        num_behaviors = random.randint(5, 20)
        behaviors = create_realistic_user_behaviors(user_id, num_behaviors)
        
        request = {
            "user_id": user_id,
            "session_id": session_id,
            "behaviors": behaviors
        }
        
        requests.append(request)
    
    return requests

def demo_single_user_inference():
    """演示单用户推理"""
    
    print("=" * 60)
    print("单用户推理演示")
    print("=" * 60)
    
    # 创建推理流水线
    pipeline = UserBehaviorInferencePipeline()
    
    # 创建用户行为数据
    user_id = "user_12345"
    session_id = "session_67890"
    behaviors = create_realistic_user_behaviors(user_id, 15)
    
    print(f"用户ID: {user_id}")
    print(f"会话ID: {session_id}")
    print(f"行为数量: {len(behaviors)}")
    
    # 显示前几个行为
    print("\n前3个行为示例:")
    for i, behavior in enumerate(behaviors[:3]):
        print(f"  行为 {i+1}:")
        print(f"    视频ID: {behavior['video_id']}")
        print(f"    观看时长: {behavior['watch_duration']}秒")
        print(f"    观看百分比: {behavior['watch_percentage']:.2f}")
        print(f"    是否点赞: {behavior['is_liked']}")
        print(f"    设备类型: {behavior['device_type']}")
        print(f"    时间: {behavior['timestamp']}")
    
    # 执行推理
    print("\n执行推理...")
    start_time = time.time()
    
    result = pipeline.infer_recommendations(
        user_id=user_id,
        session_id=session_id,
        behaviors=behaviors,
        num_recommendations=10
    )
    
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    # 输出结果
    print(f"\n推理完成，耗时: {processing_time:.2f}ms")
    print(f"序列长度: {result['sequence_length']}")
    print(f"推荐数量: {len(result['recommendations'])}")
    
    print("\n推荐结果:")
    for i, rec in enumerate(result['recommendations'][:5]):
        print(f"  {i+1}. {rec['video_id']} (分数: {rec['score']:.3f})")
    
    print(f"\n特征分数:")
    print(f"  参与度分数: {result['feature_scores']['engagement_score']:.3f}")
    print(f"  留存分数: {result['feature_scores']['retention_score']:.3f}")
    print(f"  多样性分数: {result['feature_scores']['diversity_score']:.3f}")

def demo_batch_inference():
    """演示批量推理"""
    
    print("\n" + "=" * 60)
    print("批量推理演示")
    print("=" * 60)
    
    # 创建推理流水线
    pipeline = UserBehaviorInferencePipeline()
    
    # 创建批量请求
    batch_requests = create_batch_requests(5)
    
    print(f"批量请求数量: {len(batch_requests)}")
    for i, request in enumerate(batch_requests):
        print(f"  用户 {i+1}: {request['user_id']} ({len(request['behaviors'])} 个行为)")
    
    # 执行批量推理
    print("\n执行批量推理...")
    start_time = time.time()
    
    batch_results = pipeline.batch_infer(batch_requests, num_recommendations=8)
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    avg_time = total_time / len(batch_requests)
    
    # 输出结果
    print(f"\n批量推理完成:")
    print(f"  总耗时: {total_time:.2f}ms")
    print(f"  平均耗时: {avg_time:.2f}ms/用户")
    print(f"  吞吐量: {len(batch_requests)/total_time*1000:.2f} 用户/秒")
    
    print("\n各用户推荐结果:")
    for i, result in enumerate(batch_results):
        print(f"  用户 {i+1} ({result['user_id']}):")
        print(f"    序列长度: {result['sequence_length']}")
        print(f"    推荐数量: {len(result['recommendations'])}")
        print(f"    参与度分数: {result['feature_scores']['engagement_score']:.3f}")
        print(f"    前3个推荐: {[rec['video_id'] for rec in result['recommendations'][:3]]}")

def demo_performance_test():
    """性能测试演示"""
    
    print("\n" + "=" * 60)
    print("性能测试演示")
    print("=" * 60)
    
    # 创建推理流水线
    pipeline = UserBehaviorInferencePipeline()
    
    # 创建大量测试数据
    num_users = 100
    batch_requests = create_batch_requests(num_users)
    
    print(f"性能测试: {num_users} 个用户")
    
    # 预热
    print("预热中...")
    warmup_requests = batch_requests[:5]
    pipeline.batch_infer(warmup_requests, num_recommendations=5)
    
    # 性能测试
    print("开始性能测试...")
    start_time = time.time()
    
    results = pipeline.batch_infer(batch_requests, num_recommendations=10)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    successful_requests = len([r for r in results if 'error' not in r])
    failed_requests = len(results) - successful_requests
    
    print(f"\n性能测试结果:")
    print(f"  总请求数: {num_users}")
    print(f"  成功请求: {successful_requests}")
    print(f"  失败请求: {failed_requests}")
    print(f"  成功率: {successful_requests/num_users*100:.1f}%")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均延迟: {total_time/num_users*1000:.2f}ms")
    print(f"  吞吐量: {num_users/total_time:.2f} 请求/秒")
    
    # 计算平均特征分数
    if successful_requests > 0:
        avg_engagement = sum(r['feature_scores']['engagement_score'] for r in results if 'error' not in r) / successful_requests
        avg_retention = sum(r['feature_scores']['retention_score'] for r in results if 'error' not in r) / successful_requests
        avg_diversity = sum(r['feature_scores']['diversity_score'] for r in results if 'error' not in r) / successful_requests
        
        print(f"\n平均特征分数:")
        print(f"  参与度: {avg_engagement:.3f}")
        print(f"  留存: {avg_retention:.3f}")
        print(f"  多样性: {avg_diversity:.3f}")

def main():
    """主函数"""
    
    print("用户行为序列推理流水线客户端示例")
    print("=" * 60)
    
    try:
        # 1. 单用户推理演示
        demo_single_user_inference()
        
        # 2. 批量推理演示
        demo_batch_inference()
        
        # 3. 性能测试演示
        demo_performance_test()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
