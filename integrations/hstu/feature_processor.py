#!/usr/bin/env python3
"""
HSTU特征处理器

专门处理用户行为序列，生成符合HSTU模型输入要求的特征
"""

import sys
import os
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


class HSTUFeatureProcessor:
    """
    HSTU模型专用特征处理器
    
    处理用户行为序列，生成符合HSTU模型要求的输入特征
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocab_size = config.get('vocab_size', 50000)
        self.max_seq_len = config.get('max_seq_len', 2048)
        self.embedding_dim = config.get('d_model', 1024)
        self.num_dense_features = config.get('num_dense_features', 128)
        
        # 初始化词汇表和嵌入
        self._initialize_vocabulary()
        self._initialize_embeddings()
        
        logger.info(f"✅ HSTU特征处理器初始化完成")
        logger.info(f"  词汇表大小: {self.vocab_size}")
        logger.info(f"  最大序列长度: {self.max_seq_len}")
        logger.info(f"  嵌入维度: {self.embedding_dim}")
    
    def _initialize_vocabulary(self):
        """初始化词汇表"""
        # 视频ID映射
        self.video_id_to_token = {}
        self.token_to_video_id = {}
        self.next_token_id = 1  # 0保留给padding
        
        # 特殊token
        self.special_tokens = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
            'SEP': 3,
            'MASK': 4
        }
        
        self.next_token_id = max(self.special_tokens.values()) + 1
    
    def _initialize_embeddings(self):
        """初始化嵌入层"""
        # 视频嵌入
        self.video_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        
        # 位置嵌入
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=self.embedding_dim
        )
        
        # 时间嵌入
        self.time_embedding = torch.nn.Embedding(
            num_embeddings=100,  # 100个时间桶
            embedding_dim=self.embedding_dim
        )
        
        # 密集特征处理
        self.dense_feature_processor = torch.nn.Sequential(
            torch.nn.Linear(self.num_dense_features, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )
    
    def process_user_behaviors(self, user_behaviors: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理用户行为序列
        
        Args:
            user_behaviors: 用户行为列表，每个行为包含:
                - video_id: 视频ID
                - watch_duration: 观看时长
                - watch_percentage: 观看百分比
                - is_liked: 是否点赞
                - is_shared: 是否分享
                - timestamp: 时间戳
                - category: 类别
                - engagement_score: 参与度分数
        
        Returns:
            处理后的特征字典
        """
        try:
            if not user_behaviors:
                return self._create_empty_features()
            
            # 按时间戳排序
            sorted_behaviors = sorted(user_behaviors, key=lambda x: x.get('timestamp', 0))
            
            # 限制序列长度
            if len(sorted_behaviors) > self.max_seq_len:
                sorted_behaviors = sorted_behaviors[-self.max_seq_len:]
            
            # 提取特征
            features = self._extract_features(sorted_behaviors)
            
            # 创建HSTU输入
            hstu_features = self._create_hstu_input(features)
            
            return hstu_features
            
        except Exception as e:
            logger.error(f"用户行为处理失败: {e}")
            return self._create_empty_features()
    
    def _extract_features(self, behaviors: List[Dict[str, Any]]) -> Dict[str, List]:
        """提取特征"""
        features = {
            'video_ids': [],
            'watch_durations': [],
            'watch_percentages': [],
            'is_liked': [],
            'is_shared': [],
            'timestamps': [],
            'categories': [],
            'engagement_scores': [],
            'interaction_types': []
        }
        
        for behavior in behaviors:
            # 视频ID
            video_id = str(behavior.get('video_id', 'unknown'))
            token_id = self._get_video_token(video_id)
            features['video_ids'].append(token_id)
            
            # 数值特征
            features['watch_durations'].append(float(behavior.get('watch_duration', 0.0)))
            features['watch_percentages'].append(float(behavior.get('watch_percentage', 0.0)))
            features['is_liked'].append(float(behavior.get('is_liked', False)))
            features['is_shared'].append(float(behavior.get('is_shared', False)))
            features['engagement_scores'].append(float(behavior.get('engagement_score', 0.5)))
            
            # 时间特征
            timestamp = behavior.get('timestamp', 0)
            features['timestamps'].append(timestamp)
            
            # 类别特征
            category = str(behavior.get('category', 'unknown'))
            category_id = self._get_category_token(category)
            features['categories'].append(category_id)
            
            # 交互类型
            interaction_type = self._get_interaction_type(behavior)
            features['interaction_types'].append(interaction_type)
        
        return features
    
    def _get_video_token(self, video_id: str) -> int:
        """获取视频token"""
        if video_id in self.video_id_to_token:
            return self.video_id_to_token[video_id]
        else:
            if self.next_token_id < self.vocab_size:
                token_id = self.next_token_id
                self.video_id_to_token[video_id] = token_id
                self.token_to_video_id[token_id] = video_id
                self.next_token_id += 1
                return token_id
            else:
                return self.special_tokens['UNK']
    
    def _get_category_token(self, category: str) -> int:
        """获取类别token"""
        # 简单的类别映射
        category_map = {
            'tech': 1000,
            'music': 1001,
            'sports': 1002,
            'education': 1003,
            'entertainment': 1004,
            'news': 1005,
            'unknown': self.special_tokens['UNK']
        }
        return category_map.get(category.lower(), self.special_tokens['UNK'])
    
    def _get_interaction_type(self, behavior: Dict[str, Any]) -> int:
        """获取交互类型"""
        # 定义交互类型
        if behavior.get('is_liked', False):
            return 1  # 点赞
        elif behavior.get('is_shared', False):
            return 2  # 分享
        elif behavior.get('watch_percentage', 0) > 0.8:
            return 3  # 完整观看
        elif behavior.get('watch_percentage', 0) > 0.5:
            return 4  # 部分观看
        else:
            return 5  # 快速浏览
    
    def _create_hstu_input(self, features: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """创建HSTU模型输入"""
        seq_len = len(features['video_ids'])
        
        # 视频ID序列
        input_ids = torch.tensor([features['video_ids']], dtype=torch.long)
        
        # 注意力掩码
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        
        # 位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        # 时间特征
        timestamps = torch.tensor([features['timestamps']], dtype=torch.float32)
        
        # 归一化数值特征
        watch_durations = torch.tensor([features['watch_durations']], dtype=torch.float32)
        watch_percentages = torch.tensor([features['watch_percentages']], dtype=torch.float32)
        is_liked = torch.tensor([features['is_liked']], dtype=torch.float32)
        is_shared = torch.tensor([features['is_shared']], dtype=torch.float32)
        engagement_scores = torch.tensor([features['engagement_scores']], dtype=torch.float32)
        
        # 组合密集特征
        dense_features = torch.stack([
            watch_durations,
            watch_percentages,
            is_liked,
            is_shared,
            engagement_scores
        ], dim=-1)
        
        # 填充到指定维度
        if dense_features.shape[-1] < self.num_dense_features:
            padding = torch.zeros(1, seq_len, self.num_dense_features - dense_features.shape[-1])
            dense_features = torch.cat([dense_features, padding], dim=-1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'dense_features': dense_features,
            'timestamps': timestamps,
            'sequence_length': torch.tensor([seq_len], dtype=torch.long)
        }
    
    def _create_empty_features(self) -> Dict[str, torch.Tensor]:
        """创建空特征"""
        return {
            'input_ids': torch.zeros(1, 1, dtype=torch.long),
            'attention_mask': torch.zeros(1, 1, dtype=torch.long),
            'position_ids': torch.zeros(1, 1, dtype=torch.long),
            'dense_features': torch.zeros(1, 1, self.num_dense_features),
            'timestamps': torch.zeros(1, 1),
            'sequence_length': torch.tensor([0], dtype=torch.long)
        }
    
    def get_feature_stats(self, user_behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取特征统计信息"""
        if not user_behaviors:
            return {}
        
        features = self._extract_features(user_behaviors)
        
        return {
            'sequence_length': len(features['video_ids']),
            'unique_videos': len(set(features['video_ids'])),
            'avg_watch_duration': np.mean(features['watch_durations']),
            'avg_watch_percentage': np.mean(features['watch_percentages']),
            'like_rate': np.mean(features['is_liked']),
            'share_rate': np.mean(features['is_shared']),
            'avg_engagement_score': np.mean(features['engagement_scores'])
        }
    
    def export_vocabulary(self, file_path: str):
        """导出词汇表"""
        vocab_data = {
            'video_id_to_token': self.video_id_to_token,
            'token_to_video_id': self.token_to_video_id,
            'special_tokens': self.special_tokens,
            'next_token_id': self.next_token_id
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocabulary(self, file_path: str):
        """加载词汇表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.video_id_to_token = vocab_data['video_id_to_token']
            self.token_to_video_id = vocab_data['token_to_video_id']
            self.special_tokens = vocab_data['special_tokens']
            self.next_token_id = vocab_data['next_token_id']
            
            logger.info(f"✅ 词汇表加载成功: {len(self.video_id_to_token)} 个视频")
        except Exception as e:
            logger.error(f"词汇表加载失败: {e}")


def create_hstu_feature_processor(config: Dict[str, Any]) -> HSTUFeatureProcessor:
    """创建HSTU特征处理器"""
    return HSTUFeatureProcessor(config)


if __name__ == "__main__":
    # 测试特征处理器
    config = {
        'vocab_size': 50000,
        'd_model': 1024,
        'max_seq_len': 100,
        'num_dense_features': 128
    }
    
    processor = create_hstu_feature_processor(config)
    
    # 测试数据
    test_behaviors = [
        {
            'video_id': 'video_12345',
            'watch_duration': 120.5,
            'watch_percentage': 0.85,
            'is_liked': True,
            'is_shared': False,
            'timestamp': 1234567890,
            'category': 'tech',
            'engagement_score': 0.9
        },
        {
            'video_id': 'video_67890',
            'watch_duration': 45.2,
            'watch_percentage': 0.3,
            'is_liked': False,
            'is_shared': True,
            'timestamp': 1234567891,
            'category': 'music',
            'engagement_score': 0.6
        }
    ]
    
    # 处理特征
    features = processor.process_user_behaviors(test_behaviors)
    
    print("HSTU特征处理结果:")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # 获取统计信息
    stats = processor.get_feature_stats(test_behaviors)
    print(f"特征统计: {stats}")