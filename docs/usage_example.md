# 短视频推荐系统使用示例

## 1. 输入特征

### 用户序列特征
```python
# 用户观看历史 (最近50个视频)
user_history = ["video_001", "video_015", "video_089", "video_234", "video_567"]
user_tokens = [1, 15, 89, 234, 567]  # 转换为token ID
# 形状: [batch_size=1, seq_len=50]
```

### 用户密集特征
```python
# 用户画像 (16维)
user_features = [
    25.0, 1.0, 0.8, 1.0, 14.0, 3.0, 2.0, 45.5,  # 年龄、性别、位置等
    0.8, 0.3, 0.2, 0.1, 0.4, 15.0, 120.0, 1.0   # 偏好、行为等
]
# 形状: [batch_size=1, num_features=16]
```

## 2. 特征处理流程

### 预处理模型
```python
# 1. 序列截断/填充到50长度
# 2. 特征归一化
# 3. 生成特征索引用于embedding查找
```

### 嵌入查找
```python
# GPU热缓存 + 主机缓存
embeddings = embedding_service.lookup_batch_optimized(user_tokens)
# 形状: [1, 50, 128] - 每个视频的128维嵌入向量
```

### 特征交互
```python
# Triton DSL内核计算pairwise interaction
interactions = interaction_kernel(embeddings)
# 计算50个特征间的两两交互: 50*49/2 = 1225个交互特征
```

## 3. 模型推理

### Prefill阶段
```python
# 处理完整输入序列
logits, feature_scores, hidden_states = model.forward_prefill(
    user_tokens, user_features
)
# logits: [1, 50, 10000] - 词汇表logits
# feature_scores: [1, 1] - 推荐分数
```

### Decode阶段
```python
# 增量生成推荐序列
recommendations = []
for step in range(10):
    logits, scores, hidden_states = model.forward_decode(
        current_token, hidden_states, user_features
    )
    next_video = logits.argmax(dim=-1)
    recommendations.append(next_video)
```

## 4. GPU加速

### TensorRT优化
```python
# FP16精度 + 动态shape + 算子融合
# 性能提升: 3-5倍
```

### 自定义Triton DSL内核
```python
# 向量化交互计算
# 性能提升: 5-10倍
```

## 5. 输出结果

### 推荐分数
```python
recommendations = [
    {"video_id": "video_234", "score": 0.95, "title": "搞笑猫咪视频"},
    {"video_id": "video_567", "score": 0.92, "title": "美食制作教程"},
    {"video_id": "video_123", "score": 0.89, "title": "旅行vlog"},
    # ... 更多推荐
]
```

## 6. 性能监控

### 关键指标
```python
performance = {
    "total_latency": 15.2,        # 总延迟 (ms)
    "inference_latency": 8.5,     # 推理延迟 (ms)
    "throughput": 1800,           # 吞吐量 (req/s)
    "cache_hit_rate": 0.85,       # 缓存命中率
    "gpu_utilization": 85,        # GPU利用率 (%)
    "gpu_memory": 2048            # GPU内存使用 (MB)
}
```

## 7. 完整流程示例

```python
def recommend_videos(user_id):
    # 1. 获取用户数据
    user_history = get_user_history(user_id)
    user_features = get_user_features(user_id)
    
    # 2. 特征处理
    tokens = tokenize(user_history)
    features = normalize(user_features)
    
    # 3. 嵌入查找
    embeddings = embedding_service.lookup(tokens)
    
    # 4. 模型推理
    recommendations = model.infer(embeddings, features)
    
    # 5. 返回结果
    return format_recommendations(recommendations)

# 使用示例
result = recommend_videos("user_12345")
print(f"推荐延迟: {result['latency']:.2f}ms")
print(f"推荐结果: {result['videos'][:5]}")
```

## 总结

- **输入**: 用户观看序列 + 用户特征
- **处理**: 嵌入查找 + 特征交互 + Transformer推理
- **加速**: TensorRT + Triton DSL + GPU缓存
- **输出**: 推荐视频ID + 分数
- **性能**: 15ms延迟, 1800 req/s吞吐量
