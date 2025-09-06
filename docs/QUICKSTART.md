# 快速开始指南

## 🚀 5分钟快速体验

### 第1步: 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd GR-ML-infra

# 安装基础依赖
pip install -r requirements.txt
```

### 第2步: 验证安装
```bash
# 运行集成测试（推荐先运行）
python tests/test_integration.py

# 预期输出: 6/6 tests passed (100%)
```

### 第3步: 运行演示
```bash
# 单次推理演示
python main.py --mode=single

# 预期看到类似输出:
# ✅ 开源框架集成推荐系统初始化完成
# 推理策略: auto
# 推理时间: 25.34ms
# 推荐数量: 10
```

## 📖 详细使用指南

### 🎯 运行模式详解

#### 1. 综合演示 (推荐新用户)
```bash
python main.py --mode=comprehensive
```
**包含内容**:
- 单次推理演示
- 批量推理演示  
- 性能基准测试
- 系统统计信息

#### 2. 单次推理测试
```bash
python main.py --mode=single
```
**适用场景**: 验证基本推理功能

#### 3. 批量推理测试
```bash
python main.py --mode=batch
```
**适用场景**: 测试并发推理能力

#### 4. 性能基准测试
```bash
python main.py --mode=benchmark
```
**适用场景**: 评估不同策略的性能表现

### 🔧 配置参数调整

#### GPU内存优化
```python
# 在main.py中修改配置
config['vllm']['gpu_memory_utilization'] = 0.7  # 降低GPU内存使用
config['tensorrt']['max_batch_size'] = 4        # 减少批次大小
```

#### 推理策略偏好
```python
config['inference_strategy']['auto_selection'] = False  # 禁用自动选择
config['inference_strategy']['fallback_strategy'] = 'hstu'  # 设置回退策略
```

#### 日志级别调整
```bash
python main.py --mode=single --log-level=DEBUG  # 详细日志
python main.py --mode=single --log-level=ERROR  # 仅错误日志
```

## 🧪 测试与验证

### 运行完整测试套件
```bash
# 集成测试
python tests/test_integration.py

# 交互测试
python tests/test_interaction.py

# 用户行为测试
python tests/test_user_behavior.py

# 预填充解码测试
python tests/test_prefill_decode.py
```

### 性能基准测试
```bash
# 快速性能测试
python main.py --mode=benchmark

# 查看详细结果
cat benchmark_results_*.json
```

## 📊 结果解读

### 单次推理结果
```
推理策略: auto                    # 自动选择的策略
推理时间: 25.34ms                # 端到端延迟
推荐数量: 10                     # 生成的推荐数
引擎类型: vllm                   # 实际使用的推理引擎
```

### 批量推理统计
```
批量推理完成:
  总耗时: 156.78ms              # 8个请求总耗时
  平均耗时: 19.60ms/用户         # 平均单用户耗时
  吞吐量: 51.02 用户/秒          # 系统吞吐量
```

### 性能基准对比
```
策略: VLLM
  平均延迟: 18.45ms             # 平均推理时间
  P95延迟: 31.20ms              # 95%请求的延迟
  吞吐量: 54.19 RPS             # 每秒请求数
  成功率: 100.00%               # 推理成功率
```

## 🔍 监控与调试

### 实时日志查看
```bash
# 推理日志
tail -f opensoure_inference.log

# 筛选错误日志
grep "ERROR" opensoure_inference.log

# 筛选性能日志
grep "推理时间" opensoure_inference.log
```

### 系统状态检查
```bash
# 检查框架可用性
python -c "
from integrations.framework_controller import create_integrated_controller
controller = create_integrated_controller({})
print('框架可用性:', controller.framework_availability)
"
```

### GPU状态监控
```bash
# 实时GPU监控
watch -n 1 nvidia-smi

# 内存使用情况
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🚨 常见问题解决

### 问题1: 框架导入失败
```bash
# 症状: ImportError: No module named 'vllm'
# 解决: 项目有智能回退机制，可以正常运行
pip install vllm  # 可选安装
```

### 问题2: GPU内存不足
```bash
# 症状: CUDA out of memory
# 解决: 调整配置参数
```
修改main.py中的配置:
```python
config['vllm']['gpu_memory_utilization'] = 0.6
config['tensorrt']['max_batch_size'] = 2
```

### 问题3: 推理速度慢
```bash
# 症状: 推理时间超过100ms
# 解决: 启用详细监控查看瓶颈
python main.py --mode=single --log-level=DEBUG
```

### 问题4: 测试失败
```bash
# 症状: test_integration.py有失败项
# 解决: 查看具体失败原因
python tests/test_integration.py -v
```

## 🎯 下一步

### 开发环境搭建
1. 安装完整开源框架依赖
2. 配置GPU开发环境
3. 运行性能基准测试

### 生产环境部署
1. 评估硬件资源需求
2. 配置负载均衡
3. 设置监控告警

### 自定义开发
1. 参考 [架构文档](ARCHITECTURE.md)
2. 添加新的推理策略
3. 集成其他开源框架

---

**💡 提示**: 项目设计了完善的回退机制，即使部分开源框架未安装也可以正常运行。建议先运行基础演示，再逐步安装完整依赖。