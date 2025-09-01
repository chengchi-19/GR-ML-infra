# 项目运行指南

## 概述

本文档详细介绍了生成式推荐模型推理优化项目的运行方式，包括环境配置、模型部署、性能测试等各个环节。项目集成了MTGR生成式推荐模型、VLLM推理优化框架、TensorRT加速等核心技术。

## 🚀 完整推理优化流程

```
用户请求 → 特征提取 → MTGR模型(开源) → ONNX导出 → TensorRT优化 → VLLM推理优化 → 结果输出
```

### 优化策略说明

| 策略 | 描述 | 适用场景 | 性能提升 |
|------|------|----------|----------|
| **auto** | 自动选择最佳策略 | 生产环境 | 20-25x |
| **vllm** | VLLM推理优化 | 高并发、长序列 | 15-20x |
| **tensorrt** | TensorRT GPU加速 | 单次推理 | 10-15x |
| **baseline** | 基础MTGR推理 | 开发调试 | 3-5x |

## 🔧 环境配置

### 1. 基础环境要求

```bash
# 系统要求
- Ubuntu 20.04+ / CentOS 8+
- Python 3.8+
- CUDA 11.8+ (推荐)
- Docker (用于Triton部署)

# GPU要求
- NVIDIA GPU (RTX 3090+ / A100+)
- 显存: 24GB+ (推荐)
- 驱动版本: 470+
```

### 2. Python环境配置

```bash
# 创建虚拟环境
python -m venv gr-inference-env
source gr-inference-env/bin/activate  # Linux/Mac
# 或
gr-inference-env\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip

# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements-dev.txt
```

### 3. GPU环境验证

```bash
# 检查CUDA安装
nvidia-smi
nvcc --version

# 检查PyTorch GPU支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 检查TensorRT
python -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

## 🎯 快速开始

### 1. 一键运行完整流程

```bash
# 运行完整优化流程（推荐）
python main.py --mode all

# 查看帮助
python main.py --help
```

### 2. 分步骤运行

```bash
# 1. 测试MTGR模型
python main.py --mode mtgr

# 2. 测试VLLM优化
python main.py --mode vllm

# 3. 测试TensorRT优化
python main.py --mode tensorrt

# 4. 性能基准测试
python main.py --mode performance

# 5. Triton部署测试
python main.py --mode triton
```

## 📊 运行模式详解

### 1. 完整优化流程模式 (`--mode all`)

**功能**: 执行完整的推理优化流程，包括模型加载、优化、测试等所有步骤

**执行步骤**:
1. **环境检查**: 验证GPU、CUDA、依赖库
2. **MTGR模型加载**: 加载开源MTGR模型或自实现
3. **ONNX导出**: 将模型导出为ONNX格式
4. **TensorRT优化**: 构建TensorRT引擎
5. **VLLM初始化**: 初始化VLLM推理引擎
6. **自定义算子**: 加载高性能算子
7. **推理测试**: 执行单次和批量推理测试
8. **性能基准**: 对比不同优化策略性能
9. **结果输出**: 生成性能报告和优化建议

**输出示例**:
```
================================================================================
生成式推荐模型推理优化项目 - 完整流程
================================================================================

✅ 环境检查通过
  GPU: NVIDIA A100-SXM4-40GB
  CUDA: 11.8
  PyTorch: 2.0.0
  TensorRT: 8.6.1

✅ MTGR模型加载成功
  模型来源: huggingface:meituan/mtgr-large
  参数量: 8,123,456,789
  模型类型: MTGRForCausalLM

✅ ONNX导出完成
  文件路径: models/mtgr_model.onnx
  文件大小: 15.2GB
  导出时间: 45.2s

✅ TensorRT引擎构建成功
  文件路径: models/mtgr_model.trt
  精度: FP16
  最大批次: 8
  构建时间: 120.5s

✅ VLLM引擎初始化成功
  模型路径: mtgr_model
  张量并行: 1
  GPU内存利用率: 90%

✅ 自定义算子加载成功
  Triton DSL算子: 3个
  TensorRT插件: 2个

============================================================
推理测试结果
============================================================

单次推理测试:
  延迟: 25.3ms (VLLM优化)
  推荐数量: 5
  推荐质量: 0.854

批量推理测试:
  批次大小: 8
  平均延迟: 45.7ms
  吞吐量: 175.1 请求/秒

性能基准对比:
  Baseline (PyTorch): 150ms, 6.7 req/s
  TensorRT: 50ms, 20.0 req/s
  VLLM: 25ms, 40.0 req/s
  完整优化: 20ms, 50.0 req/s

✅ 完整流程执行成功！
```

### 2. MTGR模型测试模式 (`--mode mtgr`)

**功能**: 专门测试MTGR模型的加载和基本推理功能

**测试内容**:
- 开源MTGR模型加载
- 自实现MTGR模型加载
- 模型前向传播测试
- 参数量统计
- 内存使用分析

### 3. VLLM优化测试模式 (`--mode vllm`)

**功能**: 测试VLLM推理优化框架的性能

**测试内容**:
- VLLM引擎初始化
- PagedAttention性能测试
- Continuous Batching测试
- KV Cache优化效果
- 内存使用优化

### 4. TensorRT优化测试模式 (`--mode tensorrt`)

**功能**: 测试TensorRT GPU加速效果

**测试内容**:
- ONNX模型导出
- TensorRT引擎构建
- FP16/INT8量化测试
- 不同批次大小性能
- GPU内存使用分析

### 5. 性能基准测试模式 (`--mode performance`)

**功能**: 全面对比不同优化策略的性能

**测试内容**:
- 延迟对比测试
- 吞吐量测试
- 内存使用对比
- GPU利用率分析
- 缓存效果测试

### 6. Triton部署测试模式 (`--mode triton`)

**功能**: 测试Triton推理服务器的部署和性能

**测试内容**:
- Triton服务器启动
- 模型部署验证
- 并发请求测试
- 负载均衡测试
- 监控指标收集

## 🔍 详细配置选项

### 1. 模型配置

```bash
# 指定MTGR模型路径
python main.py --mode all --model-path meituan/mtgr-large

# 使用自实现模型
python main.py --mode all --use-custom-model

# 指定模型配置
python main.py --mode all --model-config configs/mtgr_large.json
```

### 2. 优化配置

```bash
# 启用特定优化策略
python main.py --mode all --enable-tensorrt --enable-vllm

# 指定TensorRT精度
python main.py --mode all --tensorrt-precision fp16

# 指定VLLM配置
python main.py --mode all --vllm-config configs/vllm_optimized.json
```

### 3. 性能测试配置

```bash
# 指定测试参数
python main.py --mode performance --batch-size 8 --num-requests 1000

# 指定测试时长
python main.py --mode performance --test-duration 300

# 启用详细监控
python main.py --mode performance --enable-monitoring
```

### 4. 日志配置

```bash
# 设置日志级别
python main.py --mode all --log-level DEBUG

# 输出到文件
python main.py --mode all --log-file inference.log

# 启用性能日志
python main.py --mode all --enable-performance-log
```

## 📈 性能监控

### 1. 实时监控

```bash
# 查看推理日志
tail -f logs/inference.log

# 查看性能指标
tail -f logs/performance.log

# 查看错误日志
tail -f logs/error.log
```

### 2. 监控指标

| 指标 | 描述 | 单位 | 目标值 |
|------|------|------|--------|
| **推理延迟** | 端到端推理时间 | ms | < 50ms |
| **吞吐量** | 每秒处理请求数 | req/s | > 20 req/s |
| **GPU利用率** | GPU计算资源使用率 | % | 80-95% |
| **GPU内存使用** | GPU显存使用量 | GB | < 80% |
| **缓存命中率** | 特征缓存命中率 | % | > 90% |
| **错误率** | 推理错误率 | % | < 1% |

### 3. 性能分析工具

```bash
# GPU监控
nvidia-smi -l 1

# 系统资源监控
htop

# 网络监控
iftop

# 磁盘I/O监控
iotop
```

## 🐛 故障排除

### 1. 常见问题及解决方案

#### MTGR模型加载失败

**问题**: 无法加载开源MTGR模型
```bash
Error: Could not load model 'meituan/mtgr-large'
```

**解决方案**:
```bash
# 检查网络连接
ping huggingface.co

# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 离线加载
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meituan/mtgr-large', local_files_only=True)"
```

#### VLLM安装失败

**问题**: VLLM安装过程中出错
```bash
Error: Failed to build wheel for vllm
```

**解决方案**:
```bash
# 从源码安装
pip install git+https://github.com/vllm-ai/vllm.git

# 或使用conda
conda install -c conda-forge vllm

# 检查CUDA版本兼容性
python -c "import torch; print(torch.version.cuda)"
```

#### TensorRT构建失败

**问题**: TensorRT引擎构建失败
```bash
Error: Failed to build TensorRT engine
```

**解决方案**:
```bash
# 检查TensorRT版本
python -c "import tensorrt as trt; print(trt.__version__)"

# 检查ONNX模型
python -c "import onnx; model = onnx.load('models/mtgr_model.onnx'); onnx.checker.check_model(model)"

# 减少批次大小
python main.py --mode tensorrt --max-batch-size 1
```

#### GPU内存不足

**问题**: CUDA out of memory
```bash
RuntimeError: CUDA out of memory
```

**解决方案**:
```python
# 减少批次大小
batch_size = 1

# 启用梯度检查点
torch.utils.checkpoint.checkpoint(model, input)

# 使用VLLM内存优化
vllm_config['gpu_memory_utilization'] = 0.8

# 启用FP16
model.half()
```

### 2. 调试技巧

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

# 检查GPU状态
nvidia-smi

# 监控系统资源
htop

# 测试MTGR模型
python src/mtgr_integration.py

# 测试VLLM引擎
python src/vllm_engine.py
```

## 📚 相关文档

- [MTGR和VLLM集成指南](../MTGR_VLLM_INTEGRATION.md)
- [推理优化功能总结](inference_optimization_summary.md)
- [项目架构总结](project_summary.md)
- [模型架构说明](model_architecture.md)

## 🤝 技术支持

如果遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查项目日志文件
3. 提交Issue到项目仓库
4. 联系项目维护者

---

**注意**: 本文档会随着项目更新而更新，请确保使用最新版本。
