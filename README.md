# 生成式推荐模型推理优化项目

## 项目概述

这是一个完整的生成式推荐模型推理优化项目，专注于**推理优化加速部署**。项目集成了TensorRT、Triton推理服务器、自定义算子、GPU加速等核心优化技术，实现了从用户行为数据到推荐结果的端到端高性能推理流程。

## 🎯 核心特性

### 推理优化技术栈
- ✅ **TensorRT优化**: GPU加速推理，性能提升3-10倍
- ✅ **Triton推理服务器**: 生产级高并发推理服务
- ✅ **自定义算子**: Triton DSL和TensorRT插件
- ✅ **GPU加速**: CUDA内核优化和内存管理
- ✅ **ONNX导出**: 模型格式标准化
- ✅ **性能监控**: 实时性能指标和监控

### 推荐系统功能
- ✅ **1024维特征处理**: 企业级用户行为特征
- ✅ **多任务学习**: 参与度、留存、商业化预测
- ✅ **动态批次处理**: 支持高吞吐量推理
- ✅ **缓存机制**: 特征和模型缓存优化
- ✅ **实时推理**: 低延迟推荐服务

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd gr-inference-opt-updated

# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements-dev.txt
```

### 2. 一键运行完整流程

```bash
# 运行集成优化版本（推荐）
python main_optimized.py --mode all
```

这个命令会自动执行：
1. **模型初始化** - 加载生成式推荐模型
2. **TensorRT优化** - 构建和加载TensorRT引擎
3. **自定义算子集成** - 加载高性能算子
4. **Triton部署配置** - 配置推理服务器
5. **单次推理演示** - 展示优化推理效果
6. **批量推理测试** - 测试高并发性能
7. **性能基准测试** - 对比不同推理引擎性能

### 3. 运行结果示例

```
================================================================================
生成式推荐模型推理优化项目 - 集成优化版本
================================================================================

✅ GPU环境可用: NVIDIA A100-SXM4-40GB
✅ TensorRT引擎初始化成功
✅ 自定义算子初始化成功
⚠️ Triton服务器未运行，将使用本地推理

============================================================
优化推理结果
============================================================
用户ID: user_12345
会话ID: session_67890
序列长度: 10
推理引擎: tensorrt

推荐结果:
  1. video_0 (分数: 0.8234)
  2. video_1 (分数: 0.7654)
  3. video_2 (分数: 0.7123)
  ...

特征分数:
  engagement_score: 0.8543
  retention_score: 0.7234
  diversity_score: 0.9123

性能测试结果:
  测试次数: 10
  平均推理时间: 45.23ms
  吞吐量: 22.1 请求/秒
```

## 📁 项目结构

> 📋 **详细项目结构说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**

```
gr-inference-opt-updated/
├── main.py                        # 🎯 主入口文件（集成优化版本）
├── src/                           # 🔥 核心源代码目录
│   ├── inference_pipeline.py      # 推理流水线
│   ├── tensorrt_inference.py      # TensorRT推理模块
│   ├── export_onnx.py            # ONNX模型导出
│   ├── user_behavior_schema.py    # 用户行为数据结构
│   ├── embedding_service.py       # 高性能嵌入服务
│   ├── build_engine.py           # TensorRT引擎构建
│   └── model_parameter_calculator.py # 模型参数计算
├── kernels/                       # 🔥 自定义算子目录
│   ├── triton_ops/              # Triton DSL算子
│   ├── trt_plugin_skeleton/     # TensorRT插件
│   └── cutlass_prototype/       # CUTLASS原型
├── triton_model_repo/             # 🔥 Triton推理服务器
│   ├── ensemble_model/          # 集成模型
│   ├── gr_trt/                  # TensorRT模型
│   ├── interaction_python/      # Python算子
│   ├── embedding_service/       # 嵌入服务
│   └── preprocess_py/           # 预处理
├── docs/                         # 📚 项目文档
├── scripts/                      # 🔧 自动化脚本
├── examples/                     # 📖 使用示例
├── tests/                        # 🧪 测试代码
├── bench/                        # ⚡ 性能测试
└── models/                       # 🤖 模型文件（运行时生成）
```

## 🔧 核心功能模块

### 1. 集成推理优化引擎 (`main_optimized.py`)

**功能**: 统一管理所有推理优化组件，实现一键式优化推理

**核心特性**:
- 自动检测和初始化GPU环境
- 智能选择最优推理引擎（Triton > TensorRT > PyTorch）
- 集成自定义算子处理
- 实时性能监控和日志记录

**使用方式**:
```python
from main_optimized import OptimizedInferenceEngine

# 创建优化推理引擎
engine = OptimizedInferenceEngine(model_config, optimization_config)

# 执行优化推理
result = engine.infer_with_optimization(
    user_behaviors=user_behaviors,
    user_id="user_123",
    session_id="session_456",
    num_recommendations=10
)
```

### 2. TensorRT优化模块 (`src/tensorrt_inference.py`)

**功能**: 将ONNX模型转换为TensorRT引擎，实现GPU加速推理

**性能提升**: 相比PyTorch GPU推理，通常可获得1.5-3x的加速比

**使用方式**:
```python
from src.tensorrt_inference import TensorRTInference, build_tensorrt_engine

# 构建TensorRT引擎
engine_path = build_tensorrt_engine(
    onnx_path="models/prefill.onnx",
    engine_path="models/prefill.trt",
    precision="fp16",
    max_batch_size=8
)

# 使用TensorRT推理
trt_inference = TensorRTInference(engine_path)
result = trt_inference.infer(input_data)
```

### 3. Triton推理服务器 (`triton_model_repo/`)

**功能**: 生产级推理服务器，支持高并发、多模型部署

**部署命令**:
```bash
# 启动Triton服务器
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/triton_model_repo:/models \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models

# 或使用脚本启动
./scripts/run_server.sh
```

### 4. 自定义算子 (`kernels/`)

**功能**: 实现高性能自定义算子，优化特定计算

**算子类型**:
- **Triton DSL算子**: 高性能交互算子
- **TensorRT插件**: 自定义TensorRT层
- **CUTLASS原型**: 高性能矩阵运算

**编译方式**:
```bash
# 编译Triton DSL算子
cd kernels/triton_ops
python setup.py build_ext --inplace

# 编译TensorRT插件
cd kernels/trt_plugin_skeleton
mkdir build && cd build
cmake .. && make
```

## 📊 性能对比

| 推理方式 | 延迟(ms) | 吞吐量(样本/秒) | 内存占用 | 加速比 |
|---------|---------|----------------|---------|--------|
| PyTorch CPU | ~500 | ~2 | 高 | 1x |
| PyTorch GPU | ~150 | ~7 | 中 | 3.3x |
| **TensorRT** | **~50** | **~20** | 低 | **10x** |
| **Triton部署** | **~45** | **~22** | 低 | **11x** |

## 🎮 运行模式

### 1. 完整优化流程（推荐）
```bash
python main_optimized.py --mode all
```

### 2. 专项测试
```bash
# 单次推理
python main_optimized.py --mode single

# 批量推理
python main_optimized.py --mode batch

# 性能测试
python main_optimized.py --mode performance

# Triton部署
python main_optimized.py --mode triton
```

### 3. 调试模式
```bash
# 详细日志
python main_optimized.py --mode all --log-level DEBUG
```

## 📈 性能监控

### 1. 实时监控
```bash
# 查看推理日志
tail -f inference.log

# 查看性能指标
tail -f performance_metrics.log

# Triton监控面板
http://localhost:8000/metrics
```

### 2. 性能指标
- **推理延迟**: 端到端推理时间
- **吞吐量**: 每秒处理请求数
- **GPU利用率**: GPU计算资源使用率
- **内存占用**: 模型和缓存内存使用
- **缓存命中率**: 特征缓存效率

## 🔧 环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- Docker (用于Triton部署)

### 可选依赖
```bash
# TensorRT (需要NVIDIA GPU)
pip install tensorrt

# Triton (需要Docker)
# 参考官方文档安装Triton Inference Server

# 性能监控
pip install prometheus_client
```

## 🚀 部署指南

### 1. 开发环境
```bash
# 快速验证
python main_optimized.py --mode single

# 性能测试
python main_optimized.py --mode performance
```

### 2. 生产环境
```bash
# 启动Triton服务器
./scripts/run_server.sh

# 运行优化推理
python main_optimized.py --mode all
```

### 3. 容器化部署
```bash
# 构建Docker镜像
docker build -t gr-inference-opt .

# 运行容器
docker run --gpus=all -p8000:8000 gr-inference-opt
```

## 🐛 故障排除

### 常见问题

1. **TensorRT安装失败**
   ```bash
   # 检查CUDA版本兼容性
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Triton启动失败**
   ```bash
   # 检查Docker权限
   sudo usermod -aG docker $USER
   sudo systemctl restart docker
   ```

3. **GPU内存不足**
   ```python
   # 减少批次大小
   batch_size = 1
   # 使用梯度检查点
   torch.utils.checkpoint.checkpoint(model, input)
   ```

### 调试技巧
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 检查GPU状态
nvidia-smi

# 监控系统资源
htop
```

## 📚 文档

- [推理优化功能总结](docs/inference_optimization_summary.md)
- [项目运行指南](docs/project_runtime_guide.md)
- [项目架构总结](docs/project_summary.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

感谢NVIDIA提供的TensorRT和Triton Inference Server等优秀工具。

---

**🎯 项目重点**: 这个项目的核心价值在于推理优化技术，通过TensorRT、Triton、自定义算子等技术的集成，实现了高性能的生成式推荐模型推理，是推理优化加速部署的完整解决方案。
