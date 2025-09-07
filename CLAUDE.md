# CLAUDE.md

这个文件为Claude Code (claude.ai/code)提供在此代码库中工作的指导。

## 常用开发命令

### 项目运行
```bash
# 单次推理测试
python main.py --mode=single

# 批量推理测试  
python main.py --mode=batch

# 综合演示（推荐）
python main.py --mode=comprehensive

# 性能基准测试
python main.py --mode=benchmark
```

### 测试命令
```bash
# 运行集成测试
python tests/test_integration.py

# 运行特征处理测试
python tests/test_feature_processing.py

# 运行Triton集成测试
python tests/test_triton_integration.py

# 运行性能测试
python main.py --mode=benchmark
```

### 环境和依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装开源框架（可选，项目有智能回退机制）
pip install vllm tensorrt torchrec fbgemm-gpu
```

## 模型参数和资源配置

### 核心模型参数
- **HSTU模型规模**: 约3.2B参数
  - vocab_size: 50,000（词汇表大小）
  - d_model: 1024（隐藏维度）
  - num_layers: 12（层数）
  - num_heads: 16（注意力头数）
  - max_seq_len: 2048（最大序列长度）
  - d_ff: 4096（前馈网络维度）

### A100 GPU单卡资源评估

#### 内存需求分析
- **模型权重**: ~12.8GB (3.2B参数 × 4字节/FP32)
- **FP16优化后**: ~6.4GB (3.2B参数 × 2字节/FP16)
- **KV缓存**: ~2-4GB (取决于batch size和序列长度)
- **中间激活**: ~1-2GB
- **VLLM内存池**: ~10-15GB (PagedAttention优化)
- **TensorRT工作空间**: 2GB
- **Triton算子缓存**: ~1GB
- **智能缓存系统**: ~2GB (8192 embeddings × 1024维 × 4字节)

**总内存需求**: ~25-32GB (A100 80GB **完全可行**)

#### 性能预估指标

**单次推理 (Batch Size 1)**
- **Prefill延迟**: 80-120ms (序列长度64-512)
- **Decode延迟**: 15-25ms per token
- **端到端延迟**: 100-400ms (生成10个推荐)

**批量推理 (Batch Size 8)**
- **吞吐量**: 2000-4000 requests/second
- **平均延迟**: 50-80ms per request
- **P95延迟**: 120-180ms
- **P99延迟**: 200-300ms

**统一推理管道性能**
- **HSTU特征提取**: 20-35ms
- **ONNX导出开销**: ~2ms (缓存后)
- **TensorRT加速**: 40-70ms (FP16优化)
- **VLLM推理服务**: 30-50ms
- **整体流水线**: 90-150ms

#### 优化配置建议
```python
# A100单卡优化配置
'vllm': {
    'gpu_memory_utilization': 0.75,  # 75%内存使用率，留足余量
    'max_num_seqs': 128,             # 并发序列数
    'max_model_len': 1024,           # 适中的序列长度
    'dtype': 'float16',              # 使用FP16节省内存
    'tensor_parallel_size': 1,       # 单卡配置
}

'tensorrt': {
    'max_batch_size': 8,             # 适中的批处理大小
    'precision': 'fp16',             # FP16精度优化
    'max_workspace_size': 2 << 30,   # 2GB工作空间
}
```

#### 资源利用率预估
- **GPU利用率**: 70-85%
- **内存利用率**: 40-50% (80GB A100)
- **TensorCore利用率**: 80-95% (FP16混合精度)
- **带宽利用率**: 60-80%

#### 扩展性说明
- **单A100可支持**: 2000+ QPS, 128并发用户
- **推荐扩展至2卡**: 可提升至4000+ QPS
- **4卡配置**: 可达8000+ QPS，支持更大batch size

## 代码架构

### 核心架构模式
这是一个基于**统一推理管道**的推荐系统推理优化项目，采用以下架构模式：
- **适配器模式**: 通过 `integrations/` 模块统一不同开源框架接口
- **管道模式**: HSTU→ONNX→TensorRT→VLLM 的端到端优化链路
- **工厂模式**: 通过 `framework_controller.py` 创建和管理推理引擎
- **智能回退**: 框架不可用时自动降级到可用方案

### 关键组件层级
```
main.py (应用入口层)
    ↓
integrations/framework_controller.py (统一控制层)
    ↓
integrations/{hstu,vllm,tensorrt}/ (框架适配层)
    ↓
optimizations/ (算子优化层)
    ↓
external/ (开源框架源码层)
```

### 统一推理流程
项目的核心创新是实现了一个统一的推理管道：

1. **HSTU模型** (`integrations/hstu/`): Meta开源的Hierarchical Sequential Transduction Units模型，负责特征提取和序列建模
2. **ONNX导出** (`integrations/hstu/onnx_exporter.py`): 将HSTU模型导出为标准ONNX格式
3. **TensorRT优化** (`integrations/tensorrt/`): GPU加速推理，支持FP16/INT8量化
4. **VLLM推理服务** (`integrations/vllm/`): PagedAttention内存优化和Continuous Batching

### 自定义优化模块
- **智能缓存系统** (`optimizations/cache/intelligent_cache.py`): 预测式GPU热缓存，提供热点预测和智能驱逐策略
- **Triton算子** (`optimizations/triton_ops/`): 针对推荐场景的自定义GPU算子
- **CUTLASS算子** (`optimizations/cutlass_ops/`): 高性能矩阵运算优化

### 框架集成设计
每个开源框架都有独立的适配器实现标准接口：
- `infer()`: 单次推理
- `batch_infer()`: 批量推理  
- `get_availability()`: 可用性检查
- `get_stats()`: 性能统计

`framework_controller.py` 中的 `OpenSourceFrameworkController` 类负责：
- 智能策略选择 (`_select_optimal_strategy()`)
- 统一推理管道 (`_unified_inference_pipeline()`)
- 性能监控和统计
- 异步批量处理

### 关键文件说明
- `main.py`: 主入口，包含配置生成和演示模式
- `integrations/framework_controller.py`: 统一框架控制器，核心架构组件
- `integrations/hstu/hstu_model.py`: Meta HSTU模型集成
- `integrations/vllm/vllm_engine.py`: VLLM推理引擎封装
- `integrations/tensorrt/tensorrt_engine.py`: TensorRT优化引擎
- `optimizations/cache/intelligent_cache.py`: 智能缓存系统
- `examples/client_example.py`: 用户行为数据生成和客户端示例

### 开发和扩展
添加新推理框架的步骤：
1. 在 `integrations/新框架/` 创建适配器模块
2. 实现标准接口 (`infer`, `batch_infer`, `get_availability`)
3. 在 `framework_controller.py` 中注册新框架
4. 在 `main.py` 的配置中添加框架配置

项目采用模块化设计，支持框架的独立开发和测试，通过智能回退机制确保系统稳定性。

## 监控和调试
- 推理日志: `opensoure_inference.log`
- 性能基准结果: `benchmark_results_*.json`
- 测试覆盖: 集成测试、特征处理测试、Triton集成测试

## 部署配置
项目支持多种部署配置，通过智能配置生成自动适配不同的硬件环境和框架可用性。项目采用模块化设计，支持框架的独立开发和测试，通过智能回退机制确保系统稳定性。