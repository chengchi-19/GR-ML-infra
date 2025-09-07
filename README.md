# 🌟 基于生成式推荐模型的推理优化项目

## 📋 项目概述

这是一个基于**开源框架**的生成式推荐模型推理优化项目，集成了Meta开源的HSTU模型、VLLM推理引擎、TensorRT加速引擎、自定义Triton和CUTLASS算子：
- **Meta HSTU** (Hierarchical Sequential Transduction Units) 生成式推荐模型
- **VLLM** (PagedAttention + Continuous Batching) 推理优化框架  
- **TensorRT** GPU推理加速引擎
- **自定义Triton和CUTLASS算子**优化
- **智能GPU热缓存**系统

## 🎯 核心特性

### 🚀 开源框架技术栈
- ✅ **Meta HSTU模型**: Hierarchical Sequential Transduction Units架构
- ✅ **VLLM推理引擎**: PagedAttention内存优化  
- ✅ **TensorRT加速**: GPU推理优化，支持FP16/INT8量化
- ✅ **智能缓存系统**: 预测式GPU热缓存
- ✅ **自定义算子**: Triton DSL和CUTLASS高性能算子

### 🧠 统一推理流程
- 🔄 **HSTU模型**: 负责特征提取和序列建模
- 📤 **ONNX导出**: 将HSTU模型导出为标准ONNX格式
- ⚡ **TensorRT优化**: GPU加速推理，模型优化
- 🎯 **VLLM推理服务**: 内存管理和推理服务优化

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd GR-ML-infra

# 安装依赖
pip install -r requirements.txt

# 安装开源框架（可选，项目支持统一流水线）
pip install vllm tensorrt torchrec fbgemm-gpu
```

### 2. 运行演示

```bash
# 运行综合演示（推荐）
python main.py --mode=comprehensive

# 单次推理测试
python main.py --mode=single

# 批量推理测试
python main.py --mode=batch

# 性能基准测试
python main.py --mode=benchmark
```

### 3. 验证集成效果

```bash
# 运行集成测试
python tests/test_integration.py
```

## 📁 项目结构

```
GR-ML-infra/
├── main.py                           # 🎯 主入口文件
├── integrations/                     # 🔌 开源框架集成
│   ├── hstu/                         # Meta HSTU模型集成
│   │   ├── hstu_model.py            # HSTU模型实现
│   │   ├── model_parameter_calculator.py
│   │   └── user_behavior_schema.py
│   ├── vllm/                         # VLLM推理引擎
│   │   └── vllm_engine.py
│   ├── tensorrt/                     # TensorRT加速引擎
│   │   ├── tensorrt_engine.py
│   │   └── build_engine.py
│   └── framework_controller.py       # 统一框架控制器
├── optimizations/                    # ⚡ 自定义优化算子
│   ├── triton_ops/                   # Triton自定义算子
│   ├── cutlass_ops/                  # CUTLASS算子
│   └── cache/                        # 智能GPU热缓存
│       └── intelligent_cache.py
├── external/                         # 📚 开源框架源码
│   ├── meta-hstu/                    # Meta HSTU模型源码
│   └── vllm/                         # VLLM框架源码
├── examples/                         # 📖 使用示例
│   └── client_example.py
├── tests/                            # 🧪 测试代码
├── models/                           # 🤖 模型文件存储
├── data/                             # 📊 数据文件
├── logs/                             # 📝 日志文件
├── configs/                          # ⚙️ 配置文件
└── docs/                             # 📚 项目文档
```

## 🔧 核心功能模块

### 1. 统一框架控制器 (`integrations/framework_controller.py`)

**统一推理流程**:
```python
def _unified_inference_pipeline(self, user_behaviors, ...):
    # Step 1: HSTU模型特征提取
    hstu_inputs = self._prepare_hstu_inputs(user_behaviors)
    
    # Step 2: TensorRT GPU优化推理
    trt_outputs = self.tensorrt_engine.infer(hstu_inputs)
    
    # Step 3: VLLM推理服务优化
    final_result = self._vllm_service_optimization(trt_outputs, ...)
    
    return final_result
```

### 2. Meta HSTU模型集成 (`integrations/hstu/hstu_model.py`)

**核心特性**:
- 集成Meta开源的HSTU架构
- 支持Hierarchical Sequential Transduction Units
- 多任务学习(engagement, retention, monetization)

### 3. VLLM推理引擎 (`integrations/vllm/vllm_engine.py`)

**性能优化**:
- PagedAttention内存优化
- Continuous Batching批处理优化
- 智能用户行为分析和推荐生成
- 异步推理支持

### 4. TensorRT优化引擎 (`integrations/tensorrt/tensorrt_engine.py`)

**加速特性**:
- 从ONNX/HSTU模型自动构建TensorRT引擎
- 动态形状优化配置
- FP16/INT8精度优化
- 自动内存管理和缓冲区优化

### 5. 智能缓存系统 (`optimizations/cache/intelligent_cache.py`)

**智能特性**:
- GPU热缓存优化
- 热点预测算法
- 智能驱逐策略
- 与开源框架无缝集成

## 📊 系统架构

### 核心组件
- **统一框架控制器**: 智能策略选择和资源调度
- **多引擎支持**: HSTU、VLLM、TensorRT协同工作
- **自定义优化**: Triton和CUTLASS算子加速
- **智能缓存**: 预测式GPU内存管理

### 技术优势
- **开源框架集成**: 基于成熟的开源技术栈
- **统一推理管道**: HSTU→ONNX→TensorRT→VLLM的端到端优化
- **自动优化链**: 从模型到服务的完整优化链路
- **完整监控体系**: 实时性能监控和调优

## 🎮 使用示例

### 单次推理

```python
from integrations.framework_controller import create_integrated_controller
from examples.client_example import create_realistic_user_behaviors

# 创建控制器
config = create_optimized_config()
controller = create_integrated_controller(config)

# 生成用户行为数据
user_behaviors = create_realistic_user_behaviors("demo_user", 15)

# 执行推理（统一推理流程）
result = controller.infer_with_optimal_strategy(
    user_id="demo_user_001",
    session_id="demo_session_001", 
    user_behaviors=user_behaviors,
    num_recommendations=10,
    strategy="unified"
)

print(f"推理策略: {result['inference_strategy']}")
print(f"推理时间: {result['inference_time_ms']:.2f}ms")
print(f"推荐数量: {len(result['recommendations'])}")
```

### 批量异步推理

```python
import asyncio

# 创建批量请求
batch_requests = [
    {
        'user_id': f'user_{i}',
        'session_id': f'session_{i}',
        'user_behaviors': create_realistic_user_behaviors(f"user_{i}", 10),
        'num_recommendations': 5,
        'strategy': 'unified'
    }
    for i in range(8)
]

# 执行批量推理
async def run_batch():
    return await controller.batch_infer(batch_requests)

results = asyncio.run(run_batch())
```

## 🔧 环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)

### 开源框架依赖（可选安装）
```bash
# Meta HSTU模型依赖
pip install torchrec>=0.7.0 fbgemm-gpu>=0.7.0

# VLLM推理框架
pip install vllm>=0.6.0

# TensorRT支持
pip install tensorrt>=10.0.0 tensorrt-cu12>=10.0.0
```

## 🧪 测试验证

```bash
# 运行完整集成测试
python tests/test_integration.py

# 预期结果: 6/6 测试通过 (100%)
```

测试覆盖：
- ✅ 项目结构完整性测试
- ✅ 配置生成功能测试  
- ✅ 框架导入兼容性测试
- ✅ 智能缓存功能测试
- ✅ 框架控制器测试
- ✅ 端到端集成流程测试

## 📈 监控和日志

### 实时监控
```bash
# 查看推理日志
tail -f opensoure_inference.log

# 查看基准测试结果
cat benchmark_results_*.json
```

### 性能指标
- 推理延迟分布 (P50, P95, P99)
- 各策略吞吐量对比
- 框架可用性状态
- GPU内存使用情况
- 缓存命中率统计

## 🚀 部署指南

### 开发环境部署
```bash
# 快速验证
python main.py --mode=single

# 性能测试
python main.py --mode=benchmark
```

### 生产环境部署
```bash
# 完整依赖安装
pip install vllm tensorrt torchrec

# 启动生产服务
python main.py --mode=comprehensive --log-level=INFO
```

## 🐛 故障排除

### 常见问题

1. **开源框架导入失败**
   - 项目具有智能回退机制，会自动使用可用的框架
   - 可选择性安装需要的框架: `pip install vllm tensorrt torchrec`

2. **GPU内存不足**
   ```python
   # 调整配置参数
   config['vllm']['gpu_memory_utilization'] = 0.7  # 降低GPU内存使用
   config['tensorrt']['max_batch_size'] = 4        # 减少批次大小
   ```

3. **性能未达预期**
   ```python
   # 启用详细监控
   config['monitoring']['enable_detailed_logging'] = True
   config['monitoring']['log_inference_time'] = True
   ```

### 调试技巧
```bash
# 启用调试模式
python main.py --mode=single --log-level=DEBUG

# 检查框架可用性
python -c "from integrations.framework_controller import create_integrated_controller; print(create_integrated_controller({}).framework_availability)"
```


### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd GR-ML-infra

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python tests/test_integration.py
```

## 🙏 致谢

感谢以下开源项目和团队：
- **Meta AI** - HSTU生成式推荐模型
- **VLLM团队** - 高性能推理优化框架
- **NVIDIA** - TensorRT GPU加速引擎
- **OpenAI** - Triton DSL自定义算子框架

---

**🎯 项目重点**: 这是一个真正基于开源框架的推荐系统推理优化项目，通过集成Meta HSTU、VLLM、TensorRT等顶级开源技术，实现了生产级的高性能推理系统。项目针对生成式推荐模型HSTU自定义了多个Triton和CUTLASS算子，通过TensorRT加速引擎进行加速，最后通过VLLM推理引擎进行推理，实现了生产级的高性能推理系统。是推理优化的完整解决方案。