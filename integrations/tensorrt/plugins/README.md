# TensorRT自定义插件构建与部署指南

## 概述

本文档详细说明如何构建、部署和使用GR-ML-infra项目中的TensorRT自定义插件。

## 架构说明

### 问题解决

**原始问题**: 项目中的Triton自定义算子无法被TensorRT识别，导致TensorRT引擎优化时完全忽略这些高性能算子。

**解决方案**: 实现TensorRT插件(Plugin)机制，将Triton算子转换为TensorRT可识别的C++插件。

### 整体架构

```
Triton算子 → TensorRT C++插件 → ONNX自定义节点 → TensorRT引擎优化
    ↓              ↓                    ↓              ↓
优化的CUDA      插件注册机制        增强版ONNX导出    集成优化推理
   算子         (C++ + Python)        流程            引擎
```

## 核心组件

### 1. TensorRT插件基础框架

- **位置**: `integrations/tensorrt/plugins/`
- **核心文件**:
  - `include/tensorrt_plugin_base.h` - 插件基类定义
  - `cpp/tensorrt_plugin_base.cpp` - 基类实现
  - `cpp/plugin_registry.cpp` - 插件注册管理

### 2. 自定义算子插件实现

#### 融合注意力+LayerNorm插件
- **头文件**: `include/fused_attention_layernorm_plugin.h`
- **实现**: `cpp/fused_attention_layernorm_plugin.cpp`
- **CUDA Kernel**: `cpp/fused_attention_layernorm_kernel.cu`
- **对应Triton算子**: `optimizations/triton_ops/fused_attention_layernorm.py`

#### 支持的插件类型
```cpp
FusedAttentionLayerNorm      // 融合多头注意力+LayerNorm
HierarchicalSequenceFusion  // 层次化序列融合
HSTUHierarchicalAttention   // HSTU层次化注意力
InteractionTritonFast       // 快速特征交互
SequenceRecommendationInteraction // 序列推荐交互
```

### 3. Python集成接口

- **插件管理**: `plugins/python/tensorrt_plugins.py`
- **ONNX导出增强**: `integrations/hstu/enhanced_onnx_exporter.py`
- **TensorRT引擎集成**: `integrations/tensorrt/tensorrt_engine.py`

## 构建指南

### 环境要求

```bash
# 基础环境
CUDA >= 11.8
TensorRT >= 8.5
Python >= 3.8
PyTorch >= 1.13

# 构建工具
cmake >= 3.15
gcc >= 7.5 (支持C++17)
```

### 1. 安装依赖

```bash
# 安装CUDA和TensorRT
# 参考NVIDIA官方文档

# 安装Python依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorrt
pip install onnx onnxruntime onnxoptimizer
pip install pycuda

# 安装构建工具
apt-get install cmake build-essential
```

### 2. 设置环境变量

```bash
export CUDA_HOME=/usr/local/cuda
export TENSORRT_ROOT=/path/to/TensorRT
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

### 3. 构建TensorRT插件库

```bash
cd integrations/tensorrt/plugins

# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME

# 编译
make -j$(nproc)

# 安装
sudo make install
```

### 4. 验证构建

```bash
# 检查库文件
ls -la libgr_ml_infra_tensorrt_plugins.so

# 测试插件加载
python -c "from integrations.tensorrt.plugins.python.tensorrt_plugins import initialize_plugins; print('插件加载:', initialize_plugins())"
```

## 使用指南

### 1. 基础使用

```python
from integrations.hstu.enhanced_onnx_exporter import export_hstu_model_with_custom_ops
from integrations.tensorrt.tensorrt_engine import create_tensorrt_engine

# 导出包含自定义算子的ONNX模型
export_result = export_hstu_model_with_custom_ops(
    model=hstu_model,
    model_config=config,
    enable_custom_ops=True,
    export_dir="./models"
)

# 创建支持自定义插件的TensorRT引擎
engine = create_tensorrt_engine(
    onnx_path=export_result["export_paths"]["custom_ops_onnx"],
    precision="fp16",
    enable_custom_plugins=True
)

# 执行推理
outputs = engine.infer(inputs)
```

### 2. 高级配置

```python
from integrations.tensorrt.tensorrt_engine import TensorRTConfig

# 详细配置
config = TensorRTConfig(
    model_name="hstu_optimized",
    precision="fp16",
    max_batch_size=8,
    enable_custom_plugins=True,
    plugin_lib_paths=[
        "/path/to/libgr_ml_infra_tensorrt_plugins.so"
    ],
    enable_dynamic_shapes=True,
    optimization_level=5
)

engine = TensorRTOptimizedEngine(config)
```

### 3. 性能测试

```bash
# 运行完整测试套件
python tests/test_tensorrt_plugins.py

# 单独测试组件
python -m integrations.tensorrt.plugins.python.tensorrt_plugins
```

## TensorRT引擎构建流程

### 1. 标准构建流程

```bash
# 使用trtexec构建标准引擎
trtexec --onnx=hstu_standard.onnx \
        --saveEngine=hstu_standard.trt \
        --fp16 \
        --workspace=2048
```

### 2. 自定义插件构建流程

```bash
# 构建包含自定义插件的引擎
trtexec --onnx=hstu_custom_ops.onnx \
        --saveEngine=hstu_custom_ops.trt \
        --fp16 \
        --workspace=2048 \
        --plugins=libgr_ml_infra_tensorrt_plugins.so
```

### 3. 动态形状配置

```bash
# 配置动态输入形状
trtexec --onnx=hstu_custom_ops.onnx \
        --saveEngine=hstu_dynamic.trt \
        --fp16 \
        --minShapes=input_ids:1x8,attention_mask:1x8 \
        --optShapes=input_ids:4x64,attention_mask:4x64 \
        --maxShapes=input_ids:8x2048,attention_mask:8x2048 \
        --plugins=libgr_ml_infra_tensorrt_plugins.so
```

## 性能优化

### 1. 插件性能优化

#### CUDA Kernel优化
- 使用Shared Memory缓存频繁访问的数据
- 避免Bank Conflict
- 使用矢量化加载(128-bit)
- 实现数值稳定的Softmax

#### 内存访问优化
- 连续内存访问模式
- 避免非对齐访问
- 使用Pinned Memory
- 异步内存传输

### 2. TensorRT引擎优化

```python
# 高性能配置
config = TensorRTConfig(
    precision="fp16",           # 使用FP16精度
    optimization_level=5,       # 最高优化级别
    max_workspace_size=2<<30,   # 2GB工作空间
    enable_custom_plugins=True, # 启用自定义插件
    enable_strict_types=False,  # 允许精度转换
)
```

### 3. 批次和序列长度优化

#### 最优配置
- **批次大小**: 4-8 (在延迟和吞吐量之间平衡)
- **序列长度**: 64, 128, 256 (常见长度)
- **动态形状**: 根据实际使用场景配置

## 故障排除

### 1. 常见问题

#### 插件加载失败
```bash
# 检查库依赖
ldd libgr_ml_infra_tensorrt_plugins.so

# 检查TensorRT版本兼容性
python -c "import tensorrt; print(tensorrt.__version__)"

# 检查CUDA兼容性
nvidia-smi
nvcc --version
```

#### ONNX模型解析失败
```python
# 验证ONNX模型
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# 检查自定义算子
for node in model.graph.node:
    if node.domain == "gr.ml.infra":
        print(f"自定义算子: {node.op_type}")
```

#### TensorRT引擎构建失败
```bash
# 启用详细日志
export TRT_LOGGER_SEVERITY=INFO

# 检查插件注册
trtexec --help | grep -i plugin
```

### 2. 调试技巧

#### 开启调试信息
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# TensorRT详细日志
import tensorrt as trt
logger = trt.Logger(trt.Logger.VERBOSE)
```

#### 性能分析
```bash
# 使用nsight分析性能
nsys profile python inference_test.py

# 检查GPU利用率
nvidia-smi -l 1
```

## 生产部署

### 1. 容器化部署

```dockerfile
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# 复制插件库
COPY libgr_ml_infra_tensorrt_plugins.so /usr/local/lib/

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制应用代码
COPY . /app
WORKDIR /app

# 运行应用
CMD ["python", "api_server.py"]
```

### 2. 性能监控

```python
# 集成到API服务器
from integrations.tensorrt.tensorrt_engine import TensorRTOptimizedEngine

class InferenceAPI:
    def __init__(self):
        self.engine = TensorRTOptimizedEngine(config)

    def predict(self, inputs):
        start_time = time.time()
        outputs = self.engine.infer(inputs)
        latency = time.time() - start_time

        # 记录性能指标
        self.metrics.record_latency(latency)
        self.metrics.record_throughput(len(inputs))

        return outputs
```

### 3. 自动扩容

```yaml
# Kubernetes HPA配置
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tensorrt-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tensorrt-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

## 总结

通过实现TensorRT自定义插件，我们成功解决了Triton算子无法被TensorRT识别的问题，实现了：

1. **完整的插件框架**: 支持多种自定义算子的TensorRT插件
2. **增强的ONNX导出**: 自动插入自定义算子节点
3. **集成的TensorRT引擎**: 支持自定义插件的推理引擎
4. **性能优化**: 充分发挥GPU性能的CUDA kernel实现
5. **生产级部署**: 完整的构建、测试和部署流程

这个解决方案实现了真正的端到端优化，让项目中的高性能Triton算子能够在TensorRT推理引擎中发挥作用，大幅提升了推理性能。