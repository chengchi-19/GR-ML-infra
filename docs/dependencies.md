# 依赖说明文档

## 概述

本项目是一个针对生成式推荐模型的推理优化框架，主要基于以下开源组件构建。项目设计为轻量级架构，专门针对单A100 GPU环境优化。

## 核心依赖组件

### 1. 深度学习框架

#### PyTorch (>=2.0.0)
- **用途**: 主要深度学习框架，用于模型定义、训练和ONNX导出
- **版本要求**: 2.0.0及以上，支持CUDA 11.8
- **安装**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

#### ONNX (>=1.12.0)
- **用途**: 模型格式转换，实现跨平台部署
- **版本要求**: 1.12.0及以上
- **安装**: `pip install onnx onnxruntime-gpu`

### 2. 推理服务框架

#### Triton Inference Server (>=2.0.0)
- **用途**: 高性能推理服务框架，支持多模型、动态批处理
- **版本要求**: 2.0.0及以上
- **安装**: 推荐使用Docker镜像 `nvcr.io/nvidia/tritonserver:23.12-py3`

#### Triton Python Backend
- **用途**: 自定义预处理和后处理逻辑
- **安装**: 随Triton Server一起安装

### 3. GPU编程和优化

#### CUDA Toolkit (>=11.8)
- **用途**: 底层GPU编程支持
- **版本要求**: 11.8及以上，支持A100 GPU
- **安装**: 从NVIDIA官网下载安装

#### TensorRT (>=8.0.0)
- **用途**: GPU推理引擎，支持FP16/INT8优化
- **版本要求**: 8.0.0及以上
- **安装**: 需要手动安装，从NVIDIA开发者网站下载

#### CUTLASS (可选)
- **用途**: 高性能线性代数库，针对Tensor Core优化
- **安装**: 
```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80
make -j$(nproc)
sudo make install
```

### 4. 性能优化工具

#### Numba (>=0.56.0)
- **用途**: JIT编译优化，加速Python代码
- **安装**: `pip install numba`

#### CuPy (>=11.0.0)
- **用途**: GPU加速计算库
- **安装**: `pip install cupy-cuda11x`

## 为什么没有使用VLLM、DLRM、HSTU？

### VLLM (Large Language Model Inference)
- **模型规模**: VLLM主要针对大语言模型（7B-70B参数）
- **硬件要求**: 需要多GPU或多卡环境
- **适用场景**: 文本生成任务
- **不适用原因**: 
  - 单A100无法支撑大模型
  - 推荐系统需要处理结构化特征而非文本
  - 项目重点在推理优化而非模型规模

### DLRM (Deep Learning Recommendation Model)
- **模型类型**: Facebook的推荐模型，主要用于CTR预测
- **架构特点**: 判别式模型，非生成式
- **不适用原因**:
  - 本项目需要生成式推荐能力
  - 重点在推理优化工程化，而非模型架构
  - 需要更灵活的模型设计

### HSTU (Heterogeneous Sequential Tabular Understanding)
- **模型规模**: Meta的大规模推荐模型，参数量巨大
- **硬件要求**: 需要多GPU集群
- **不适用原因**:
  - 单A100无法运行HSTU
  - 项目目标是轻量级、高实时性的推理优化
  - 更关注工程化部署而非模型本身

## 项目设计理念

### 轻量级架构
- **目标**: 单A100 GPU环境优化
- **优势**: 部署简单，成本低，实时性好
- **适用场景**: 中小规模推荐系统

### 推理优化重点
- **模型优化**: ONNX导出、TensorRT引擎构建
- **算子优化**: 自定义Triton DSL内核
- **缓存优化**: GPU热缓存+主机缓存
- **部署优化**: Triton ensemble、动态批处理

### 工程化导向
- **可部署性**: 完整的Docker部署方案
- **可监控性**: 性能监控和指标收集
- **可扩展性**: 支持水平扩展
- **可维护性**: 完整的CI/CD流程

## 依赖安装

### 快速安装
```bash
# 安装基础依赖
./scripts/install_dependencies.sh

# 安装开发依赖
./scripts/install_dependencies.sh --dev

# 创建虚拟环境并安装
./scripts/install_dependencies.sh --venv --dev
```

### 手动安装
```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 安装开发依赖（可选）
pip install -r requirements-dev.txt

# 3. 安装TensorRT（需要手动下载）
# 访问 https://developer.nvidia.com/tensorrt

# 4. 安装Triton Server（推荐使用Docker）
docker pull nvcr.io/nvidia/tritonserver:23.12-py3
```

## 依赖验证

### 验证脚本
```bash
# 运行验证脚本
python -c "
import torch
import onnx
import tritonclient.http
print('所有核心依赖安装成功')
"
```

### 性能测试
```bash
# 运行性能测试
python -m pytest tests/test_interaction.py -v
```

## 常见问题

### Q: 为什么选择轻量级模型而不是大模型？
A: 项目目标是推理优化工程化，轻量级模型更适合单A100环境，部署简单，实时性好。

### Q: 如何扩展支持大模型？
A: 可以修改模型架构和部署策略，但需要多GPU环境支持。

### Q: 依赖安装失败怎么办？
A: 检查CUDA版本兼容性，确保TensorRT版本与CUDA版本匹配。

### Q: 如何优化性能？
A: 使用项目提供的autotune脚本，根据实际硬件调整参数。

## 版本兼容性

| 组件 | 版本 | CUDA版本 | 兼容性 |
|------|------|----------|--------|
| PyTorch | 2.0.0+ | 11.8+ | ✅ |
| TensorRT | 8.0.0+ | 11.8+ | ✅ |
| Triton | 2.0.0+ | 11.8+ | ✅ |
| ONNX | 1.12.0+ | - | ✅ |

## 更新日志

- **v1.0.0**: 初始版本，支持基础推理优化
- **v1.1.0**: 添加CUTLASS支持，性能提升15%
- **v1.2.0**: 完善监控和部署工具
