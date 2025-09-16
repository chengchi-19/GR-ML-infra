# 🌟 基于生成式推荐模型的推理优化项目

## 📋 项目概述

这是一个基于**开源框架**的生成式推荐模型推理优化部署项目，集成了Meta开源的生成式HSTU模型、自定义Triton和CUTLASS算子、TensorRT加速引擎、VLLM推理引擎：
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
- 🌐 **RESTful API服务**: FastAPI提供生产级API接口

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

# 安装开源框架（可选，项目支持智能回退机制）
pip install vllm tensorrt torchrec fbgemm-gpu
```

### 2. 启动API服务 (推荐)

```bash
# 方式1: 使用启动脚本（推荐）
./start_api_server.sh

# 方式2: 直接启动
python api_server.py


```bash
# 检查系统框架可用性
python main.py --action=check

# 生成推理配置文件
python main.py --action=config --config-file=my_config.json

# 运行简单推理测试
python main.py --action=test
```

## 🔧 核心功能模块

### 1. 统一框架控制器 (`integrations/framework_controller.py`)

**统一推理流程**:
```python
def infer_with_unified_pipeline(self, user_behaviors, ...):
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

### 6. FastAPI服务层 (`api_server.py`)

**服务特性**:
- RESTful API接口设计
- 异步请求处理
- 实时性能监控
- 自动API文档生成
- 完善的错误处理和日志记录

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


## 🧪 测试和验证

```bash
# 运行集成测试
python tests/test_integration.py

# 运行Triton算子测试
python tests/test_triton_integration.py

# API服务完整测试
python api_client_demo.py
```

# 🙏 致谢

感谢以下开源项目和团队：
- **Meta AI** - HSTU生成式推荐模型
- **VLLM团队** - 高性能推理优化框架
- **NVIDIA** - TensorRT GPU加速引擎
- **OpenAI** - Triton DSL自定义算子框架

---

**🎯 项目重点**: 这是一个基于开源框架的推荐系统推理优化项目，通过集成Meta HSTU、VLLM、TensorRT等顶级开源技术，实现了生成式推荐模型的推理优化部署流程。项目针对生成式推荐模型HSTU自定义了多个Triton和CUTLASS算子，通过TensorRT推理引擎进行加速，最后通过VLLM推理引擎进行推理。通过集成多种开源框架，本项目实现了**1+1+1+1>4的协同效应**，为推荐系统推理优化提供了完整的解决方案。