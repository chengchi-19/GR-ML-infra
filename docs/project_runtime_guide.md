# 项目运行指南和文件依赖关系

## 项目概述

这是一个生成式推荐模型推理优化项目，实现了从用户行为数据到推荐结果的完整推理流程。

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或者安装开发依赖
pip install -r requirements-dev.txt
```

### 2. 运行项目

```bash
# 运行完整流程（推荐）
python main.py --mode all

# 运行单次推理
python main.py --mode single

# 运行批量推理
python main.py --mode batch

# 运行性能测试
python main.py --mode performance

# 导出ONNX模型
python main.py --mode export

# 设置日志级别
python main.py --mode all --log-level DEBUG
```

## 文件依赖关系拓扑图

```
main.py (主入口)
├── src/
│   ├── inference_pipeline.py (推理流水线)
│   │   ├── src/export_onnx.py (模型定义)
│   │   ├── src/user_behavior_schema.py (用户行为模式)
│   │   └── src/model_parameter_calculator.py (参数计算)
│   ├── export_onnx.py (ONNX导出)
│   ├── user_behavior_schema.py (用户行为定义)
│   ├── model_parameter_calculator.py (模型参数分析)
│   └── build_engine.py (TensorRT引擎构建)
├── examples/
│   └── client_example.py (客户端示例)
├── tests/
│   └── test_interaction.py (测试文件)
└── triton_model_repo/ (Triton模型仓库)
    ├── interaction_python/ (Python后端)
    └── preprocess_py/ (预处理后端)
```

## 详细文件依赖关系

### 1. 核心文件依赖链

```
main.py
├── src/inference_pipeline.py
│   ├── src/export_onnx.py (GenerativeRecommendationModel)
│   ├── src/user_behavior_schema.py (UserBehaviorProcessor)
│   └── src/model_parameter_calculator.py (参数计算)
├── examples/client_example.py (示例数据)
└── 日志输出到 inference.log
```

### 2. 模型定义依赖

```
src/export_onnx.py
├── torch.nn.Module (基础类)
├── src/user_behavior_schema.py (用户行为模式)
└── 输出: prefill.onnx, decode.onnx
```

### 3. 用户行为处理依赖

```
src/user_behavior_schema.py
├── dataclasses (数据类定义)
├── datetime (时间处理)
└── typing (类型注解)
```

### 4. 推理流水线依赖

```
src/inference_pipeline.py
├── src/export_onnx.py (模型)
├── src/user_behavior_schema.py (行为处理)
├── torch (深度学习框架)
├── numpy (数值计算)
└── logging (日志记录)
```

## 运行时文件执行顺序

### 阶段1: 初始化 (Initialization)
1. **main.py** - 项目入口
2. **src/inference_pipeline.py** - 创建推理流水线
3. **src/export_onnx.py** - 初始化模型
4. **src/model_parameter_calculator.py** - 计算模型参数
5. **src/user_behavior_schema.py** - 初始化用户行为处理器

### 阶段2: 数据准备 (Data Preparation)
1. **examples/client_example.py** - 生成示例用户行为数据
2. **src/user_behavior_schema.py** - 处理用户行为序列
3. **src/inference_pipeline.py** - 特征提取和转换

### 阶段3: 模型推理 (Model Inference)
1. **src/export_onnx.py** - 模型前向传播
2. **src/inference_pipeline.py** - 推理流水线执行
3. **torch** - 深度学习计算

### 阶段4: 结果输出 (Result Output)
1. **src/inference_pipeline.py** - 结果格式化
2. **main.py** - 结果展示和日志记录

## 关键文件功能说明

### 核心文件

| 文件 | 功能 | 依赖 |
|------|------|------|
| `main.py` | 项目主入口，串联整个流程 | 所有src模块 |
| `src/inference_pipeline.py` | 推理流水线核心逻辑 | 模型、行为处理 |
| `src/export_onnx.py` | 模型定义和ONNX导出 | PyTorch |
| `src/user_behavior_schema.py` | 用户行为数据结构和处理 | dataclasses |
| `src/model_parameter_calculator.py` | 模型参数分析和计算 | PyTorch |

### 示例和测试文件

| 文件 | 功能 | 依赖 |
|------|------|------|
| `examples/client_example.py` | 客户端示例和测试数据 | 用户行为模式 |
| `tests/test_interaction.py` | 交互层测试 | 模型定义 |

### 部署相关文件

| 文件 | 功能 | 依赖 |
|------|------|------|
| `src/build_engine.py` | TensorRT引擎构建 | TensorRT |
| `triton_model_repo/` | Triton推理服务器配置 | Triton |

## 运行模式详解

### 1. 单次推理模式 (`--mode single`)
```
main.py → 初始化 → 单次推理 → 结果输出
```

### 2. 批量推理模式 (`--mode batch`)
```
main.py → 初始化 → 批量推理 → 性能统计 → 结果输出
```

### 3. 性能测试模式 (`--mode performance`)
```
main.py → 初始化 → 预热 → 性能测试 → 统计结果
```

### 4. 模型导出模式 (`--mode export`)
```
main.py → 初始化 → ONNX导出 → 模型保存
```

### 5. 完整模式 (`--mode all`)
```
main.py → 初始化 → 单次推理 → 批量推理 → 性能测试 → 模型导出
```

## 输出文件说明

### 日志文件
- `inference.log` - 推理过程日志

### 模型文件
- `models/prefill.onnx` - Prefill阶段ONNX模型
- `models/decode.onnx` - Decode阶段ONNX模型

### 临时文件
- 缓存文件 (如果启用缓存)
- 临时张量数据

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/gr-inference-opt-updated
   python main.py
   ```

2. **依赖缺失**
   ```bash
   # 安装所有依赖
   pip install -r requirements.txt
   ```

3. **内存不足**
   ```bash
   # 使用较小的批次大小
   python main.py --mode single
   ```

4. **CUDA错误**
   ```bash
   # 检查CUDA版本兼容性
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### 调试模式

```bash
# 启用详细日志
python main.py --mode all --log-level DEBUG

# 查看日志文件
tail -f inference.log
```

## 性能优化建议

1. **使用GPU加速** (如果可用)
2. **调整批次大小** 根据内存情况
3. **启用缓存** 提高重复请求性能
4. **使用ONNX模型** 提高推理速度
5. **批量处理** 提高吞吐量

## 扩展开发

### 添加新功能
1. 在 `src/` 目录下创建新模块
2. 在 `main.py` 中添加相应的调用
3. 更新依赖关系和文档

### 自定义模型
1. 修改 `src/export_onnx.py` 中的模型定义
2. 更新 `src/inference_pipeline.py` 中的推理逻辑
3. 重新运行测试确保功能正常
