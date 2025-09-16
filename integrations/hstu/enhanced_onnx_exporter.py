#!/usr/bin/env python3
"""
增强版HSTU模型ONNX导出工具

支持TensorRT自定义插件的ONNX导出，包括自定义算子节点的创建和集成。
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort
    import onnxoptimizer
    from onnx import helper, TensorProto, GraphProto, ModelProto
    ONNX_AVAILABLE = True
    logger.info("✅ ONNX相关库导入成功")
except ImportError as e:
    ONNX_AVAILABLE = False
    logger.warning(f"⚠️ ONNX相关库导入失败: {e}")

# 导入TensorRT插件支持
try:
    from ..tensorrt.plugins.python.tensorrt_plugins import (
        get_plugin_manager,
        create_fused_attention_layernorm_node,
        create_hierarchical_sequence_fusion_node,
        create_interaction_triton_fast_node
    )
    TENSORRT_PLUGINS_AVAILABLE = True
    logger.info("✅ TensorRT插件支持可用")
except ImportError as e:
    TENSORRT_PLUGINS_AVAILABLE = False
    logger.warning(f"⚠️ TensorRT插件支持不可用: {e}")


class EnhancedHSTUOnnxExporter:
    """
    增强版HSTU模型ONNX导出器

    支持TensorRT自定义插件和算子优化
    """

    def __init__(self,
                 model,
                 model_config,
                 export_dir: str = "./models",
                 opset_version: int = 17,
                 enable_custom_ops: bool = True):
        """
        初始化增强版ONNX导出器

        Args:
            model: HSTU模型实例
            model_config: 模型配置对象
            export_dir: 导出目录
            opset_version: ONNX操作集版本
            enable_custom_ops: 是否启用自定义算子
        """
        self.model = model
        self.config = model_config
        self.export_dir = export_dir
        self.opset_version = opset_version
        self.enable_custom_ops = enable_custom_ops and TENSORRT_PLUGINS_AVAILABLE

        # 创建导出目录
        os.makedirs(export_dir, exist_ok=True)

        # 检查ONNX可用性
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX相关库不可用，请安装: pip install onnx onnxruntime onnxoptimizer")

        # 初始化插件管理器
        self.plugin_manager = get_plugin_manager() if TENSORRT_PLUGINS_AVAILABLE else None

        logger.info(f"✅ 增强版HSTU ONNX导出器初始化成功 (自定义算子: {self.enable_custom_ops})")

    def export_with_custom_ops(self,
                              batch_sizes: List[int] = [1, 4, 8],
                              sequence_lengths: List[int] = [64, 128, 256, 512],
                              verify_export: bool = True,
                              optimize_model: bool = True) -> Dict[str, Any]:
        """
        导出包含自定义算子的ONNX模型

        Args:
            batch_sizes: 支持的批次大小列表
            sequence_lengths: 支持的序列长度列表
            verify_export: 是否验证导出模型
            optimize_model: 是否优化模型

        Returns:
            包含导出文件路径和自定义节点信息的字典
        """

        logger.info("开始导出包含自定义算子的HSTU模型...")

        # 准备导出路径
        base_name = f"hstu_custom_ops_opset{self.opset_version}"
        onnx_path = os.path.join(self.export_dir, f"{base_name}.onnx")
        custom_ops_path = os.path.join(self.export_dir, f"{base_name}_with_plugins.onnx")

        try:
            # 第一步: 导出标准ONNX模型
            logger.info("导出标准ONNX模型...")
            standard_export_result = self._export_standard_model(
                onnx_path, batch_sizes, sequence_lengths
            )

            if not standard_export_result:
                raise RuntimeError("标准ONNX模型导出失败")

            result_paths = {"standard_onnx": onnx_path}

            # 第二步: 插入自定义算子节点
            if self.enable_custom_ops:
                logger.info("插入自定义算子节点...")
                custom_ops_result = self._insert_custom_ops_nodes(
                    onnx_path, custom_ops_path
                )

                if custom_ops_result["success"]:
                    result_paths["custom_ops_onnx"] = custom_ops_path
                    result_paths["custom_nodes"] = custom_ops_result["custom_nodes"]
                    logger.info(f"✅ 自定义算子ONNX模型生成: {custom_ops_path}")
                else:
                    logger.warning("⚠️ 自定义算子插入失败，使用标准模型")

            # 第三步: 验证模型
            if verify_export:
                logger.info("验证导出的ONNX模型...")
                dummy_inputs = self._create_dummy_inputs(
                    max(batch_sizes), max(sequence_lengths)
                )

                # 验证标准模型
                if self._verify_onnx_model(onnx_path, dummy_inputs):
                    logger.info("✅ 标准ONNX模型验证通过")
                else:
                    logger.error("❌ 标准ONNX模型验证失败")

                # 如果有自定义算子模型，验证结构（不执行推理）
                if "custom_ops_onnx" in result_paths:
                    if self._verify_custom_ops_model_structure(custom_ops_path):
                        logger.info("✅ 自定义算子ONNX模型结构验证通过")
                    else:
                        logger.warning("⚠️ 自定义算子ONNX模型结构验证失败")

            # 第四步: 生成TensorRT构建配置
            tensorrt_config = self._generate_tensorrt_build_config(result_paths)
            result_paths["tensorrt_config"] = tensorrt_config

            # 生成模型信息
            self._generate_enhanced_model_info(result_paths, batch_sizes, sequence_lengths)

            return {
                "success": True,
                "export_paths": result_paths,
                "custom_ops_enabled": self.enable_custom_ops,
                "plugin_info": self._get_plugin_info(),
                "message": "增强版HSTU模型ONNX导出成功"
            }

        except Exception as e:
            logger.error(f"❌ 增强版ONNX导出失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "增强版HSTU模型ONNX导出失败"
            }

    def _export_standard_model(self, onnx_path: str, batch_sizes: List[int], seq_lens: List[int]) -> bool:
        """导出标准ONNX模型"""

        try:
            # 设置模型为评估模式
            self.model.eval()

            # 创建动态输入示例
            dummy_inputs = self._create_dummy_inputs(
                batch_size=max(batch_sizes),
                seq_len=max(seq_lens)
            )

            # 配置动态轴
            dynamic_axes = self._get_dynamic_axes_config()

            # 导出到ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model=self.model,
                    args=dummy_inputs,
                    f=onnx_path,
                    input_names=list(dummy_inputs.keys()),
                    output_names=['logits', 'hidden_states', 'engagement_scores',
                                'retention_scores', 'monetization_scores'],
                    dynamic_axes=dynamic_axes,
                    opset_version=self.opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    keep_initializers_as_inputs=False,
                    verbose=False
                )

            logger.info(f"✅ 标准ONNX模型导出成功: {onnx_path}")
            return True

        except Exception as e:
            logger.error(f"❌ 标准ONNX模型导出失败: {e}")
            return False

    def _insert_custom_ops_nodes(self, input_onnx_path: str, output_onnx_path: str) -> Dict[str, Any]:
        """在ONNX模型中插入自定义算子节点"""

        try:
            # 加载原始模型
            original_model = onnx.load(input_onnx_path)
            graph = original_model.graph

            # 分析模型图，识别可以被自定义算子替换的子图
            replacement_info = self._analyze_graph_for_custom_ops(graph)

            if not replacement_info["replaceable_subgraphs"]:
                logger.info("未找到可替换的子图，保持原始模型")
                return {"success": False, "reason": "no_replaceable_subgraphs"}

            # 创建新的图
            new_graph = self._create_graph_with_custom_ops(graph, replacement_info)

            # 创建新模型
            new_model = helper.make_model(
                new_graph,
                producer_name="GR-ML-infra",
                producer_version="1.0",
                opset_imports=[helper.make_opsetid("", self.opset_version)]
            )

            # 添加自定义域
            custom_opset = helper.make_opsetid("gr.ml.infra", 1)
            new_model.opset_import.append(custom_opset)

            # 保存新模型
            onnx.save(new_model, output_onnx_path)

            logger.info(f"✅ 自定义算子节点插入成功，替换了 {len(replacement_info['replaceable_subgraphs'])} 个子图")

            return {
                "success": True,
                "custom_nodes": replacement_info["custom_nodes"],
                "replaced_subgraphs": replacement_info["replaceable_subgraphs"]
            }

        except Exception as e:
            logger.error(f"❌ 自定义算子节点插入失败: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_graph_for_custom_ops(self, graph: GraphProto) -> Dict[str, Any]:
        """分析图中可以被自定义算子替换的子图"""

        replaceable_subgraphs = []
        custom_nodes = []

        # 查找融合注意力+LayerNorm模式
        attention_patterns = self._find_attention_layernorm_patterns(graph)
        for pattern in attention_patterns:
            custom_node = {
                "type": "FusedAttentionLayerNorm",
                "name": f"FusedAttentionLayerNorm_{len(custom_nodes)}",
                "inputs": pattern["inputs"],
                "outputs": pattern["outputs"],
                "attributes": {
                    "hidden_dim": self.config.d_model,
                    "num_heads": self.config.num_heads,
                    "dropout_rate": getattr(self.config, 'dropout', 0.1),
                    "layer_norm_eps": getattr(self.config, 'layer_norm_eps', 1e-5)
                }
            }
            custom_nodes.append(custom_node)
            replaceable_subgraphs.append(pattern)

        # 查找层次化序列融合模式
        hierarchical_patterns = self._find_hierarchical_sequence_patterns(graph)
        for pattern in hierarchical_patterns:
            custom_node = {
                "type": "HierarchicalSequenceFusion",
                "name": f"HierarchicalSequenceFusion_{len(custom_nodes)}",
                "inputs": pattern["inputs"],
                "outputs": pattern["outputs"],
                "attributes": {
                    "hidden_dim": self.config.d_model,
                    "num_levels": 3
                }
            }
            custom_nodes.append(custom_node)
            replaceable_subgraphs.append(pattern)

        # 查找交互算子模式
        interaction_patterns = self._find_interaction_patterns(graph)
        for pattern in interaction_patterns:
            custom_node = {
                "type": "InteractionTritonFast",
                "name": f"InteractionTritonFast_{len(custom_nodes)}",
                "inputs": pattern["inputs"],
                "outputs": pattern["outputs"],
                "attributes": {
                    "num_features": pattern.get("num_features", 20),
                    "embedding_dim": self.config.d_model
                }
            }
            custom_nodes.append(custom_node)
            replaceable_subgraphs.append(pattern)

        return {
            "replaceable_subgraphs": replaceable_subgraphs,
            "custom_nodes": custom_nodes
        }

    def _find_attention_layernorm_patterns(self, graph: GraphProto) -> List[Dict[str, Any]]:
        """查找注意力+LayerNorm模式"""
        patterns = []

        # 简化的模式匹配 - 在实际实现中需要更复杂的图分析
        # 这里我们假设找到了一些注意力模式
        for i, node in enumerate(graph.node):
            if "attention" in node.name.lower() or node.op_type == "MultiHeadAttention":
                # 检查后续是否有LayerNorm
                following_nodes = graph.node[i+1:i+5]  # 检查后续几个节点
                for follow_node in following_nodes:
                    if follow_node.op_type == "LayerNormalization":
                        pattern = {
                            "start_node": node.name,
                            "end_node": follow_node.name,
                            "inputs": list(node.input),
                            "outputs": list(follow_node.output),
                            "node_range": (i, i + following_nodes.index(follow_node) + 1)
                        }
                        patterns.append(pattern)
                        break

        return patterns

    def _find_hierarchical_sequence_patterns(self, graph: GraphProto) -> List[Dict[str, Any]]:
        """查找层次化序列处理模式"""
        patterns = []

        # 查找多层序列处理模式
        for i, node in enumerate(graph.node):
            if "sequence" in node.name.lower() and "fusion" in node.name.lower():
                pattern = {
                    "start_node": node.name,
                    "end_node": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "node_range": (i, i + 1)
                }
                patterns.append(pattern)

        return patterns

    def _find_interaction_patterns(self, graph: GraphProto) -> List[Dict[str, Any]]:
        """查找特征交互模式"""
        patterns = []

        # 查找矩阵乘法+点积模式
        for i, node in enumerate(graph.node):
            if node.op_type == "MatMul" and "interaction" in node.name.lower():
                pattern = {
                    "start_node": node.name,
                    "end_node": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "node_range": (i, i + 1),
                    "num_features": 20  # 默认值，实际需要从图中推断
                }
                patterns.append(pattern)

        return patterns

    def _create_graph_with_custom_ops(self, original_graph: GraphProto, replacement_info: Dict[str, Any]) -> GraphProto:
        """创建包含自定义算子的新图"""

        # 复制原始图的基本信息
        new_nodes = []
        removed_node_names = set()

        # 收集要删除的节点
        for subgraph in replacement_info["replaceable_subgraphs"]:
            start_idx, end_idx = subgraph["node_range"]
            for i in range(start_idx, end_idx):
                if i < len(original_graph.node):
                    removed_node_names.add(original_graph.node[i].name)

        # 复制未被替换的节点
        for node in original_graph.node:
            if node.name not in removed_node_names:
                new_nodes.append(node)

        # 添加自定义算子节点
        for custom_node_info in replacement_info["custom_nodes"]:
            custom_node = self._create_custom_op_node(custom_node_info)
            new_nodes.append(custom_node)

        # 创建新图
        new_graph = helper.make_graph(
            new_nodes,
            original_graph.name + "_with_custom_ops",
            original_graph.input,
            original_graph.output,
            original_graph.initializer
        )

        return new_graph

    def _create_custom_op_node(self, node_info: Dict[str, Any]):
        """创建自定义算子节点"""

        # 创建属性
        attributes = []
        for attr_name, attr_value in node_info["attributes"].items():
            if isinstance(attr_value, int):
                attr = helper.make_attribute(attr_name, attr_value)
            elif isinstance(attr_value, float):
                attr = helper.make_attribute(attr_name, attr_value)
            else:
                attr = helper.make_attribute(attr_name, str(attr_value))
            attributes.append(attr)

        # 创建节点
        custom_node = helper.make_node(
            node_info["type"],
            node_info["inputs"],
            node_info["outputs"],
            node_info["name"],
            domain="gr.ml.infra",
            **{attr.name: attr for attr in attributes}
        )

        return custom_node

    def _generate_tensorrt_build_config(self, result_paths: Dict[str, str]) -> str:
        """生成TensorRT构建配置文件"""

        config_content = {
            "tensorrt_build": {
                "input_model": result_paths.get("custom_ops_onnx", result_paths.get("standard_onnx")),
                "engine_path": "hstu_with_custom_ops.trt",
                "precision": "fp16",
                "max_batch_size": 8,
                "optimization_level": 5,
                "workspace_size": "2GB",
                "dynamic_shapes": {
                    "input_ids": {
                        "min": [1, 8],
                        "opt": [4, 64],
                        "max": [8, 2048]
                    },
                    "attention_mask": {
                        "min": [1, 8],
                        "opt": [4, 64],
                        "max": [8, 2048]
                    },
                    "dense_features": {
                        "min": [1, 1024],
                        "opt": [4, 1024],
                        "max": [8, 1024]
                    }
                },
                "custom_plugins": {
                    "enabled": self.enable_custom_ops,
                    "plugin_library": "libgr_ml_infra_tensorrt_plugins.so",
                    "custom_ops": [node["type"] for node in result_paths.get("custom_nodes", [])]
                }
            }
        }

        # 保存配置
        config_path = os.path.join(self.export_dir, "tensorrt_build_config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(config_content, f, indent=2)

        logger.info(f"✅ TensorRT构建配置保存到: {config_path}")
        return config_path

    def _create_dummy_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """创建虚拟输入数据"""

        dummy_inputs = {
            'input_ids': torch.randint(
                0, self.config.vocab_size,
                (batch_size, seq_len),
                dtype=torch.long
            ),
            'attention_mask': torch.ones(
                batch_size, seq_len,
                dtype=torch.long
            ),
            'dense_features': torch.randn(
                batch_size, getattr(self.config, 'dense_feature_dim', 1024),
                dtype=torch.float32
            ),
            'position_ids': torch.arange(
                seq_len, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1),
        }

        return dummy_inputs

    def _get_dynamic_axes_config(self) -> Dict[str, Dict[int, str]]:
        """获取动态轴配置"""

        return {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'dense_features': {0: 'batch_size'},
            'position_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'},
            'hidden_states': {0: 'batch_size', 1: 'sequence'},
            'engagement_scores': {0: 'batch_size'},
            'retention_scores': {0: 'batch_size'},
            'monetization_scores': {0: 'batch_size'},
        }

    def _verify_onnx_model(self, onnx_path: str, dummy_inputs: Dict[str, torch.Tensor]) -> bool:
        """验证导出的ONNX模型"""

        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 创建推理会话（仅使用CPU以避免自定义算子问题）
            ort_session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']
            )

            # 准备输入数据
            ort_inputs = {name: tensor.detach().cpu().numpy()
                         for name, tensor in dummy_inputs.items()}

            # 执行推理
            ort_outputs = ort_session.run(None, ort_inputs)

            logger.info("✅ ONNX模型验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ ONNX模型验证失败: {e}")
            return False

    def _verify_custom_ops_model_structure(self, onnx_path: str) -> bool:
        """验证自定义算子模型的结构"""

        try:
            # 加载并检查模型结构
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 检查是否包含自定义算子
            custom_ops_found = []
            for node in onnx_model.graph.node:
                if node.domain == "gr.ml.infra":
                    custom_ops_found.append(node.op_type)

            if custom_ops_found:
                logger.info(f"✅ 发现自定义算子: {custom_ops_found}")
                return True
            else:
                logger.warning("⚠️ 未发现自定义算子")
                return False

        except Exception as e:
            logger.error(f"❌ 自定义算子模型结构验证失败: {e}")
            return False

    def _get_plugin_info(self) -> Dict[str, Any]:
        """获取插件信息"""

        if not self.plugin_manager:
            return {"available": False}

        return {
            "available": self.plugin_manager.is_available(),
            "num_registered": self.plugin_manager.get_num_registered_plugins(),
            "supported_plugins": self.plugin_manager.get_supported_plugins()
        }

    def _generate_enhanced_model_info(self,
                                    result_paths: Dict[str, str],
                                    batch_sizes: List[int],
                                    seq_lens: List[int]):
        """生成增强版模型信息文件"""

        info = {
            'model_info': {
                'model_type': 'HSTU_Generative_Recommender_Enhanced',
                'opset_version': self.opset_version,
                'custom_ops_enabled': self.enable_custom_ops,
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'max_seq_len': self.config.max_seq_len,
            },
            'export_config': {
                'supported_batch_sizes': batch_sizes,
                'supported_sequence_lengths': seq_lens,
                'dynamic_axes_enabled': True,
                'tensorrt_plugins_support': self.enable_custom_ops,
            },
            'files': result_paths,
            'custom_ops_info': self._get_plugin_info(),
            'tensorrt_build_instructions': {
                'standard_build': f"trtexec --onnx={result_paths.get('standard_onnx')} --saveEngine=hstu_standard.trt --fp16",
                'custom_ops_build': f"trtexec --onnx={result_paths.get('custom_ops_onnx', 'N/A')} --saveEngine=hstu_custom_ops.trt --fp16 --plugins=libgr_ml_infra_tensorrt_plugins.so" if self.enable_custom_ops else "需要编译自定义插件库",
                'build_config': result_paths.get('tensorrt_config', 'N/A')
            }
        }

        # 保存信息文件
        info_path = os.path.join(self.export_dir, "enhanced_model_info.json")
        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 增强版模型信息保存到: {info_path}")


def export_hstu_model_with_custom_ops(model,
                                     model_config,
                                     export_dir: str = "./models",
                                     batch_sizes: List[int] = [1, 2, 4, 8],
                                     sequence_lengths: List[int] = [64, 128, 256, 512],
                                     enable_custom_ops: bool = True,
                                     optimize: bool = True) -> Dict[str, Any]:
    """
    导出包含自定义算子的HSTU模型到ONNX格式

    Args:
        model: HSTU模型实例
        model_config: 模型配置
        export_dir: 导出目录
        batch_sizes: 支持的批次大小
        sequence_lengths: 支持的序列长度
        enable_custom_ops: 是否启用自定义算子
        optimize: 是否优化模型

    Returns:
        包含导出结果的字典
    """

    logger.info("🚀 开始增强版HSTU模型ONNX导出流程...")

    # 创建增强版导出器
    exporter = EnhancedHSTUOnnxExporter(
        model=model,
        model_config=model_config,
        export_dir=export_dir,
        enable_custom_ops=enable_custom_ops
    )

    try:
        # 导出包含自定义算子的模型
        result = exporter.export_with_custom_ops(
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths,
            verify_export=True,
            optimize_model=optimize
        )

        logger.info("🎉 增强版HSTU模型ONNX导出完成!")
        logger.info(f"导出文件:")
        for key, value in result["export_paths"].items():
            if isinstance(value, list):
                for v in value:
                    logger.info(f"  - {v}")
            else:
                logger.info(f"  - {key}: {value}")

        return result

    except Exception as e:
        logger.error(f"❌ 增强版HSTU模型ONNX导出失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': '增强版HSTU模型ONNX导出失败'
        }


if __name__ == "__main__":
    # 测试增强版ONNX导出功能
    from .hstu_model import HSTUGenerativeRecommender, HSTUModelConfig

    # 创建测试模型
    config = HSTUModelConfig(
        vocab_size=10000,
        d_model=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=512
    )

    model = HSTUGenerativeRecommender(config)

    # 执行增强版导出
    result = export_hstu_model_with_custom_ops(
        model=model,
        model_config=config,
        export_dir="./test_models_enhanced",
        batch_sizes=[1, 2, 4],
        sequence_lengths=[64, 128, 256],
        enable_custom_ops=True,
        optimize=True
    )

    print("增强版导出结果:", result)