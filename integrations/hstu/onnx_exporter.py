#!/usr/bin/env python3
"""
HSTU模型ONNX导出工具

提供完整的HSTU模型ONNX导出、验证和优化功能，
支持动态批次、多输出和TensorRT兼容性优化。
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
    ONNX_AVAILABLE = True
    logger.info("✅ ONNX相关库导入成功")
except ImportError as e:
    ONNX_AVAILABLE = False
    logger.warning(f"⚠️ ONNX相关库导入失败: {e}")


class HSTUOnnxExporter:
    """
    HSTU模型ONNX导出器
    
    支持完整的导出、验证、优化流程，确保TensorRT兼容性
    """
    
    def __init__(self, 
                 model,
                 model_config,
                 export_dir: str = "./models",
                 opset_version: int = 17):
        """
        初始化ONNX导出器
        
        Args:
            model: HSTU模型实例
            model_config: 模型配置对象
            export_dir: 导出目录
            opset_version: ONNX操作集版本
        """
        self.model = model
        self.config = model_config
        self.export_dir = export_dir
        self.opset_version = opset_version
        
        # 创建导出目录
        os.makedirs(export_dir, exist_ok=True)
        
        # 检查ONNX可用性
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX相关库不可用，请安装: pip install onnx onnxruntime onnxoptimizer")
        
        logger.info("✅ HSTU ONNX导出器初始化成功")
    
    def export_full_model(self,
                         batch_sizes: List[int] = [1, 4, 8],
                         sequence_lengths: List[int] = [64, 128, 256, 512],
                         verify_export: bool = True,
                         optimize_model: bool = True) -> Dict[str, str]:
        """
        导出完整的HSTU模型到ONNX
        
        Args:
            batch_sizes: 支持的批次大小列表
            sequence_lengths: 支持的序列长度列表  
            verify_export: 是否验证导出模型
            optimize_model: 是否优化模型
            
        Returns:
            包含导出文件路径的字典
        """
        
        logger.info("开始导出HSTU模型到ONNX格式...")
        
        # 准备导出路径
        base_name = f"hstu_model_opset{self.opset_version}"
        onnx_path = os.path.join(self.export_dir, f"{base_name}.onnx")
        optimized_path = os.path.join(self.export_dir, f"{base_name}_optimized.onnx")
        
        try:
            # 设置模型为评估模式
            self.model.eval()
            
            # 创建动态输入示例
            dummy_inputs = self._create_dummy_inputs(
                batch_size=max(batch_sizes),
                seq_len=max(sequence_lengths)
            )
            
            # 配置动态轴
            dynamic_axes = self._get_dynamic_axes_config(batch_sizes, sequence_lengths)
            
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
            
            logger.info(f"✅ ONNX模型导出成功: {onnx_path}")
            
            result_paths = {"onnx_model": onnx_path}
            
            # 验证导出模型
            if verify_export:
                logger.info("验证导出的ONNX模型...")
                is_valid = self._verify_onnx_model(onnx_path, dummy_inputs)
                if not is_valid:
                    logger.error("❌ ONNX模型验证失败")
                    return result_paths
                logger.info("✅ ONNX模型验证成功")
            
            # 优化模型
            if optimize_model:
                logger.info("优化ONNX模型...")
                success = self._optimize_onnx_model(onnx_path, optimized_path)
                if success:
                    result_paths["optimized_model"] = optimized_path
                    logger.info(f"✅ ONNX模型优化成功: {optimized_path}")
                else:
                    logger.warning("⚠️ ONNX模型优化失败，使用原始模型")
            
            # 生成模型信息
            self._generate_model_info(result_paths, batch_sizes, sequence_lengths)
            
            return result_paths
            
        except Exception as e:
            logger.error(f"❌ ONNX导出失败: {e}")
            raise
    
    def export_inference_only(self,
                            batch_size: int = 1,
                            sequence_length: int = 128,
                            include_past_key_values: bool = False) -> str:
        """
        导出仅推理版本的模型（去除训练相关组件）
        
        Args:
            batch_size: 固定批次大小
            sequence_length: 固定序列长度
            include_past_key_values: 是否包含KV缓存
            
        Returns:
            导出的ONNX文件路径
        """
        
        logger.info("导出推理专用ONNX模型...")
        
        # 创建推理专用模型包装器
        inference_model = self._create_inference_wrapper(include_past_key_values)
        
        # 准备固定输入
        dummy_inputs = self._create_dummy_inputs(batch_size, sequence_length)
        if not include_past_key_values:
            # 移除不必要的输入
            dummy_inputs = {k: v for k, v in dummy_inputs.items() 
                          if 'past_key_values' not in k}
        
        # 导出路径
        onnx_path = os.path.join(
            self.export_dir, 
            f"hstu_inference_b{batch_size}_s{sequence_length}.onnx"
        )
        
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model=inference_model,
                    args=tuple(dummy_inputs.values()),
                    f=onnx_path,
                    input_names=list(dummy_inputs.keys()),
                    output_names=['logits', 'engagement_scores', 'retention_scores', 'monetization_scores'],
                    opset_version=self.opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    keep_initializers_as_inputs=False,
                    verbose=False
                )
            
            logger.info(f"✅ 推理专用ONNX模型导出成功: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"❌ 推理专用ONNX导出失败: {e}")
            raise
    
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
        }
        
        # 添加可选输入
        if hasattr(self.config, 'dense_feature_dim'):
            dummy_inputs['dense_features'] = torch.randn(
                batch_size, self.config.dense_feature_dim,
                dtype=torch.float32
            )
        else:
            dummy_inputs['dense_features'] = torch.randn(
                batch_size, 1024,
                dtype=torch.float32  
            )
        
        # 添加位置ID（可选）
        dummy_inputs['position_ids'] = torch.arange(
            seq_len, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        
        return dummy_inputs
    
    def _get_dynamic_axes_config(self, batch_sizes: List[int], seq_lens: List[int]) -> Dict[str, Dict[int, str]]:
        """获取动态轴配置"""
        
        dynamic_axes = {
            # 输入动态轴
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'dense_features': {0: 'batch_size'},
            'position_ids': {0: 'batch_size', 1: 'sequence'},
            
            # 输出动态轴
            'logits': {0: 'batch_size', 1: 'sequence'},
            'hidden_states': {0: 'batch_size', 1: 'sequence'},
            'engagement_scores': {0: 'batch_size'},
            'retention_scores': {0: 'batch_size'},
            'monetization_scores': {0: 'batch_size'},
        }
        
        return dynamic_axes
    
    def _create_inference_wrapper(self, include_kv_cache: bool = False):
        """创建推理专用模型包装器"""
        
        class InferenceWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask, dense_features, position_ids):
                outputs = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dense_features=dense_features,
                    position_ids=position_ids,
                    return_dict=True
                )
                
                # 只返回推理需要的输出
                return (
                    outputs['logits'],
                    outputs['engagement_scores'],
                    outputs['retention_scores'], 
                    outputs['monetization_scores']
                )
        
        return InferenceWrapper(self.model)
    
    def _verify_onnx_model(self, onnx_path: str, dummy_inputs: Dict[str, torch.Tensor]) -> bool:
        """验证导出的ONNX模型"""
        
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 检查模型结构
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX模型结构检查通过")
            
            # 创建推理会话
            ort_session = ort.InferenceSession(
                onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # 准备输入数据
            ort_inputs = {}
            for name, tensor in dummy_inputs.items():
                ort_inputs[name] = tensor.detach().cpu().numpy()
            
            # 执行推理
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # 与PyTorch模型对比
            with torch.no_grad():
                torch_outputs = self.model(**dummy_inputs)
            
            # 检查输出一致性
            torch_logits = torch_outputs['logits'].detach().cpu().numpy()
            onnx_logits = ort_outputs[0]
            
            if np.allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-3):
                logger.info("✅ PyTorch和ONNX输出一致性验证通过")
                return True
            else:
                logger.error("❌ PyTorch和ONNX输出不一致")
                logger.error(f"最大差异: {np.max(np.abs(torch_logits - onnx_logits))}")
                return False
            
        except Exception as e:
            logger.error(f"❌ ONNX模型验证失败: {e}")
            return False
    
    def _optimize_onnx_model(self, input_path: str, output_path: str) -> bool:
        """优化ONNX模型"""
        
        try:
            # 加载模型
            model = onnx.load(input_path)
            
            # 应用优化
            optimized_model = onnxoptimizer.optimize(
                model,
                passes=[
                    'eliminate_deadend',
                    'eliminate_identity',
                    'eliminate_nop_dropout',
                    'eliminate_nop_monotone_argmax',
                    'eliminate_nop_pad',
                    'extract_constant_to_initializer',
                    'eliminate_unused_initializer',
                    'eliminate_nop_transpose',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm',
                ]
            )
            
            # 保存优化后的模型
            onnx.save(optimized_model, output_path)
            
            # 验证优化后的模型
            onnx.checker.check_model(optimized_model)
            
            logger.info("✅ ONNX模型优化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX模型优化失败: {e}")
            return False
    
    def _generate_model_info(self, 
                           result_paths: Dict[str, str], 
                           batch_sizes: List[int], 
                           seq_lens: List[int]):
        """生成模型信息文件"""
        
        info = {
            'model_info': {
                'model_type': 'HSTU_Generative_Recommender',
                'opset_version': self.opset_version,
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
            },
            'files': result_paths,
            'usage': {
                'tensorrt_build': f"trtexec --onnx={result_paths.get('optimized_model', result_paths['onnx_model'])} --saveEngine=hstu.trt --fp16",
                'onnxruntime_inference': "ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])"
            }
        }
        
        # 保存信息文件
        info_path = os.path.join(self.export_dir, "model_info.json")
        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 模型信息保存到: {info_path}")


def export_hstu_model(model, 
                     model_config,
                     export_dir: str = "./models",
                     batch_sizes: List[int] = [1, 2, 4, 8],
                     sequence_lengths: List[int] = [64, 128, 256, 512],
                     export_inference_only: bool = True,
                     optimize: bool = True) -> Dict[str, Any]:
    """
    导出HSTU模型到ONNX格式的便捷函数
    
    Args:
        model: HSTU模型实例
        model_config: 模型配置
        export_dir: 导出目录
        batch_sizes: 支持的批次大小
        sequence_lengths: 支持的序列长度
        export_inference_only: 是否导出推理专用版本
        optimize: 是否优化模型
        
    Returns:
        包含导出结果的字典
    """
    
    logger.info("🚀 开始HSTU模型ONNX导出流程...")
    
    # 创建导出器
    exporter = HSTUOnnxExporter(
        model=model,
        model_config=model_config,
        export_dir=export_dir
    )
    
    results = {}
    
    try:
        # 导出完整模型
        logger.info("导出完整动态ONNX模型...")
        full_model_paths = exporter.export_full_model(
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths,
            verify_export=True,
            optimize_model=optimize
        )
        results.update(full_model_paths)
        
        # 导出推理专用模型
        if export_inference_only:
            logger.info("导出推理专用ONNX模型...")
            inference_paths = []
            
            # 为几个常用配置导出推理专用模型
            common_configs = [
                (1, 128),   # 单样本中等长度
                (4, 64),    # 小批次短序列
                (8, 256),   # 大批次长序列
            ]
            
            for batch_size, seq_len in common_configs:
                try:
                    inference_path = exporter.export_inference_only(
                        batch_size=batch_size,
                        sequence_length=seq_len
                    )
                    inference_paths.append(inference_path)
                except Exception as e:
                    logger.warning(f"推理专用模型导出失败 (b{batch_size}_s{seq_len}): {e}")
            
            results['inference_models'] = inference_paths
        
        logger.info("🎉 HSTU模型ONNX导出完成!")
        logger.info(f"导出文件:")
        for key, value in results.items():
            if isinstance(value, list):
                for v in value:
                    logger.info(f"  - {v}")
            else:
                logger.info(f"  - {key}: {value}")
        
        return {
            'success': True,
            'export_paths': results,
            'message': 'HSTU模型ONNX导出成功'
        }
        
    except Exception as e:
        logger.error(f"❌ HSTU模型ONNX导出失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'HSTU模型ONNX导出失败'
        }


if __name__ == "__main__":
    # 测试ONNX导出功能
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
    
    # 执行导出
    result = export_hstu_model(
        model=model,
        model_config=config,
        export_dir="./test_models",
        batch_sizes=[1, 2, 4],
        sequence_lengths=[64, 128, 256],
        export_inference_only=True,
        optimize=True
    )
    
    print("导出结果:", result)