#!/usr/bin/env python3
"""
MTGR模型ONNX导出脚本
支持prefill和decode两阶段导出，兼容TensorRT优化
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import logging

from src.mtgr_model import create_mtgr_model, MTGRWrapper

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTGRONNXExporter:
    """MTGR模型ONNX导出器"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model = create_mtgr_model(model_config)
        self.model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"MTGR模型加载完成，总参数量: {total_params:,} (约{total_params/1e9:.1f}B)")
    
    def create_dummy_data(self, batch_size: int = 1, seq_len: int = 100) -> Tuple[torch.Tensor, ...]:
        """创建用于ONNX导出的虚拟数据"""
        vocab_size = self.model_config.get('vocab_size', 50000)
        num_features = self.model_config.get('num_features', 1024)
        user_profile_dim = self.model_config.get('user_profile_dim', 256)
        item_feature_dim = self.model_config.get('item_feature_dim', 512)
        
        # 输入数据
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        dense_features = torch.randn(batch_size, num_features, dtype=torch.float32)
        user_profile = torch.randn(batch_size, user_profile_dim, dtype=torch.float32)
        item_features = torch.randn(batch_size, item_feature_dim, dtype=torch.float32)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        return input_ids, dense_features, user_profile, item_features, attention_mask
    
    def export_prefill_model(self, output_path: str, batch_size: int = 1, seq_len: int = 100):
        """导出Prefill阶段模型"""
        logger.info(f"导出Prefill模型到: {output_path}")
        
        # 创建虚拟数据
        dummy_data = self.create_dummy_data(batch_size, seq_len)
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_data,
            output_path,
            input_names=[
                'input_ids', 'dense_features', 'user_profile', 
                'item_features', 'attention_mask'
            ],
            output_names=[
                'logits', 'recommendation_score', 'engagement_score',
                'retention_score', 'monetization_score', 'hidden_states'
            ],
            opset_version=14,
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'dense_features': {0: 'batch_size'},
                'user_profile': {0: 'batch_size'},
                'item_features': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                'logits': {0: 'batch_size', 1: 'seq_len'},
                'recommendation_score': {0: 'batch_size'},
                'engagement_score': {0: 'batch_size'},
                'retention_score': {0: 'batch_size'},
                'monetization_score': {0: 'batch_size'},
                'hidden_states': {0: 'batch_size', 1: 'seq_len'}
            },
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        logger.info(f"Prefill模型导出完成: {output_path}")
    
    def export_decode_model(self, output_path: str, batch_size: int = 1, past_len: int = 500):
        """导出Decode阶段模型"""
        logger.info(f"导出Decode模型到: {output_path}")
        
        # 创建虚拟数据
        vocab_size = self.model_config.get('vocab_size', 50000)
        num_features = self.model_config.get('num_features', 1024)
        user_profile_dim = self.model_config.get('user_profile_dim', 256)
        item_feature_dim = self.model_config.get('item_feature_dim', 512)
        d_model = self.model_config.get('d_model', 1024)
        
        # Decode阶段的输入
        token_id = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.long)
        past_key_value_states = torch.zeros(batch_size, past_len, d_model, dtype=torch.float32)
        dense_features = torch.randn(batch_size, num_features, dtype=torch.float32)
        user_profile = torch.randn(batch_size, user_profile_dim, dtype=torch.float32)
        item_features = torch.randn(batch_size, item_feature_dim, dtype=torch.float32)
        
        # 创建Decode包装器
        decode_wrapper = DecodeWrapper(self.model)
        
        # 导出ONNX
        torch.onnx.export(
            decode_wrapper,
            (token_id, past_key_value_states, dense_features, user_profile, item_features),
            output_path,
            input_names=[
                'token_id', 'past_key_value_states', 'dense_features',
                'user_profile', 'item_features'
            ],
            output_names=[
                'logits', 'recommendation_score', 'engagement_score',
                'retention_score', 'monetization_score', 'present_key_value_states'
            ],
            opset_version=14,
            dynamic_axes={
                'token_id': {0: 'batch_size'},
                'past_key_value_states': {0: 'batch_size', 1: 'past_len'},
                'dense_features': {0: 'batch_size'},
                'user_profile': {0: 'batch_size'},
                'item_features': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'recommendation_score': {0: 'batch_size'},
                'engagement_score': {0: 'batch_size'},
                'retention_score': {0: 'batch_size'},
                'monetization_score': {0: 'batch_size'},
                'present_key_value_states': {0: 'batch_size', 1: 'present_len'}
            },
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        logger.info(f"Decode模型导出完成: {output_path}")
    
    def export_ensemble_model(self, prefill_path: str, decode_path: str, output_path: str):
        """导出集成模型配置"""
        logger.info(f"创建集成模型配置: {output_path}")
        
        # 这里可以创建Triton的集成模型配置
        # 由于ONNX本身不支持集成，这里只是创建配置文件
        config = {
            "prefill_model": prefill_path,
            "decode_model": decode_path,
            "ensemble_config": {
                "input_mapping": {
                    "input_ids": "prefill.input_ids",
                    "dense_features": "prefill.dense_features",
                    "user_profile": "prefill.user_profile",
                    "item_features": "prefill.item_features",
                    "attention_mask": "prefill.attention_mask"
                },
                "output_mapping": {
                    "logits": "prefill.logits",
                    "hidden_states": "prefill.hidden_states"
                }
            }
        }
        
        # 保存配置
        import json
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"集成模型配置已保存: {output_path}")

class DecodeWrapper(nn.Module):
    """Decode阶段的包装器"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, token_id, past_key_value_states, dense_features, user_profile, item_features):
        return self.model.forward_decode(
            token_id, past_key_value_states, dense_features, user_profile, item_features
        )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Export MTGR Model to ONNX')
    parser.add_argument('--prefill', default='mtgr_prefill.onnx', help='Output path for prefill ONNX')
    parser.add_argument('--decode', default='mtgr_decode.onnx', help='Output path for decode ONNX')
    parser.add_argument('--ensemble', default='mtgr_ensemble.json', help='Output path for ensemble config')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for export')
    parser.add_argument('--past_len', type=int, default=500, help='Past sequence length for decode')
    
    # MTGR模型配置参数
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=1024, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=24, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=4096, help='Feedforward dimension')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--num_features', type=int, default=1024, help='Number of dense features')
    parser.add_argument('--user_profile_dim', type=int, default=256, help='User profile dimension')
    parser.add_argument('--item_feature_dim', type=int, default=512, help='Item feature dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # 创建模型配置
    model_config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_seq_len': args.max_seq_len,
        'num_features': args.num_features,
        'user_profile_dim': args.user_profile_dim,
        'item_feature_dim': args.item_feature_dim,
        'dropout': args.dropout
    }
    
    print("=" * 60)
    print("MTGR模型ONNX导出")
    print("=" * 60)
    print(f"模型配置: {model_config}")
    print(f"导出配置: batch_size={args.batch_size}, seq_len={args.seq_len}")
    
    try:
        # 创建导出器
        exporter = MTGRONNXExporter(model_config)
        
        # 导出Prefill模型
        exporter.export_prefill_model(args.prefill, args.batch_size, args.seq_len)
        
        # 导出Decode模型
        exporter.export_decode_model(args.decode, args.batch_size, args.past_len)
        
        # 创建集成配置
        exporter.export_ensemble_model(args.prefill, args.decode, args.ensemble)
        
        print("\n✅ 所有模型导出完成!")
        print(f"  Prefill模型: {args.prefill}")
        print(f"  Decode模型: {args.decode}")
        print(f"  集成配置: {args.ensemble}")
        
        # 验证模型
        print("\n🔍 验证导出的模型...")
        try:
            import onnx
            prefill_model = onnx.load(args.prefill)
            decode_model = onnx.load(args.decode)
            print(f"  Prefill模型: {prefill_model.graph.input[0].type.tensor_type.shape.dim}")
            print(f"  Decode模型: {decode_model.graph.input[0].type.tensor_type.shape.dim}")
            print("✅ 模型验证通过")
        except ImportError:
            print("⚠️  ONNX未安装，跳过模型验证")
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
