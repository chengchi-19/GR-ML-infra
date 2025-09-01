#!/usr/bin/env python3
"""
快速测试MTGR模型功能
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mtgr_model import create_mtgr_model

def test_mtgr_basic():
    """测试MTGR模型基本功能"""
    print("测试MTGR模型基本功能...")
    
    try:
        # 1. 创建模型
        model_config = {
            'vocab_size': 50000,
            'd_model': 1024,
            'nhead': 16,
            'num_layers': 24,
            'd_ff': 4096,
            'max_seq_len': 2048,
            'num_features': 1024,
            'user_profile_dim': 256,
            'item_feature_dim': 512,
            'dropout': 0.1
        }
        
        model = create_mtgr_model(model_config)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功，参数量: {total_params:,}")
        
        # 2. 创建测试数据
        batch_size = 1
        seq_len = 50
        num_features = 1024
        
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        dense_features = torch.randn(batch_size, num_features)
        user_profile = torch.randn(batch_size, 256)
        item_features = torch.randn(batch_size, 512)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        print(f"✅ 测试数据创建成功")
        print(f"   输入形状: {input_ids.shape}")
        
        # 3. 测试前向传播
        with torch.no_grad():
            outputs = model.forward_prefill(
                input_ids, dense_features, user_profile, item_features, attention_mask
            )
        
        print(f"✅ 前向传播成功")
        print(f"   输出形状: {[out.shape for out in outputs]}")
        
        # 4. 测试Decode阶段
        last_token = input_ids[:, -1:]
        # 使用正确的past states格式
        past_states = torch.randn(batch_size, seq_len, model_config['d_model'])
        
        decode_outputs = model.forward_decode(
            last_token, past_states, dense_features, user_profile, item_features
        )
        
        print(f"✅ Decode阶段成功")
        print(f"   解码输出形状: {[out.shape for out in decode_outputs]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mtgr_basic()
    if success:
        print("\n🎉 MTGR模型测试通过！")
    else:
        print("\n❌ MTGR模型测试失败！")
