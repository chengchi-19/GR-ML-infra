#!/usr/bin/env python3
import torch
import pytest

from src.mtgr_model import create_mtgr_model


def test_prefill_decode_shapes_and_progression():
    model = create_mtgr_model({
        'vocab_size': 50000,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'd_ff': 512,
        'max_seq_len': 256,
        'num_features': 1024,
        'user_profile_dim': 256,
        'item_feature_dim': 512,
        'dropout': 0.1,
    })
    model.eval()

    batch_size = 2
    seq_len = 10
    vocab = 50000

    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
    dense_features = torch.randn(batch_size, 1024, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        logits, rec, eng, ret, mon, hidden = model.forward_prefill(
            input_ids, dense_features, None, None, attention_mask
        )

    assert logits.shape[:2] == (batch_size, seq_len)
    assert hidden.shape[:2] == (batch_size, seq_len + 1) or hidden.shape[:2] == (batch_size, seq_len + 1) or hidden.shape[:2] == (batch_size, seq_len + 1)
    # 允许额外特征位拼接后的长度>=seq_len
    assert hidden.shape[1] >= seq_len

    # decode 前进一步
    last_token = input_ids[:, -1:]
    with torch.no_grad():
        d_logits, d_rec, d_eng, d_ret, d_mon, d_hidden = model.forward_decode(
            last_token, hidden, dense_features, None, None
        )

    assert d_logits.shape[0] == batch_size
    assert d_logits.shape[1] == 1
    # 新的隐藏应当长度>=原隐藏
    assert d_hidden.shape[1] >= hidden.shape[1]


