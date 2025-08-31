#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InteractionLayer(nn.Module):
    """Pairwise interaction layer for recommendation features"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # x: [batch_size, num_features, feature_dim]
        batch_size, num_features, feature_dim = x.shape
        # Generate pairwise interactions
        interactions = []
        for i in range(num_features):
            for j in range(i+1, num_features):
                interaction = torch.sum(x[:, i, :] * x[:, j, :], dim=1, keepdim=True)
                interactions.append(interaction)
        return torch.cat(interactions, dim=1)  # [batch_size, num_interactions]

class GenerativeRecommendationModel(nn.Module):
    """生成式推荐模型 - 扩展版本，支持大规模输入参数"""
    
    def __init__(self, vocab_size=10000, embedding_dim=512, hidden_dim=1024, 
                 num_features=1024, num_layers=6, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.max_seq_len = max_seq_len
        
        # Token和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # 特征投影层 - 扩展到1024维
        self.feature_projection = nn.Linear(num_features, embedding_dim)
        
        # 用户画像投影层 - 新增
        self.user_profile_projection = nn.Linear(256, embedding_dim)  # 256维用户画像
        
        # 视频特征投影层 - 新增
        self.video_feature_projection = nn.Linear(512, embedding_dim)  # 512维视频特征
        
        # 交互层
        self.interaction_layer = InteractionLayer(embedding_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=16,  # 增加注意力头数
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.feature_output = nn.Linear(embedding_dim, 1)  # 推荐分数
        
        # 多任务输出层 - 新增
        self.engagement_output = nn.Linear(embedding_dim, 1)  # 参与度预测
        self.retention_output = nn.Linear(embedding_dim, 1)   # 留存预测
        self.monetization_output = nn.Linear(embedding_dim, 1) # 商业化预测
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward_prefill(self, input_ids, dense_features, user_profile=None, 
                       video_features=None, attention_mask=None):
        """
        Prefill阶段: 处理输入序列和密集特征
        Args:
            input_ids: [batch_size, seq_len]
            dense_features: [batch_size, num_features] - 扩展的1024维特征
            user_profile: [batch_size, 256] - 用户画像特征
            video_features: [batch_size, 512] - 视频特征
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token嵌入
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # 组合嵌入
        embeddings = token_emb + pos_emb
        
        # 处理密集特征
        feature_emb = self.feature_projection(dense_features.unsqueeze(1))  # [batch_size, 1, embedding_dim]
        
        # 处理用户画像 - 新增
        if user_profile is not None:
            user_emb = self.user_profile_projection(user_profile.unsqueeze(1))  # [batch_size, 1, embedding_dim]
            feature_emb = feature_emb + user_emb
        
        # 处理视频特征 - 新增
        if video_features is not None:
            video_emb = self.video_feature_projection(video_features.unsqueeze(1))  # [batch_size, 1, embedding_dim]
            feature_emb = feature_emb + video_emb
        
        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # 应用Transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=(attention_mask == 0))
        
        # 生成输出
        logits = self.output_projection(transformer_output)  # [batch_size, seq_len, vocab_size]
        
        # 多任务输出 - 新增
        feature_scores = self.feature_output(feature_emb).squeeze(-1)  # [batch_size, 1]
        engagement_scores = self.engagement_output(feature_emb).squeeze(-1)  # [batch_size, 1]
        retention_scores = self.retention_output(feature_emb).squeeze(-1)  # [batch_size, 1]
        monetization_scores = self.monetization_output(feature_emb).squeeze(-1)  # [batch_size, 1]
        
        return logits, feature_scores, engagement_scores, retention_scores, monetization_scores, transformer_output
    
    def forward_decode(self, token_id, past_key_value_states, dense_features, 
                      user_profile=None, video_features=None):
        """
        Decode阶段: 生成下一个token
        Args:
            token_id: [batch_size, 1]
            past_key_value_states: [batch_size, past_len, embedding_dim]
            dense_features: [batch_size, num_features]
            user_profile: [batch_size, 256]
            video_features: [batch_size, 512]
        """
        batch_size = token_id.shape[0]
        
        # 当前token嵌入
        token_emb = self.token_embedding(token_id)  # [batch_size, 1, embedding_dim]
        
        # 位置嵌入
        current_pos = past_key_value_states.shape[1] if past_key_value_states is not None else 0
        pos_emb = self.position_embedding(torch.tensor([current_pos], device=token_id.device)).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 组合嵌入
        current_emb = token_emb + pos_emb
        
        # 连接历史状态
        if past_key_value_states is not None:
            combined_emb = torch.cat([past_key_value_states, current_emb], dim=1)
        else:
            combined_emb = current_emb
        
        # 应用Transformer
        transformer_output = self.transformer(combined_emb)
        new_output = transformer_output[:, -1:, :]  # 只取最后一个token的输出
        
        # 生成输出
        logits = self.output_projection(new_output)  # [batch_size, 1, vocab_size]
        
        # 处理特征
        feature_emb = self.feature_projection(dense_features.unsqueeze(1))
        
        # 处理用户画像
        if user_profile is not None:
            user_emb = self.user_profile_projection(user_profile.unsqueeze(1))
            feature_emb = feature_emb + user_emb
        
        # 处理视频特征
        if video_features is not None:
            video_emb = self.video_feature_projection(video_features.unsqueeze(1))
            feature_emb = feature_emb + video_emb
        
        # 多任务输出
        feature_scores = self.feature_output(feature_emb).squeeze(-1)
        engagement_scores = self.engagement_output(feature_emb).squeeze(-1)
        retention_scores = self.retention_output(feature_emb).squeeze(-1)
        monetization_scores = self.monetization_output(feature_emb).squeeze(-1)
        
        return logits, feature_scores, engagement_scores, retention_scores, monetization_scores, transformer_output

class DecodeWrapper(nn.Module):
    """Wrapper for decode stage ONNX export"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, token_id, past_key_value_states, dense_features, user_profile, video_features):
        logits, feature_scores, engagement_scores, retention_scores, monetization_scores, present_key_value = self.model.forward_decode(
            token_id, past_key_value_states, dense_features, user_profile, video_features
        )
        return logits, feature_scores, engagement_scores, retention_scores, monetization_scores, present_key_value

def create_dummy_data(batch_size=1, seq_len=1000, num_features=1024, vocab_size=10000):
    """Create dummy data for ONNX export - 扩展版本"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dense_features = torch.randn(batch_size, num_features, dtype=torch.float32)
    user_profile = torch.randn(batch_size, 256, dtype=torch.float32)  # 256维用户画像
    video_features = torch.randn(batch_size, 512, dtype=torch.float32)  # 512维视频特征
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, dense_features, user_profile, video_features, attention_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Generative Recommendation Model to ONNX')
    parser.add_argument('--prefill', default='prefill.onnx', help='Output path for prefill ONNX')
    parser.add_argument('--decode', default='decode.onnx', help='Output path for decode ONNX')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--num_features', type=int, default=1024, help='Number of dense features (extended for user behavior)')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length') 
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--seq_len', type=int, default=1000, help='Sequence length for export')
    
    args = parser.parse_args()
    
    # Create model
    model = GenerativeRecommendationModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_features=args.num_features,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    )
    model.eval()
    
    # Export prefill model
    print(f"Exporting prefill model to {args.prefill}...")
    dummy_ids, dummy_dense, dummy_user, dummy_video, dummy_mask = create_dummy_data(
        args.batch_size, args.seq_len, args.num_features, args.vocab_size
    )
    
    torch.onnx.export(
        model,
        (dummy_ids, dummy_dense, dummy_user, dummy_video, dummy_mask),
        args.prefill,
        input_names=['input_ids', 'dense_features', 'user_profile', 'video_features', 'attention_mask'],
        output_names=['logits', 'feature_scores', 'engagement_scores', 'retention_scores', 'monetization_scores', 'hidden_states'],
        opset_version=14,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'dense_features': {0: 'batch_size'},
            'user_profile': {0: 'batch_size'},
            'video_features': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'},
            'feature_scores': {0: 'batch_size'},
            'engagement_scores': {0: 'batch_size'},
            'retention_scores': {0: 'batch_size'},
            'monetization_scores': {0: 'batch_size'},
            'hidden_states': {0: 'batch_size', 1: 'seq_len'}
        },
        do_constant_folding=True,
        export_params=True
    )
    print(f"Prefill model exported to {args.prefill}")
    
    # Export decode model
    print(f"Exporting decode model to {args.decode}...")
    dummy_token = torch.randint(0, args.vocab_size, (args.batch_size, 1), dtype=torch.long)
    dummy_past = torch.zeros(args.batch_size, 500, args.embedding_dim, dtype=torch.float32)  # 增加KV缓存长度
    dummy_user = torch.randn(args.batch_size, 256, dtype=torch.float32)
    dummy_video = torch.randn(args.batch_size, 512, dtype=torch.float32)
    
    decode_wrapper = DecodeWrapper(model)
    torch.onnx.export(
        decode_wrapper,
        (dummy_token, dummy_past, dummy_dense, dummy_user, dummy_video),
        args.decode,
        input_names=['token_id', 'past_key_value_states', 'dense_features', 'user_profile', 'video_features'],
        output_names=['logits', 'feature_scores', 'engagement_scores', 'retention_scores', 'monetization_scores', 'present_key_value_states'],
        opset_version=14,
        dynamic_axes={
            'token_id': {0: 'batch_size'},
            'past_key_value_states': {0: 'batch_size', 1: 'past_len'},
            'dense_features': {0: 'batch_size'},
            'user_profile': {0: 'batch_size'},
            'video_features': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'feature_scores': {0: 'batch_size'},
            'engagement_scores': {0: 'batch_size'},
            'retention_scores': {0: 'batch_size'},
            'monetization_scores': {0: 'batch_size'},
            'present_key_value_states': {0: 'batch_size', 1: 'present_len'}
        },
        do_constant_folding=True,
        export_params=True
    )
    print(f"Decode model exported to {args.decode}")
    
    print("ONNX export completed successfully!")
