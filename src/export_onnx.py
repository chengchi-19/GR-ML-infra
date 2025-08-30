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
    """Generative Recommendation Model with prefill and decode stages"""
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, 
                 num_features=16, num_layers=4, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Feature processing
        self.feature_projection = nn.Linear(num_features, embedding_dim)
        self.interaction_layer = InteractionLayer(embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.feature_output = nn.Linear(embedding_dim, 1)  # For recommendation scores
        
    def forward_prefill(self, input_ids, dense_features, attention_mask=None):
        """
        Prefill stage: process input sequence and dense features
        Args:
            input_ids: [batch_size, seq_len]
            dense_features: [batch_size, num_features]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_emb + pos_emb
        
        # Process dense features
        feature_emb = self.feature_projection(dense_features.unsqueeze(1))  # [batch_size, 1, embedding_dim]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Apply transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=(attention_mask == 0))
        
        # Generate logits for next token prediction
        logits = self.output_projection(transformer_output)  # [batch_size, seq_len, vocab_size]
        
        # Generate recommendation scores from features
        feature_scores = self.feature_output(feature_emb).squeeze(-1)  # [batch_size, 1]
        
        return logits, feature_scores, transformer_output
    
    def forward_decode(self, token_id, past_key_value_states, dense_features):
        """
        Decode stage: generate next token
        Args:
            token_id: [batch_size, 1]
            past_key_value_states: [batch_size, past_len, embedding_dim]
            dense_features: [batch_size, num_features]
        """
        batch_size = token_id.shape[0]
        
        # Current token embedding
        token_emb = self.token_embedding(token_id)  # [batch_size, 1, embedding_dim]
        
        # Position embedding for current position
        current_pos = past_key_value_states.shape[1] if past_key_value_states is not None else 0
        pos_emb = self.position_embedding(torch.tensor([current_pos], device=token_id.device)).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine embeddings
        current_emb = token_emb + pos_emb
        
        # Concatenate with past states
        if past_key_value_states is not None:
            combined_emb = torch.cat([past_key_value_states, current_emb], dim=1)
        else:
            combined_emb = current_emb
        
        # Apply transformer (only to the new token)
        transformer_output = self.transformer(combined_emb)
        new_output = transformer_output[:, -1:, :]  # Only the last token output
        
        # Generate logits
        logits = self.output_projection(new_output)  # [batch_size, 1, vocab_size]
        
        # Process dense features for recommendation
        feature_emb = self.feature_projection(dense_features.unsqueeze(1))
        feature_scores = self.feature_output(feature_emb).squeeze(-1)
        
        return logits, feature_scores, transformer_output

class DecodeWrapper(nn.Module):
    """Wrapper for decode stage ONNX export"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, token_id, past_key_value_states, dense_features):
        logits, feature_scores, present_key_value = self.model.forward_decode(
            token_id, past_key_value_states, dense_features
        )
        return logits, feature_scores, present_key_value

def create_dummy_data(batch_size=1, seq_len=16, num_features=16, vocab_size=10000):
    """Create dummy data for ONNX export"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dense_features = torch.randn(batch_size, num_features, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, dense_features, attention_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Generative Recommendation Model to ONNX')
    parser.add_argument('--prefill', default='prefill.onnx', help='Output path for prefill ONNX')
    parser.add_argument('--decode', default='decode.onnx', help='Output path for decode ONNX')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_features', type=int, default=16, help='Number of dense features')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length for export')
    
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
    dummy_ids, dummy_dense, dummy_mask = create_dummy_data(
        args.batch_size, args.seq_len, args.num_features, args.vocab_size
    )
    
    torch.onnx.export(
        model,
        (dummy_ids, dummy_dense, dummy_mask),
        args.prefill,
        input_names=['input_ids', 'dense_features', 'attention_mask'],
        output_names=['logits', 'feature_scores', 'hidden_states'],
        opset_version=14,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'dense_features': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'},
            'feature_scores': {0: 'batch_size'},
            'hidden_states': {0: 'batch_size', 1: 'seq_len'}
        },
        do_constant_folding=True,
        export_params=True
    )
    print(f"Prefill model exported to {args.prefill}")
    
    # Export decode model
    print(f"Exporting decode model to {args.decode}...")
    dummy_token = torch.randint(0, args.vocab_size, (args.batch_size, 1), dtype=torch.long)
    dummy_past = torch.zeros(args.batch_size, 1, args.embedding_dim, dtype=torch.float32)
    
    decode_wrapper = DecodeWrapper(model)
    torch.onnx.export(
        decode_wrapper,
        (dummy_token, dummy_past, dummy_dense),
        args.decode,
        input_names=['token_id', 'past_key_value_states', 'dense_features'],
        output_names=['logits', 'feature_scores', 'present_key_value_states'],
        opset_version=14,
        dynamic_axes={
            'token_id': {0: 'batch_size'},
            'past_key_value_states': {0: 'batch_size', 1: 'past_len'},
            'dense_features': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'feature_scores': {0: 'batch_size'},
            'present_key_value_states': {0: 'batch_size', 1: 'present_len'}
        },
        do_constant_folding=True,
        export_params=True
    )
    print(f"Decode model exported to {args.decode}")
    
    print("ONNX export completed successfully!")
