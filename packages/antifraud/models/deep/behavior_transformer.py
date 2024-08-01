"""
Transformer-based Behavioral Analysis
Analyzes sequences of user interactions (mouse, keystroke) to detect non-human patterns
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BehavioralTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super().__init__()
        
        # Feature projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, input_dim]
        
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        
        # Transformer expects [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        # Global Average Pooling over sequence dimension
        # Output shape: [batch_size, d_model]
        x_pooled = output.mean(dim=0)
        
        # Classification
        prediction = self.classifier(x_pooled)
        return prediction

class BehavioralDetector:
    def __init__(self):
        self.model = BehavioralTransformer(input_dim=5, d_model=64)
        self.model.eval()
        
    def analyze_sequence(self, event_sequence: list):
        """
        Analyze a sequence of behavioral events.
        Each event: [x, y, pressure, dwell_time, velocity]
        """
        tensor_seq = torch.tensor([event_sequence], dtype=torch.float)
        
        with torch.no_grad():
            score = self.model(tensor_seq)
            
        return score.item()  # Probability of being human (0-1)

if __name__ == "__main__":
    detector = BehavioralDetector()
    # Mock sequence: 5 features x 20 time steps
    mock_seq = [[0.1, 0.2, 0.5, 100, 1.5] for _ in range(20)]
    score = detector.analyze_sequence(mock_seq)
    print(f"Human Probability Score: {score:.4f}")
