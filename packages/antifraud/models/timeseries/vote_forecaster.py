"""
Temporal Fusion Transformer for Vote Volume Forecasting
Predicts expected vote counts per jurisdiction for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ForecastConfig:
    input_size: int = 16       # Number of input features
    hidden_size: int = 64      # Hidden layer size
    lstm_layers: int = 2       # LSTM depth
    num_attention_heads: int = 4
    dropout: float = 0.1
    quantiles: List[float] = None  # For uncertainty estimation
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]  # 10th, median, 90th percentile


class GatedResidualNetwork(nn.Module):
    """
    GRN component from TFT paper.
    Provides non-linear processing with skip connections.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Skip connection projection if dims differ
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # Primary path
        hidden = self.fc1(x)
        
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_proj(context)
        
        hidden = F.elu(hidden)
        
        # GLU activation
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        output = self.dropout(output) * gate
        
        # Skip connection
        skip = self.skip_proj(x) if self.skip_proj else x
        
        return self.layer_norm(output + skip)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for vote volume forecasting.
    
    Static features: jurisdiction demographics, historical turnout
    Dynamic features: time of day, weather, social media sentiment
    """
    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.input_size, config.hidden_size)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention for interpretable patterns
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # GRN for enrichment
        self.grn = GatedResidualNetwork(
            config.hidden_size, config.hidden_size, config.hidden_size, config.dropout
        )
        
        # Quantile output heads
        self.output_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, 1) for _ in config.quantiles
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [batch, seq_len, input_size]
        Returns quantile predictions for uncertainty estimation.
        """
        # Embed inputs
        embedded = self.input_embedding(x)
        
        # Temporal processing
        lstm_out, _ = self.lstm(embedded)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Enrichment
        enriched = self.grn(attn_out)
        
        # Take last time step for prediction
        final = enriched[:, -1, :]
        
        # Quantile predictions
        quantile_preds = {}
        for i, q in enumerate(self.config.quantiles):
            quantile_preds[f"q{int(q*100)}"] = self.output_layers[i](final).squeeze(-1)
        
        return {
            "predictions": quantile_preds,
            "attention_weights": attn_weights
        }


class VoteForecaster:
    """
    Production vote volume forecasting service.
    """
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = ForecastConfig()
        self.model = TemporalFusionTransformer(self.config).to(self.device)
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, historical_data: List[Dict], horizon: int = 1) -> Dict:
        """
        Predict vote volume for next `horizon` time periods.
        
        historical_data: List of feature dicts for past time steps
        """
        # Extract features
        features = self._prepare_features(historical_data)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        predictions = output["predictions"]
        
        return {
            "expected_votes": predictions["q50"].item(),
            "lower_bound": predictions["q10"].item(),
            "upper_bound": predictions["q90"].item(),
            "uncertainty": predictions["q90"].item() - predictions["q10"].item()
        }

    def detect_anomaly(self, predicted: float, actual: float, lower: float, upper: float) -> Dict:
        """
        Check if actual vote count is anomalous relative to prediction.
        """
        if actual < lower:
            return {"anomaly": True, "type": "suppression", "severity": (lower - actual) / lower}
        elif actual > upper:
            return {"anomaly": True, "type": "inflation", "severity": (actual - upper) / upper}
        else:
            return {"anomaly": False, "type": None, "severity": 0.0}

    def _prepare_features(self, data: List[Dict]) -> List[List[float]]:
        """Convert raw data to feature matrix"""
        features = []
        for d in data:
            f = [
                d.get("hour", 0) / 24,
                d.get("day_of_week", 0) / 7,
                d.get("temperature", 70) / 100,
                d.get("precipitation", 0),
                d.get("historical_turnout", 0.5),
                d.get("registered_voters", 100000) / 1000000,
                d.get("early_vote_pct", 0.3),
                d.get("mail_ballot_pct", 0.2),
            ]
            # Pad to input_size
            f.extend([0.0] * (self.config.input_size - len(f)))
            features.append(f)
        return features


if __name__ == "__main__":
    config = ForecastConfig()
    model = TemporalFusionTransformer(config)
    
    # Test forward pass
    dummy_input = torch.randn(4, 24, 16)  # batch=4, seq=24 hours, features=16
    output = model(dummy_input)
    
    print(f"Median prediction shape: {output['predictions']['q50'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")
    
    # Test forecaster
    forecaster = VoteForecaster()
    mock_history = [{"hour": h, "day_of_week": 2} for h in range(24)]
    prediction = forecaster.predict(mock_history)
    print(f"Forecast: {prediction}")
