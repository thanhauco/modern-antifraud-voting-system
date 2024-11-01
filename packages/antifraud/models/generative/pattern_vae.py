"""
Variational Autoencoder for Voting Pattern Anomaly Detection
Learns normal voting distributions and flags reconstruction anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

class VotingPatternEncoder(nn.Module):
    """Encoder network for VAE"""
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VotingPatternDecoder(nn.Module):
    """Decoder network for VAE"""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128, output_dim: int = 50):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class VotingPatternVAE(nn.Module):
    """
    Variational Autoencoder for unsupervised anomaly detection.
    
    Features encoded:
    - Vote timing patterns (hour distribution)
    - Geographic clustering
    - Device fingerprint similarity
    - Voter demographics
    """
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        
        self.encoder = VotingPatternEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VotingPatternDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through stochastic node"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor, 
                      beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        ELBO loss = Reconstruction + KL Divergence
        Beta-VAE formulation for disentanglement
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl_divergence": kl_loss
        }

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error for anomaly scoring"""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            errors = torch.mean((x - x_recon) ** 2, dim=1)
        return errors


class PatternAnomalyDetector:
    """
    Production anomaly detection service using VAE.
    """
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VotingPatternVAE().to(self.device)
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Threshold calibrated on validation set (99th percentile)
        self.anomaly_threshold = 0.15
        
        # Running statistics for online calibration
        self.error_history = []

    def extract_features(self, vote_event: dict) -> np.ndarray:
        """
        Extract feature vector from raw vote event.
        
        Features include:
        - Hour of day (one-hot, 24 dims)
        - Day of week (one-hot, 7 dims)
        - Device risk score (1 dim)
        - IP risk score (1 dim)
        - Session duration (1 dim)
        - Click entropy (1 dim)
        - Geographic distance from registration (1 dim)
        - ... (total 50 dims)
        """
        features = np.zeros(50)
        
        # Time features
        hour = vote_event.get("hour", 12)
        features[hour] = 1.0
        
        # Additional features
        features[24] = vote_event.get("device_risk", 0.5)
        features[25] = vote_event.get("ip_risk", 0.5)
        features[26] = min(vote_event.get("session_duration", 300) / 600, 1.0)
        features[27] = vote_event.get("click_entropy", 0.5)
        
        return features

    def detect(self, vote_event: dict) -> dict:
        """
        Detect if a vote event is anomalous.
        """
        features = self.extract_features(vote_event)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        error = self.model.get_reconstruction_error(x).item()
        self.error_history.append(error)
        
        # Dynamic threshold adjustment
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
            dynamic_threshold = np.percentile(self.error_history, 99)
            self.anomaly_threshold = max(self.anomaly_threshold, dynamic_threshold)
        
        is_anomaly = error > self.anomaly_threshold
        
        return {
            "is_anomaly": is_anomaly,
            "reconstruction_error": round(error, 6),
            "threshold": round(self.anomaly_threshold, 6),
            "severity": "high" if error > 2 * self.anomaly_threshold else "medium" if is_anomaly else "low"
        }


if __name__ == "__main__":
    # Test VAE
    model = VotingPatternVAE(input_dim=50, latent_dim=32)
    dummy_batch = torch.randn(16, 50)
    x_recon, mu, logvar = model(dummy_batch)
    
    losses = model.loss_function(dummy_batch, x_recon, mu, logvar)
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Total loss: {losses['total'].item():.4f}")
    
    # Test detector
    detector = PatternAnomalyDetector()
    result = detector.detect({"hour": 3, "device_risk": 0.9, "session_duration": 5})
    print(f"Anomaly result: {result}")
