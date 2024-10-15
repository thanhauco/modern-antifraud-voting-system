"""
GAN-based Adversarial Bot Discriminator
Training a discriminator to detect sophisticated automated voting bots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

class BotBehaviorGenerator(nn.Module):
    """
    Generator that learns to simulate bot behavior.
    Used to create adversarial examples for discriminator training.
    """
    def __init__(self, noise_dim: int = 64, feature_dim: int = 32):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, feature_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        self.noise_dim = noise_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.noise_dim, device=device)
        return self.forward(z)


class BotDiscriminator(nn.Module):
    """
    Discriminator that learns to distinguish human from bot behavior.
    Trained adversarially against the generator.
    """
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdversarialBotGAN(nn.Module):
    """
    Full GAN for adversarial bot detection training.
    """
    def __init__(self, noise_dim: int = 64, feature_dim: int = 32, lr: float = 0.0002):
        super().__init__()
        
        self.generator = BotBehaviorGenerator(noise_dim, feature_dim)
        self.discriminator = BotDiscriminator(feature_dim)
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()

    def train_step(self, real_human: torch.Tensor, real_bot: torch.Tensor) -> Dict[str, float]:
        """
        One training step for both G and D.
        
        real_human: Authentic human behavioral features
        real_bot: Known bot behavioral features
        """
        batch_size = real_human.size(0)
        device = real_human.device
        
        # Labels
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()
        
        # Real human samples (label=1)
        d_real = self.discriminator(real_human)
        d_loss_real = self.criterion(d_real, real_label)
        
        # Real bot samples (label=0)
        d_bot = self.discriminator(real_bot)
        d_loss_bot = self.criterion(d_bot, fake_label)
        
        # Generated fake bot samples (label=0)
        fake_bot = self.generator.sample(batch_size, device)
        d_fake = self.discriminator(fake_bot.detach())
        d_loss_fake = self.criterion(d_fake, fake_label)
        
        d_loss = d_loss_real + 0.5 * (d_loss_bot + d_loss_fake)
        d_loss.backward()
        self.d_optimizer.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        self.g_optimizer.zero_grad()
        
        # Generator tries to fool discriminator
        fake_bot = self.generator.sample(batch_size, device)
        d_fake = self.discriminator(fake_bot)
        g_loss = self.criterion(d_fake, real_label)  # Generator wants D to output 1
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_real_acc": (d_real > 0.5).float().mean().item(),
            "d_fake_acc": (d_fake < 0.5).float().mean().item()
        }


class BotDetectionService:
    """
    Production bot detection using trained discriminator.
    """
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discriminator = BotDiscriminator(feature_dim=32).to(self.device)
        self.discriminator.eval()
        
        if model_path:
            self.discriminator.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.threshold = 0.5

    def extract_features(self, session: Dict) -> np.ndarray:
        """
        Extract behavioral features from session data.
        
        Features:
        - Mouse movement statistics (mean velocity, std, entropy)
        - Keystroke timing (mean dwell, flight time variance)
        - Click patterns (inter-click intervals)
        - Scroll behavior
        """
        features = np.zeros(32)
        
        # Mouse features (0-7)
        mouse = session.get("mouse", {})
        features[0] = mouse.get("mean_velocity", 0.5)
        features[1] = mouse.get("velocity_std", 0.2)
        features[2] = mouse.get("entropy", 0.5)
        features[3] = mouse.get("straightness", 0.8)
        features[4] = mouse.get("acceleration_changes", 10) / 100
        
        # Keystroke features (8-15)
        keys = session.get("keystrokes", {})
        features[8] = keys.get("mean_dwell", 100) / 200
        features[9] = keys.get("dwell_std", 30) / 100
        features[10] = keys.get("mean_flight", 150) / 300
        features[11] = keys.get("typing_speed", 50) / 100
        
        # Click features (16-23)
        clicks = session.get("clicks", {})
        features[16] = clicks.get("mean_interval", 500) / 1000
        features[17] = clicks.get("double_click_ratio", 0.1)
        
        # Scroll features (24-31)
        scroll = session.get("scroll", {})
        features[24] = scroll.get("total_distance", 1000) / 5000
        features[25] = scroll.get("direction_changes", 5) / 20
        
        return features

    def detect(self, session: Dict) -> Dict:
        """
        Detect if session behavior is from a bot.
        """
        features = self.extract_features(session)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            human_prob = self.discriminator(x).item()
        
        is_bot = human_prob < self.threshold
        
        return {
            "is_bot": is_bot,
            "human_probability": round(human_prob, 4),
            "confidence": abs(human_prob - 0.5) * 2,
            "risk_level": "high" if is_bot else "low"
        }


if __name__ == "__main__":
    # Test GAN initialization
    gan = AdversarialBotGAN()
    
    # Simulated training step
    real_human = torch.randn(16, 32)
    real_bot = torch.randn(16, 32) * 0.5  # Bots have lower variance
    
    metrics = gan.train_step(real_human, real_bot)
    print(f"Training metrics: {metrics}")
    
    # Test detector
    detector = BotDetectionService()
    result = detector.detect({"mouse": {"mean_velocity": 0.1}, "keystrokes": {"mean_dwell": 50}})
    print(f"Detection: {result}")
