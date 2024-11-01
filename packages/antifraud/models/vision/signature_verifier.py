"""
Siamese Convolutional Neural Network for Signature Verification
Compares mail-in ballot signatures against voter registration records.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SiameseEncoder(nn.Module):
    """
    Shared CNN encoder for signature embedding.
    Uses a ResNet-inspired architecture.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings


class SiameseSignatureNet(nn.Module):
    """
    Siamese Network for one-shot signature verification.
    Uses contrastive loss for training.
    """
    def __init__(self, embedding_dim: int = 128, margin: float = 1.0):
        super().__init__()
        self.encoder = SiameseEncoder(embedding_dim)
        self.margin = margin

    def forward(self, anchor: torch.Tensor, comparison: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Siamese network.
        Returns embeddings for both inputs.
        """
        emb_anchor = self.encoder(anchor)
        emb_comparison = self.encoder(comparison)
        return emb_anchor, emb_comparison

    def get_similarity(self, anchor: torch.Tensor, comparison: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two signatures.
        Returns value in [0, 1] where 1 = identical.
        """
        emb_a, emb_c = self.forward(anchor, comparison)
        similarity = F.cosine_similarity(emb_a, emb_c)
        return (similarity + 1) / 2  # Normalize to [0, 1]

    def contrastive_loss(self, emb_a: torch.Tensor, emb_c: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss for training.
        label = 1 for same person, 0 for different.
        """
        euclidean_distance = F.pairwise_distance(emb_a, emb_c)
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


class SignatureVerifier:
    """
    Production-ready signature verification service.
    """
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseSignatureNet().to(self.device)
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.threshold = 0.85  # Similarity threshold for match

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess signature image for model input.
        Assumes grayscale 128x64 input.
        """
        import io
        from PIL import Image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img = Image.open(io.BytesIO(image_bytes))
        return transform(img).unsqueeze(0).to(self.device)

    def verify(self, ballot_sig: bytes, reference_sig: bytes) -> dict:
        """
        Verify if ballot signature matches reference.
        Returns match result and confidence score.
        """
        with torch.no_grad():
            sig1 = self.preprocess(ballot_sig)
            sig2 = self.preprocess(reference_sig)
            
            similarity = self.model.get_similarity(sig1, sig2).item()
            
            return {
                "match": similarity >= self.threshold,
                "confidence": round(similarity, 4),
                "requires_manual_review": 0.7 <= similarity < self.threshold,
                "model_version": "siamese-cnn-v1"
            }


if __name__ == "__main__":
    # Test initialization
    model = SiameseSignatureNet()
    dummy_sig = torch.randn(1, 1, 64, 128)
    emb_a, emb_b = model(dummy_sig, dummy_sig)
    print(f"Embedding shape: {emb_a.shape}")
    print(f"Self-similarity: {model.get_similarity(dummy_sig, dummy_sig).item():.4f}")
