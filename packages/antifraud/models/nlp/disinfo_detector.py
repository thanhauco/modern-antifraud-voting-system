"""
BERT-based Disinformation Detection
Fine-tuned transformer for detecting election misinformation and influence campaigns.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re

@dataclass
class DisinfoConfig:
    model_name: str = "roberta-base"
    num_labels: int = 4  # [authentic, misleading, coordinated, spam]
    max_length: int = 256
    dropout: float = 0.1


class DisinformationClassifier(nn.Module):
    """
    Transformer-based multi-label classifier for election disinformation.
    
    Labels:
    - authentic: Genuine voter communication
    - misleading: Contains false or misleading information
    - coordinated: Part of organized influence campaign
    - spam: Automated/repetitive content
    """
    def __init__(self, config: DisinfoConfig):
        super().__init__()
        self.config = config
        
        # Simulating BERT-like architecture without loading actual model
        # In production, use: self.encoder = AutoModel.from_pretrained(config.model_name)
        
        self.embedding = nn.Embedding(30522, 768)  # BERT vocab size
        self.position_embedding = nn.Embedding(512, 768)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, 
            nhead=12, 
            dim_feedforward=3072,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_labels)
        )
        
        self.label_names = ["authentic", "misleading", "coordinated", "spam"]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.embedding(input_ids) + self.position_embedding(position_ids)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Encode
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pool ([CLS] token equivalent - first token)
        pooled = encoded[:, 0, :]
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)  # Multi-label, not softmax
        
        return {name: probs[:, i] for i, name in enumerate(self.label_names)}


class SimpleTokenizer:
    """
    Minimal tokenizer for demonstration.
    In production, use: AutoTokenizer.from_pretrained("roberta-base")
    """
    def __init__(self, vocab_size: int = 30522, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        # Simple hash-based tokenization
        words = re.findall(r'\w+', text.lower())
        token_ids = [self.cls_token_id]
        
        for word in words[:self.max_length - 2]:
            token_ids.append(hash(word) % (self.vocab_size - 3) + 3)
        
        token_ids.append(self.sep_token_id)
        
        # Padding
        attention_mask = [1] * len(token_ids)
        padding_length = self.max_length - len(token_ids)
        
        token_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            "input_ids": torch.tensor([token_ids]),
            "attention_mask": torch.tensor([attention_mask])
        }


class DisinfoDetector:
    """
    Production disinformation detection service.
    """
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = DisinfoConfig()
        self.model = DisinformationClassifier(self.config).to(self.device)
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.tokenizer = SimpleTokenizer(max_length=self.config.max_length)
        
        # Thresholds for each label
        self.thresholds = {
            "authentic": 0.5,
            "misleading": 0.6,
            "coordinated": 0.7,
            "spam": 0.5
        }

    def analyze(self, text: str) -> Dict:
        """
        Analyze text for disinformation signals.
        """
        # Tokenize
        inputs = self.tokenizer.encode(text)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            probs = self.model.predict_proba(input_ids, attention_mask)
        
        # Convert to dict
        scores = {k: v.item() for k, v in probs.items()}
        
        # Determine labels above threshold
        detected_labels = [k for k, v in scores.items() if v >= self.thresholds[k]]
        
        # Risk assessment
        if "coordinated" in detected_labels:
            risk = "critical"
        elif "misleading" in detected_labels:
            risk = "high"
        elif "spam" in detected_labels:
            risk = "medium"
        else:
            risk = "low"
        
        return {
            "text_length": len(text),
            "scores": scores,
            "detected_labels": detected_labels,
            "risk_level": risk,
            "requires_review": risk in ["critical", "high"]
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts efficiently"""
        return [self.analyze(t) for t in texts]


# Influence Campaign Pattern Detector
class CampaignPatternAnalyzer:
    """
    Detects coordinated influence campaigns across multiple messages.
    """
    def __init__(self):
        self.message_fingerprints = {}
        self.timing_patterns = []
        
    def add_message(self, message_id: str, text: str, timestamp: float, source_ip: str):
        # Create content fingerprint
        words = set(re.findall(r'\w{4,}', text.lower()))
        fingerprint = frozenset(words)
        
        # Track similar messages
        for fid, (fp, sources) in self.message_fingerprints.items():
            similarity = len(fingerprint & fp) / max(len(fingerprint | fp), 1)
            if similarity > 0.7:
                sources.add(source_ip)
                return {"coordinated": True, "cluster_id": fid, "cluster_size": len(sources)}
        
        # New fingerprint
        self.message_fingerprints[message_id] = (fingerprint, {source_ip})
        return {"coordinated": False}


if __name__ == "__main__":
    # Test model
    config = DisinfoConfig()
    model = DisinformationClassifier(config)
    
    dummy_input = torch.randint(0, 30000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    
    logits = model(dummy_input, dummy_mask)
    print(f"Output shape: {logits.shape}")
    
    # Test detector
    detector = DisinfoDetector()
    result = detector.analyze("Vote now! The election is being stolen! Share this before they delete it!")
    print(f"Analysis: {result}")
