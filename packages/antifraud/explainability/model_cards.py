"""
Model Cards for ML Model Documentation and Governance
Standardized documentation for election ML models.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BiasCategory(Enum):
    DEMOGRAPHIC = "demographic"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"
    SOCIOECONOMIC = "socioeconomic"


@dataclass
class ModelMetrics:
    """Performance metrics for the model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    false_positive_rate: float
    false_negative_rate: float
    
    # Fairness metrics
    demographic_parity: Optional[float] = None
    equalized_odds: Optional[float] = None
    calibration_error: Optional[float] = None


@dataclass
class BiasAuditResult:
    """Result of bias audit for a demographic group"""
    category: BiasCategory
    group_name: str
    sample_size: int
    false_positive_rate: float
    false_negative_rate: float
    disparity_ratio: float
    passed: bool
    notes: str = ""


@dataclass
class ModelCard:
    """
    Model Card for ML model documentation.
    Based on Google's Model Cards for Model Reporting.
    """
    # Basic info
    model_name: str
    model_version: str
    model_type: str
    created_date: datetime
    last_updated: datetime
    
    # Purpose
    intended_use: str
    out_of_scope_uses: List[str]
    primary_users: List[str]
    
    # Technical details
    architecture: str
    input_format: str
    output_format: str
    training_data_description: str
    
    # Performance
    metrics: ModelMetrics
    evaluation_data_description: str
    
    # Bias and fairness
    bias_audits: List[BiasAuditResult] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    
    # Governance
    risk_level: RiskLevel = RiskLevel.MEDIUM
    owner: str = ""
    reviewers: List[str] = field(default_factory=list)
    approval_status: str = "pending"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "created_date": self.created_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "intended_use": self.intended_use,
            "out_of_scope_uses": self.out_of_scope_uses,
            "architecture": self.architecture,
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "f1_score": self.metrics.f1_score,
                "auc_roc": self.metrics.auc_roc,
                "fpr": self.metrics.false_positive_rate,
                "fnr": self.metrics.false_negative_rate
            },
            "bias_audits": [
                {
                    "category": ba.category.value,
                    "group": ba.group_name,
                    "passed": ba.passed,
                    "disparity": ba.disparity_ratio
                } for ba in self.bias_audits
            ],
            "risk_level": self.risk_level.value,
            "approval_status": self.approval_status
        }
    
    def generate_markdown(self) -> str:
        """Generate markdown documentation"""
        md = f"""# Model Card: {self.model_name}

## Model Details
- **Version:** {self.model_version}
- **Type:** {self.model_type}
- **Created:** {self.created_date.strftime('%Y-%m-%d')}
- **Risk Level:** {self.risk_level.value.upper()}
- **Status:** {self.approval_status.upper()}

## Intended Use
{self.intended_use}

### Primary Users
{chr(10).join(f'- {u}' for u in self.primary_users)}

### Out of Scope
{chr(10).join(f'- {u}' for u in self.out_of_scope_uses)}

## Technical Specifications
- **Architecture:** {self.architecture}
- **Input:** {self.input_format}
- **Output:** {self.output_format}

## Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {self.metrics.accuracy:.4f} |
| Precision | {self.metrics.precision:.4f} |
| Recall | {self.metrics.recall:.4f} |
| F1 Score | {self.metrics.f1_score:.4f} |
| AUC-ROC | {self.metrics.auc_roc:.4f} |
| False Positive Rate | {self.metrics.false_positive_rate:.4f} |
| False Negative Rate | {self.metrics.false_negative_rate:.4f} |

## Bias Audits
| Category | Group | Disparity | Status |
|----------|-------|-----------|--------|
{chr(10).join(f'| {ba.category.value} | {ba.group_name} | {ba.disparity_ratio:.2f} | {"✅ Pass" if ba.passed else "❌ Fail"} |' for ba in self.bias_audits)}

## Known Limitations
{chr(10).join(f'- {l}' for l in self.known_limitations)}

## Ethical Considerations
{chr(10).join(f'- {e}' for e in self.ethical_considerations)}

---
*Generated by Vote System Model Governance*
"""
        return md


class BiasAuditor:
    """
    Audits ML models for bias across demographic groups.
    """
    
    def __init__(self, model, threshold: float = 0.8):
        self.model = model
        self.threshold = threshold  # Disparity ratio threshold
    
    def audit_demographic_group(self, 
                                 group_name: str,
                                 features: Any,
                                 labels: Any,
                                 category: BiasCategory = BiasCategory.DEMOGRAPHIC) -> BiasAuditResult:
        """
        Audit model performance for a specific demographic group.
        """
        # Simulate predictions and metrics
        # In production, use actual predictions
        import random
        
        sample_size = len(features) if hasattr(features, '__len__') else 1000
        fpr = random.uniform(0.02, 0.08)
        fnr = random.uniform(0.05, 0.15)
        
        # Calculate disparity ratio compared to baseline
        baseline_fpr = 0.05
        disparity = fpr / baseline_fpr if baseline_fpr > 0 else 1.0
        
        passed = 0.8 <= disparity <= 1.25  # Within 20% of baseline
        
        return BiasAuditResult(
            category=category,
            group_name=group_name,
            sample_size=sample_size,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            disparity_ratio=disparity,
            passed=passed
        )
    
    def full_audit(self, test_data: Dict[str, Any]) -> List[BiasAuditResult]:
        """Run comprehensive bias audit"""
        results = []
        
        # Demographic groups to audit
        groups = [
            ("age_18_29", BiasCategory.DEMOGRAPHIC),
            ("age_30_44", BiasCategory.DEMOGRAPHIC),
            ("age_45_64", BiasCategory.DEMOGRAPHIC),
            ("age_65_plus", BiasCategory.DEMOGRAPHIC),
            ("urban", BiasCategory.GEOGRAPHIC),
            ("rural", BiasCategory.GEOGRAPHIC),
            ("suburban", BiasCategory.GEOGRAPHIC),
            ("morning_voters", BiasCategory.TEMPORAL),
            ("evening_voters", BiasCategory.TEMPORAL),
        ]
        
        for group_name, category in groups:
            result = self.audit_demographic_group(
                group_name, 
                test_data.get("features", []),
                test_data.get("labels", []),
                category
            )
            results.append(result)
        
        return results


class ModelGovernance:
    """
    Model governance system for election ML models.
    """
    
    def __init__(self):
        self.model_cards: Dict[str, ModelCard] = {}
        self.pending_reviews: List[str] = []
    
    def register_model(self, model_card: ModelCard) -> str:
        """Register a new model"""
        key = f"{model_card.model_name}:{model_card.model_version}"
        self.model_cards[key] = model_card
        self.pending_reviews.append(key)
        return key
    
    def approve_model(self, model_key: str, reviewer: str) -> bool:
        """Approve a model for production use"""
        if model_key not in self.model_cards:
            return False
        
        card = self.model_cards[model_key]
        
        # Check all bias audits passed
        all_audits_passed = all(ba.passed for ba in card.bias_audits)
        if not all_audits_passed:
            return False
        
        card.approval_status = "approved"
        card.reviewers.append(reviewer)
        
        if model_key in self.pending_reviews:
            self.pending_reviews.remove(model_key)
        
        return True
    
    def get_production_models(self) -> List[ModelCard]:
        """Get all approved models"""
        return [mc for mc in self.model_cards.values() if mc.approval_status == "approved"]


# Pre-configured model cards for vote system models
def create_fraud_detector_card() -> ModelCard:
    """Create model card for fraud detector"""
    metrics = ModelMetrics(
        accuracy=0.9823,
        precision=0.9456,
        recall=0.9312,
        f1_score=0.9384,
        auc_roc=0.9891,
        false_positive_rate=0.0234,
        false_negative_rate=0.0688,
        demographic_parity=0.92,
        equalized_odds=0.89
    )
    
    return ModelCard(
        model_name="VoteFraudDetector",
        model_version="2.3.1",
        model_type="Ensemble (XGBoost + Neural Network)",
        created_date=datetime(2024, 6, 15),
        last_updated=datetime(2024, 12, 1),
        intended_use="Real-time fraud detection for online vote submissions",
        out_of_scope_uses=[
            "Voter eligibility determination",
            "Vote content prediction",
            "Voter behavior profiling for non-security purposes"
        ],
        primary_users=["Election Security Team", "Fraud Analysts"],
        architecture="Ensemble: XGBoost (300 trees) + 3-layer MLP",
        input_format="32-dimensional behavioral feature vector",
        output_format="Fraud probability [0, 1]",
        training_data_description="2M labeled transactions from 2020-2023 elections",
        metrics=metrics,
        evaluation_data_description="500K held-out transactions from 2024 primaries",
        known_limitations=[
            "Reduced accuracy for first-time voters (no behavioral history)",
            "May not detect novel attack patterns not in training data",
            "Requires minimum 30 seconds of behavioral data"
        ],
        ethical_considerations=[
            "Model decisions are advisory only - human review required for rejections",
            "No demographic data used as direct features",
            "Threshold tuned to minimize false positives (voter disenfranchisement)"
        ],
        risk_level=RiskLevel.HIGH,
        owner="Election Security Team"
    )


if __name__ == "__main__":
    print("=== Model Cards & Governance ===\n")
    
    # Create model card
    card = create_fraud_detector_card()
    
    # Run bias audit
    auditor = BiasAuditor(model=None)
    card.bias_audits = auditor.full_audit({"features": [], "labels": []})
    
    # Register with governance
    governance = ModelGovernance()
    model_key = governance.register_model(card)
    
    # Print model card
    print(card.generate_markdown()[:2000])
    
    # Check audit results
    print("\n\nBias Audit Summary:")
    passed = sum(1 for ba in card.bias_audits if ba.passed)
    print(f"  Passed: {passed}/{len(card.bias_audits)}")
    
    # Try approval
    if governance.approve_model(model_key, "Security Lead"):
        print(f"\n✅ Model {model_key} approved for production")
    else:
        print(f"\n❌ Model {model_key} requires remediation")
