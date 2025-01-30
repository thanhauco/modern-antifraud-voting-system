"""
AI Explainability Module for Fraud Detection Transparency
SHAP/LIME-based explanations for model decisions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import math

class ExplanationType(Enum):
    """Types of explanations"""
    SHAP = "shap"           # SHapley Additive exPlanations
    LIME = "lime"           # Local Interpretable Model-agnostic Explanations
    COUNTERFACTUAL = "counterfactual"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class FeatureContribution:
    """Contribution of a feature to prediction"""
    feature_name: str
    feature_value: Any
    contribution: float  # Positive = increases fraud score, negative = decreases
    importance_rank: int
    
    @property
    def direction(self) -> str:
        return "increases" if self.contribution > 0 else "decreases"


@dataclass
class Explanation:
    """Complete explanation for a model prediction"""
    prediction: float
    explanation_type: ExplanationType
    contributions: List[FeatureContribution]
    base_value: float
    confidence: float
    human_readable: str


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for fraud detection.
    
    Provides theoretically grounded feature attributions based on
    cooperative game theory.
    """
    
    def __init__(self, model, feature_names: List[str], background_data: np.ndarray = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.base_value = 0.5  # Would be computed from background
    
    def explain(self, instance: np.ndarray) -> Explanation:
        """
        Generate SHAP explanation for a single instance.
        
        In production, uses shap library with TreeExplainer/DeepExplainer.
        """
        # Simulate SHAP values (production uses actual SHAP library)
        num_features = len(self.feature_names)
        shap_values = self._approximate_shap_values(instance)
        
        # Get model prediction
        prediction = self._predict(instance)
        
        # Create feature contributions
        contributions = []
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            contributions.append(FeatureContribution(
                feature_name=self.feature_names[idx],
                feature_value=float(instance[idx]),
                contribution=float(shap_values[idx]),
                importance_rank=rank + 1
            ))
        
        # Generate human-readable explanation
        human_readable = self._generate_narrative(prediction, contributions[:5])
        
        return Explanation(
            prediction=prediction,
            explanation_type=ExplanationType.SHAP,
            contributions=contributions,
            base_value=self.base_value,
            confidence=min(1.0, abs(prediction - 0.5) * 2),
            human_readable=human_readable
        )
    
    def _approximate_shap_values(self, instance: np.ndarray) -> np.ndarray:
        """Approximate SHAP values using sampling (simplified)"""
        num_features = len(instance)
        shap_values = np.zeros(num_features)
        
        # Simplified: use feature value deviation from mean as proxy
        for i in range(num_features):
            # Higher deviation = higher contribution
            deviation = instance[i] - 0.5
            weight = random.uniform(0.1, 0.5)
            shap_values[i] = deviation * weight
        
        return shap_values
    
    def _predict(self, instance: np.ndarray) -> float:
        """Get model prediction"""
        if self.model:
            return float(self.model.predict(instance.reshape(1, -1))[0])
        # Simulated prediction
        return float(np.mean(instance) * 0.5 + random.uniform(0.2, 0.4))
    
    def _generate_narrative(self, prediction: float, top_contributions: List[FeatureContribution]) -> str:
        """Generate human-readable explanation"""
        risk_level = "high" if prediction > 0.7 else "medium" if prediction > 0.4 else "low"
        
        lines = [f"Fraud risk assessment: {risk_level.upper()} ({prediction:.1%})"]
        lines.append("\nKey factors:")
        
        for contrib in top_contributions[:3]:
            direction = "↑" if contrib.contribution > 0 else "↓"
            lines.append(f"  {direction} {contrib.feature_name}: {contrib.contribution:+.3f}")
        
        return "\n".join(lines)


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations).
    
    Creates local linear approximation of the model for interpretability.
    """
    
    def __init__(self, model, feature_names: List[str], num_samples: int = 1000):
        self.model = model
        self.feature_names = feature_names
        self.num_samples = num_samples
    
    def explain(self, instance: np.ndarray, num_features: int = 5) -> Explanation:
        """
        Generate LIME explanation by fitting local linear model.
        """
        # Sample around the instance
        perturbations = self._generate_perturbations(instance)
        
        # Get predictions for all perturbations
        predictions = self._predict_batch(perturbations)
        
        # Compute sample weights (closer = higher weight)
        weights = self._compute_weights(instance, perturbations)
        
        # Fit weighted linear regression
        coefficients = self._fit_local_model(perturbations, predictions, weights)
        
        # Create contributions
        contributions = []
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            contributions.append(FeatureContribution(
                feature_name=self.feature_names[idx],
                feature_value=float(instance[idx]),
                contribution=float(coefficients[idx] * instance[idx]),
                importance_rank=rank + 1
            ))
        
        prediction = float(np.mean(predictions))
        
        return Explanation(
            prediction=prediction,
            explanation_type=ExplanationType.LIME,
            contributions=contributions,
            base_value=float(np.mean(predictions)),
            confidence=float(np.std(predictions)),
            human_readable=self._generate_narrative(prediction, contributions[:num_features])
        )
    
    def _generate_perturbations(self, instance: np.ndarray) -> np.ndarray:
        """Generate perturbations around instance"""
        perturbations = []
        for _ in range(self.num_samples):
            perturbed = instance + np.random.normal(0, 0.1, len(instance))
            perturbed = np.clip(perturbed, 0, 1)
            perturbations.append(perturbed)
        return np.array(perturbations)
    
    def _predict_batch(self, instances: np.ndarray) -> np.ndarray:
        """Batch predictions"""
        if self.model:
            return self.model.predict(instances)
        return np.mean(instances, axis=1) * 0.5 + 0.25
    
    def _compute_weights(self, instance: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        """Compute sample weights based on distance"""
        distances = np.sqrt(np.sum((perturbations - instance) ** 2, axis=1))
        kernel_width = 0.75 * np.sqrt(len(instance))
        return np.exp(-(distances ** 2) / (kernel_width ** 2))
    
    def _fit_local_model(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Fit weighted linear regression (simplified)"""
        # Ridge regression with weighting
        W = np.diag(weights)
        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            coefficients = np.linalg.solve(XtWX + 0.01 * np.eye(X.shape[1]), XtWy)
        except:
            coefficients = np.random.uniform(-0.1, 0.1, X.shape[1])
        return coefficients
    
    def _generate_narrative(self, prediction: float, contributions: List[FeatureContribution]) -> str:
        risk = "HIGH RISK" if prediction > 0.7 else "MEDIUM RISK" if prediction > 0.4 else "LOW RISK"
        parts = [f"Assessment: {risk} ({prediction:.1%} fraud probability)"]
        parts.append("\nContributing factors:")
        for c in contributions:
            parts.append(f"  • {c.feature_name}: {c.contribution:+.3f}")
        return "\n".join(parts)


class FraudExplanationService:
    """
    Service for generating explanations for fraud detection decisions.
    """
    
    def __init__(self, model=None):
        self.feature_names = [
            "session_duration", "mouse_entropy", "keystroke_rhythm",
            "ip_risk_score", "device_fingerprint_match", "time_of_day",
            "geographic_distance", "previous_votes", "browser_anomaly",
            "network_latency"
        ]
        self.shap_explainer = SHAPExplainer(model, self.feature_names)
        self.lime_explainer = LIMEExplainer(model, self.feature_names)
    
    def explain(self, features: Dict[str, float], 
                method: ExplanationType = ExplanationType.SHAP) -> Explanation:
        """Generate explanation for fraud decision"""
        # Convert dict to array
        instance = np.array([features.get(f, 0.5) for f in self.feature_names])
        
        if method == ExplanationType.SHAP:
            return self.shap_explainer.explain(instance)
        else:
            return self.lime_explainer.explain(instance)
    
    def generate_audit_report(self, vote_id: str, features: Dict[str, float]) -> dict:
        """Generate comprehensive audit report"""
        shap_exp = self.explain(features, ExplanationType.SHAP)
        lime_exp = self.explain(features, ExplanationType.LIME)
        
        return {
            "vote_id": vote_id,
            "timestamp": "2024-01-15T10:30:00Z",
            "fraud_score": shap_exp.prediction,
            "risk_level": "high" if shap_exp.prediction > 0.7 else "medium" if shap_exp.prediction > 0.4 else "low",
            "explanations": {
                "shap": {
                    "narrative": shap_exp.human_readable,
                    "top_factors": [
                        {"name": c.feature_name, "contribution": c.contribution}
                        for c in shap_exp.contributions[:5]
                    ]
                },
                "lime": {
                    "narrative": lime_exp.human_readable
                }
            },
            "recommendation": self._get_recommendation(shap_exp.prediction),
            "auditor_notes": ""
        }
    
    def _get_recommendation(self, score: float) -> str:
        if score > 0.8:
            return "REJECT - Manual review required before processing"
        elif score > 0.6:
            return "HOLD - Additional verification recommended"
        elif score > 0.4:
            return "REVIEW - Flag for post-election audit"
        else:
            return "ACCEPT - Proceed with normal processing"


if __name__ == "__main__":
    print("=== AI Explainability for Fraud Detection ===\n")
    
    service = FraudExplanationService()
    
    # Example vote features
    features = {
        "session_duration": 0.8,
        "mouse_entropy": 0.3,
        "keystroke_rhythm": 0.7,
        "ip_risk_score": 0.9,
        "device_fingerprint_match": 0.4,
        "time_of_day": 0.2,
        "geographic_distance": 0.6,
        "previous_votes": 0.1,
        "browser_anomaly": 0.8,
        "network_latency": 0.5
    }
    
    # Get SHAP explanation
    explanation = service.explain(features, ExplanationType.SHAP)
    print("SHAP Explanation:")
    print(explanation.human_readable)
    print(f"\nTop contributors:")
    for c in explanation.contributions[:5]:
        print(f"  {c.importance_rank}. {c.feature_name}: {c.contribution:+.4f}")
    
    # Generate audit report
    print("\n" + "="*50)
    report = service.generate_audit_report("vote_12345", features)
    print(f"\nAudit Report:")
    print(f"  Vote ID: {report['vote_id']}")
    print(f"  Risk Level: {report['risk_level'].upper()}")
    print(f"  Recommendation: {report['recommendation']}")
