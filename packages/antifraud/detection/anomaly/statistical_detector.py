"""
Fraud Detection Model using Isolation Forest
Detects statistical anomalies in voting patterns
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class VoteEvent:
    """Represents a single vote event with features for fraud detection"""
    voter_id: str
    election_id: str
    timestamp: float
    device_fingerprint: str
    ip_address: str
    latitude: float
    longitude: float
    session_duration_ms: int
    mouse_entropy: float
    keystroke_rhythm: float
    scroll_behavior: float
    time_to_submit_ms: int


@dataclass
class FraudScore:
    """Result of fraud detection analysis"""
    vote_event_id: str
    risk_score: float  # 0-100
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    anomaly_factors: List[str]
    recommendation: str  # ALLOW, CHALLENGE, BLOCK
    confidence: float


class StatisticalDetector:
    """
    Uses Isolation Forest for outlier detection in voting patterns.
    Trained on historical legitimate voting behavior.
    """
    
    def __init__(self, contamination: float = 0.01):
        """
        Args:
            contamination: Expected proportion of outliers (fraud attempts)
        """
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = [
            "session_duration_ms",
            "mouse_entropy",
            "keystroke_rhythm",
            "scroll_behavior",
            "time_to_submit_ms",
            "hour_of_day",
            "day_of_week",
        ]
    
    def extract_features(self, event: VoteEvent) -> np.ndarray:
        """Extract numerical features from a vote event"""
        from datetime import datetime
        
        dt = datetime.fromtimestamp(event.timestamp)
        
        return np.array([
            event.session_duration_ms,
            event.mouse_entropy,
            event.keystroke_rhythm,
            event.scroll_behavior,
            event.time_to_submit_ms,
            dt.hour,
            dt.weekday(),
        ])
    
    def train(self, historical_events: List[VoteEvent]) -> None:
        """
        Train the model on historical legitimate voting data.
        
        Args:
            historical_events: List of verified legitimate vote events
        """
        X = np.array([self.extract_features(e) for e in historical_events])
        self.model.fit(X)
        self.is_trained = True
        print(f"Model trained on {len(historical_events)} events")
    
    def predict(self, event: VoteEvent) -> FraudScore:
        """
        Predict fraud risk for a new vote event.
        
        Args:
            event: The vote event to analyze
            
        Returns:
            FraudScore with risk assessment
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        features = self.extract_features(event).reshape(1, -1)
        
        # Get anomaly score (-1 for outliers, 1 for inliers)
        prediction = self.model.predict(features)[0]
        
        # Get the raw score (lower = more anomalous)
        raw_score = self.model.decision_function(features)[0]
        
        # Convert to 0-100 risk score (inverted so higher = more risky)
        risk_score = max(0, min(100, 50 - raw_score * 50))
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
            recommendation = "BLOCK"
        elif risk_score >= 60:
            risk_level = "HIGH"
            recommendation = "CHALLENGE"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            recommendation = "CHALLENGE"
        else:
            risk_level = "LOW"
            recommendation = "ALLOW"
        
        # Identify anomaly factors
        anomaly_factors = self._identify_anomaly_factors(event, features[0])
        
        return FraudScore(
            vote_event_id=event.voter_id,
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            anomaly_factors=anomaly_factors,
            recommendation=recommendation,
            confidence=0.85  # Model confidence
        )
    
    def _identify_anomaly_factors(self, event: VoteEvent, features: np.ndarray) -> List[str]:
        """Identify which features contributed to anomaly detection"""
        factors = []
        
        # Check for specific anomalies
        if event.session_duration_ms < 5000:
            factors.append("SUSPICIOUSLY_FAST_SESSION")
        
        if event.mouse_entropy < 0.1:
            factors.append("LOW_MOUSE_ENTROPY")
        
        if event.keystroke_rhythm < 0.2:
            factors.append("UNNATURAL_TYPING_PATTERN")
        
        if event.time_to_submit_ms < 3000:
            factors.append("INSTANT_SUBMISSION")
        
        return factors
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        import joblib
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk"""
        import joblib
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")


# Benford's Law analyzer for vote count distributions
class BenfordsLawAnalyzer:
    """
    Checks if vote count distributions follow Benford's Law.
    Deviations may indicate fraud or manipulation.
    """
    
    EXPECTED_DISTRIBUTION = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }
    
    def analyze(self, vote_counts: List[int]) -> dict:
        """
        Analyze vote counts against Benford's Law.
        
        Args:
            vote_counts: List of vote counts (e.g., by precinct)
            
        Returns:
            Analysis results with chi-square test
        """
        # Extract first digits
        first_digits = []
        for count in vote_counts:
            if count > 0:
                first_digit = int(str(count)[0])
                first_digits.append(first_digit)
        
        if len(first_digits) < 30:
            return {"error": "Insufficient data for Benford's analysis"}
        
        # Calculate observed distribution
        observed = {i: 0 for i in range(1, 10)}
        for digit in first_digits:
            observed[digit] += 1
        
        total = len(first_digits)
        observed_pct = {k: v / total for k, v in observed.items()}
        
        # Chi-square test
        chi_square = sum(
            ((observed_pct[i] - self.EXPECTED_DISTRIBUTION[i]) ** 2) / self.EXPECTED_DISTRIBUTION[i]
            for i in range(1, 10)
        )
        
        # Critical value for df=8, alpha=0.05 is ~15.5
        is_suspicious = chi_square > 15.5
        
        return {
            "sample_size": total,
            "chi_square": round(chi_square, 4),
            "is_suspicious": is_suspicious,
            "observed_distribution": observed_pct,
            "expected_distribution": self.EXPECTED_DISTRIBUTION,
        }


if __name__ == "__main__":
    # Example usage
    print("Fraud Detection Model - Statistical Detector")
    print("=" * 50)
    
    # Create sample training data
    np.random.seed(42)
    training_events = [
        VoteEvent(
            voter_id=f"voter_{i}",
            election_id="WA_GOV_2024",
            timestamp=1699200000 + i * 60,
            device_fingerprint=f"device_{i % 1000}",
            ip_address=f"192.168.1.{i % 256}",
            latitude=47.6 + np.random.randn() * 0.1,
            longitude=-122.3 + np.random.randn() * 0.1,
            session_duration_ms=int(30000 + np.random.randn() * 10000),
            mouse_entropy=0.5 + np.random.randn() * 0.2,
            keystroke_rhythm=0.6 + np.random.randn() * 0.15,
            scroll_behavior=0.4 + np.random.randn() * 0.1,
            time_to_submit_ms=int(15000 + np.random.randn() * 5000),
        )
        for i in range(1000)
    ]
    
    detector = StatisticalDetector()
    detector.train(training_events)
    
    # Test with a suspicious event
    suspicious_event = VoteEvent(
        voter_id="suspicious_voter",
        election_id="WA_GOV_2024",
        timestamp=1699200000,
        device_fingerprint="unknown_device",
        ip_address="10.0.0.1",
        latitude=47.6,
        longitude=-122.3,
        session_duration_ms=1000,  # Very fast
        mouse_entropy=0.01,  # Almost no mouse movement
        keystroke_rhythm=0.05,  # Robotic
        scroll_behavior=0.0,  # No scrolling
        time_to_submit_ms=500,  # Instant
    )
    
    result = detector.predict(suspicious_event)
    print(f"\nSuspicious Event Analysis:")
    print(f"  Risk Score: {result.risk_score}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Recommendation: {result.recommendation}")
    print(f"  Anomaly Factors: {result.anomaly_factors}")
