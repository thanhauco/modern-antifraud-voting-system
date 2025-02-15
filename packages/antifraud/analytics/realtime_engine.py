"""
Real-Time Fraud Analytics Engine
Streaming analytics for live fraud detection and monitoring.
"""

import asyncio
import time
import random
import hashlib
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class FraudType(Enum):
    """Types of detected fraud"""
    BOT_ACTIVITY = "bot_activity"
    VELOCITY_ANOMALY = "velocity_anomaly"
    GEOGRAPHIC_IMPOSSIBLE = "geographic_impossible"
    SIGNATURE_MISMATCH = "signature_mismatch"
    DEVICE_FINGERPRINT = "device_fingerprint"
    COORDINATED_ATTACK = "coordinated_attack"
    REPLAY_ATTACK = "replay_attack"
    INJECTION_ATTEMPT = "injection_attempt"


@dataclass
class FraudAlert:
    """A fraud alert from the analytics engine"""
    alert_id: str
    timestamp: float
    fraud_type: FraudType
    severity: AlertSeverity
    vote_id: str
    voter_hash: str
    score: float
    details: Dict[str, Any]
    jurisdiction: str
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AnalyticsWindow:
    """Sliding window for time-series analytics"""
    window_size_seconds: int
    data_points: deque = field(default_factory=deque)
    
    def add(self, value: float, timestamp: float = None):
        ts = timestamp or time.time()
        self.data_points.append((ts, value))
        self._cleanup(ts)
    
    def _cleanup(self, current_time: float):
        cutoff = current_time - self.window_size_seconds
        while self.data_points and self.data_points[0][0] < cutoff:
            self.data_points.popleft()
    
    def get_values(self) -> List[float]:
        return [v for _, v in self.data_points]
    
    def mean(self) -> float:
        values = self.get_values()
        return statistics.mean(values) if values else 0.0
    
    def std(self) -> float:
        values = self.get_values()
        return statistics.stdev(values) if len(values) > 1 else 0.0
    
    def count(self) -> int:
        return len(self.data_points)


class RealTimeFraudAnalytics:
    """
    Real-time streaming fraud analytics engine.
    
    Features:
    - Sub-second latency fraud detection
    - Sliding window anomaly detection
    - Geographic velocity analysis
    - Pattern correlation across votes
    - Automatic threshold adjustment
    """
    
    def __init__(self):
        # Sliding windows for different metrics
        self.vote_velocity = AnalyticsWindow(window_size_seconds=60)
        self.fraud_score_window = AnalyticsWindow(window_size_seconds=300)
        self.jurisdiction_windows: Dict[str, AnalyticsWindow] = {}
        
        # Alerts
        self.alerts: List[FraudAlert] = []
        self.alert_handlers: List[Callable[[FraudAlert], None]] = []
        
        # Thresholds
        self.thresholds = {
            "fraud_score": 0.7,
            "velocity_per_minute": 1000,
            "velocity_std_multiplier": 3.0,
            "geographic_velocity_kmh": 500,  # Impossible travel speed
        }
        
        # State
        self.total_votes_analyzed = 0
        self.total_alerts_generated = 0
        self.recent_voter_locations: Dict[str, tuple] = {}
        self.recent_voter_times: Dict[str, float] = {}
        
    async def analyze_vote(self, vote_data: Dict) -> Dict:
        """
        Analyze a vote in real-time for fraud indicators.
        
        Returns analysis result with fraud score and any alerts.
        """
        start_time = time.time()
        self.total_votes_analyzed += 1
        
        vote_id = vote_data.get("vote_id", f"vote_{self.total_votes_analyzed}")
        voter_hash = hashlib.sha256(vote_data.get("voter_id", "").encode()).hexdigest()[:16]
        jurisdiction = vote_data.get("jurisdiction", "unknown")
        
        # Initialize jurisdiction window if needed
        if jurisdiction not in self.jurisdiction_windows:
            self.jurisdiction_windows[jurisdiction] = AnalyticsWindow(60)
        
        # Record vote for velocity tracking
        self.vote_velocity.add(1.0)
        self.jurisdiction_windows[jurisdiction].add(1.0)
        
        # Run all fraud checks
        fraud_checks = await asyncio.gather(
            self._check_fraud_score(vote_data),
            self._check_velocity_anomaly(vote_data, jurisdiction),
            self._check_geographic_velocity(vote_data, voter_hash),
            self._check_bot_indicators(vote_data),
            self._check_replay_attack(vote_data),
        )
        
        # Aggregate results
        alerts = []
        max_score = 0.0
        
        for check_result in fraud_checks:
            if check_result["detected"]:
                alert = self._create_alert(
                    vote_id=vote_id,
                    voter_hash=voter_hash,
                    jurisdiction=jurisdiction,
                    fraud_type=check_result["type"],
                    severity=check_result["severity"],
                    score=check_result["score"],
                    details=check_result["details"]
                )
                alerts.append(alert)
                max_score = max(max_score, check_result["score"])
        
        # Record fraud score
        self.fraud_score_window.add(max_score)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "vote_id": vote_id,
            "analyzed": True,
            "fraud_score": max_score,
            "alerts_generated": len(alerts),
            "alerts": [a.alert_id for a in alerts],
            "latency_ms": latency_ms,
            "recommendation": self._get_recommendation(max_score)
        }
    
    async def _check_fraud_score(self, vote_data: Dict) -> Dict:
        """Check ML-based fraud score"""
        # Simulate ML model call
        features = vote_data.get("features", {})
        score = features.get("fraud_score", random.uniform(0.1, 0.4))
        
        detected = score > self.thresholds["fraud_score"]
        
        return {
            "detected": detected,
            "type": FraudType.BOT_ACTIVITY,
            "severity": AlertSeverity.CRITICAL if score > 0.9 else AlertSeverity.HIGH,
            "score": score,
            "details": {"ml_score": score, "threshold": self.thresholds["fraud_score"]}
        }
    
    async def _check_velocity_anomaly(self, vote_data: Dict, jurisdiction: str) -> Dict:
        """Check for abnormal voting velocity"""
        votes_per_minute = self.vote_velocity.count()
        mean = self.vote_velocity.mean()
        std = self.vote_velocity.std()
        
        # Z-score based anomaly
        if std > 0:
            z_score = (votes_per_minute - mean) / std
        else:
            z_score = 0
        
        detected = (votes_per_minute > self.thresholds["velocity_per_minute"] or 
                    z_score > self.thresholds["velocity_std_multiplier"])
        
        return {
            "detected": detected,
            "type": FraudType.VELOCITY_ANOMALY,
            "severity": AlertSeverity.WARNING if z_score < 4 else AlertSeverity.HIGH,
            "score": min(1.0, z_score / 5) if detected else 0.0,
            "details": {
                "votes_per_minute": votes_per_minute,
                "z_score": z_score,
                "jurisdiction": jurisdiction
            }
        }
    
    async def _check_geographic_velocity(self, vote_data: Dict, voter_hash: str) -> Dict:
        """Check for impossible geographic velocity (voter can't travel that fast)"""
        current_location = vote_data.get("location", {})
        current_time = time.time()
        
        if voter_hash in self.recent_voter_locations:
            prev_location = self.recent_voter_locations[voter_hash]
            prev_time = self.recent_voter_times[voter_hash]
            
            # Calculate distance (simplified)
            lat1, lon1 = prev_location.get("lat", 0), prev_location.get("lon", 0)
            lat2, lon2 = current_location.get("lat", 0), current_location.get("lon", 0)
            
            distance_km = abs(lat2 - lat1) * 111 + abs(lon2 - lon1) * 85  # Rough estimate
            time_hours = (current_time - prev_time) / 3600
            
            if time_hours > 0:
                velocity_kmh = distance_km / time_hours
                detected = velocity_kmh > self.thresholds["geographic_velocity_kmh"]
                
                if detected:
                    return {
                        "detected": True,
                        "type": FraudType.GEOGRAPHIC_IMPOSSIBLE,
                        "severity": AlertSeverity.CRITICAL,
                        "score": min(1.0, velocity_kmh / 1000),
                        "details": {
                            "velocity_kmh": velocity_kmh,
                            "distance_km": distance_km,
                            "time_hours": time_hours
                        }
                    }
        
        # Update location tracking
        self.recent_voter_locations[voter_hash] = current_location
        self.recent_voter_times[voter_hash] = current_time
        
        return {"detected": False, "type": None, "severity": None, "score": 0.0, "details": {}}
    
    async def _check_bot_indicators(self, vote_data: Dict) -> Dict:
        """Check behavioral indicators of bot activity"""
        features = vote_data.get("features", {})
        
        mouse_entropy = features.get("mouse_entropy", 0.5)
        keystroke_rhythm = features.get("keystroke_rhythm", 0.5)
        session_duration = features.get("session_duration", 60)
        
        # Bot indicators: low entropy, regular rhythm, fast session
        bot_score = 0.0
        if mouse_entropy < 0.2:
            bot_score += 0.4
        if keystroke_rhythm > 0.9:  # Too regular
            bot_score += 0.3
        if session_duration < 10:  # Less than 10 seconds
            bot_score += 0.3
        
        detected = bot_score > 0.5
        
        return {
            "detected": detected,
            "type": FraudType.BOT_ACTIVITY,
            "severity": AlertSeverity.HIGH,
            "score": bot_score,
            "details": {
                "mouse_entropy": mouse_entropy,
                "keystroke_rhythm": keystroke_rhythm,
                "session_duration": session_duration
            }
        }
    
    async def _check_replay_attack(self, vote_data: Dict) -> Dict:
        """Check for replay attacks (duplicate vote submissions)"""
        # In production, check against seen vote hashes
        vote_hash = vote_data.get("vote_hash", "")
        
        # Simplified check - would use bloom filter or cache in production
        detected = False
        
        return {
            "detected": detected,
            "type": FraudType.REPLAY_ATTACK,
            "severity": AlertSeverity.CRITICAL,
            "score": 1.0 if detected else 0.0,
            "details": {"vote_hash": vote_hash}
        }
    
    def _create_alert(self, vote_id: str, voter_hash: str, jurisdiction: str,
                      fraud_type: FraudType, severity: AlertSeverity,
                      score: float, details: Dict) -> FraudAlert:
        """Create and dispatch a fraud alert"""
        alert = FraudAlert(
            alert_id=f"alert_{hashlib.sha256(f'{vote_id}{time.time()}'.encode()).hexdigest()[:12]}",
            timestamp=time.time(),
            fraud_type=fraud_type,
            severity=severity,
            vote_id=vote_id,
            voter_hash=voter_hash,
            score=score,
            details=details,
            jurisdiction=jurisdiction
        )
        
        self.alerts.append(alert)
        self.total_alerts_generated += 1
        
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception:
                pass
        
        return alert
    
    def _get_recommendation(self, score: float) -> str:
        """Get action recommendation based on fraud score"""
        if score >= 0.9:
            return "REJECT"
        elif score >= 0.7:
            return "MANUAL_REVIEW"
        elif score >= 0.4:
            return "FLAG_FOR_AUDIT"
        else:
            return "ACCEPT"
    
    def register_alert_handler(self, handler: Callable[[FraudAlert], None]):
        """Register a callback for new alerts"""
        self.alert_handlers.append(handler)
    
    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for dashboard display"""
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]
        
        return {
            "total_votes_analyzed": self.total_votes_analyzed,
            "total_alerts": self.total_alerts_generated,
            "alerts_last_hour": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            "current_velocity": self.vote_velocity.count(),
            "avg_fraud_score": self.fraud_score_window.mean(),
            "unresolved_alerts": len([a for a in self.alerts if not a.resolved]),
            "jurisdictions_active": len(self.jurisdiction_windows)
        }


class AlertManager:
    """
    Manages fraud alerts and escalation.
    """
    
    def __init__(self):
        self.analytics = RealTimeFraudAnalytics()
        self.escalation_rules: List[Dict] = []
        self.notification_channels: List[str] = ["email", "slack", "pagerduty"]
        
        # Register alert handler
        self.analytics.register_alert_handler(self._on_alert)
    
    def _on_alert(self, alert: FraudAlert):
        """Handle new alert"""
        if alert.severity == AlertSeverity.CRITICAL:
            self._escalate(alert)
    
    def _escalate(self, alert: FraudAlert):
        """Escalate critical alert"""
        # In production, send to notification channels
        print(f"🚨 CRITICAL ALERT: {alert.fraud_type.value} - {alert.vote_id}")
    
    def acknowledge_alert(self, alert_id: str, admin_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.analytics.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str, admin_id: str, resolution: str) -> bool:
        """Resolve an alert"""
        for alert in self.analytics.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.details["resolution"] = resolution
                alert.details["resolved_by"] = admin_id
                return True
        return False
    
    def get_alerts(self, 
                   severity: AlertSeverity = None,
                   resolved: bool = None,
                   limit: int = 100) -> List[FraudAlert]:
        """Get filtered alerts"""
        alerts = self.analytics.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return alerts[-limit:]


if __name__ == "__main__":
    async def main():
        print("=== Real-Time Fraud Analytics ===\n")
        
        manager = AlertManager()
        analytics = manager.analytics
        
        # Simulate vote stream
        print("Analyzing vote stream...")
        for i in range(100):
            vote_data = {
                "vote_id": f"vote_{i}",
                "voter_id": f"voter_{i % 20}",
                "jurisdiction": random.choice(["WA", "OR", "CA"]),
                "location": {"lat": 47.6 + random.uniform(-0.1, 0.1), "lon": -122.3 + random.uniform(-0.1, 0.1)},
                "features": {
                    "fraud_score": random.uniform(0.1, 0.5) if random.random() > 0.05 else random.uniform(0.7, 0.95),
                    "mouse_entropy": random.uniform(0.3, 0.8),
                    "keystroke_rhythm": random.uniform(0.4, 0.7),
                    "session_duration": random.uniform(30, 120)
                }
            }
            
            result = await analytics.analyze_vote(vote_data)
            
            if result["alerts_generated"] > 0:
                print(f"  ⚠️ Vote {i}: {result['alerts_generated']} alerts (score: {result['fraud_score']:.2f})")
        
        # Dashboard metrics
        metrics = analytics.get_dashboard_metrics()
        print(f"\n📊 Dashboard Metrics:")
        print(f"  Total analyzed: {metrics['total_votes_analyzed']}")
        print(f"  Total alerts: {metrics['total_alerts']}")
        print(f"  Critical: {metrics['critical_alerts']}")
        print(f"  Avg fraud score: {metrics['avg_fraud_score']:.3f}")
    
    asyncio.run(main())
