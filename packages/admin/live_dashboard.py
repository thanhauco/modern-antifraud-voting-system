"""
Live Dashboard WebSocket Service
Real-time updates for admin dashboards.
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Set, Callable, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class DashboardMetricType(Enum):
    """Types of dashboard metrics"""
    VOTE_COUNT = "vote_count"
    FRAUD_SCORE = "fraud_score"
    ALERT_COUNT = "alert_count"
    VELOCITY = "velocity"
    TURNOUT = "turnout"
    SYSTEM_HEALTH = "system_health"


@dataclass
class DashboardUpdate:
    """A dashboard update message"""
    metric_type: str
    value: float
    timestamp: float
    jurisdiction: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class LiveMetrics:
    """Real-time metrics snapshot"""
    total_votes: int
    votes_per_minute: float
    fraud_alerts_active: int
    fraud_alerts_resolved: int
    avg_fraud_score: float
    turnout_percentage: float
    jurisdictions_reporting: int
    system_status: str
    timestamp: float


class MetricsAggregator:
    """
    Aggregates metrics from multiple sources for dashboard.
    """
    
    def __init__(self):
        self.vote_counts: Dict[str, int] = {}
        self.fraud_scores: List[float] = []
        self.alerts_active = 0
        self.alerts_resolved = 0
        self.registered_voters = 5000000  # 5M registered
        self.last_update = time.time()
    
    def record_vote(self, jurisdiction: str):
        """Record a vote"""
        self.vote_counts[jurisdiction] = self.vote_counts.get(jurisdiction, 0) + 1
        self.last_update = time.time()
    
    def record_fraud_score(self, score: float):
        """Record a fraud score"""
        self.fraud_scores.append(score)
        if len(self.fraud_scores) > 10000:
            self.fraud_scores = self.fraud_scores[-10000:]
    
    def record_alert(self, resolved: bool = False):
        """Record an alert"""
        if resolved:
            self.alerts_resolved += 1
        else:
            self.alerts_active += 1
    
    def resolve_alert(self):
        """Mark an alert as resolved"""
        if self.alerts_active > 0:
            self.alerts_active -= 1
            self.alerts_resolved += 1
    
    def get_metrics(self) -> LiveMetrics:
        """Get current metrics snapshot"""
        total_votes = sum(self.vote_counts.values())
        avg_score = sum(self.fraud_scores) / len(self.fraud_scores) if self.fraud_scores else 0.0
        
        return LiveMetrics(
            total_votes=total_votes,
            votes_per_minute=len(self.fraud_scores),  # Simplified
            fraud_alerts_active=self.alerts_active,
            fraud_alerts_resolved=self.alerts_resolved,
            avg_fraud_score=avg_score,
            turnout_percentage=(total_votes / self.registered_voters) * 100,
            jurisdictions_reporting=len(self.vote_counts),
            system_status="healthy" if self.alerts_active < 10 else "degraded",
            timestamp=time.time()
        )


class WebSocketConnection:
    """Simulated WebSocket connection"""
    
    def __init__(self, connection_id: str, user_id: str):
        self.connection_id = connection_id
        self.user_id = user_id
        self.subscriptions: Set[str] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.connected = True
    
    async def send(self, data: Dict):
        """Send data to client"""
        if self.connected:
            await self.message_queue.put(json.dumps(data))
    
    def subscribe(self, topic: str):
        """Subscribe to topic"""
        self.subscriptions.add(topic)
    
    def unsubscribe(self, topic: str):
        """Unsubscribe from topic"""
        self.subscriptions.discard(topic)
    
    def disconnect(self):
        """Disconnect"""
        self.connected = False


class LiveDashboardService:
    """
    WebSocket service for real-time dashboard updates.
    
    Features:
    - Sub-second metric updates
    - Topic-based subscriptions
    - Multi-jurisdiction support
    - Automatic reconnection handling
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.metrics = MetricsAggregator()
        self.topics = {
            "metrics": set(),
            "alerts": set(),
            "votes": set(),
        }
        self.running = False
    
    async def connect(self, connection_id: str, user_id: str) -> WebSocketConnection:
        """Handle new connection"""
        conn = WebSocketConnection(connection_id, user_id)
        self.connections[connection_id] = conn
        
        # Send initial state
        await conn.send({
            "type": "connected",
            "connection_id": connection_id,
            "timestamp": time.time()
        })
        
        return conn
    
    def disconnect(self, connection_id: str):
        """Handle disconnection"""
        if connection_id in self.connections:
            conn = self.connections[connection_id]
            conn.disconnect()
            
            # Remove from all topics
            for topic_subs in self.topics.values():
                topic_subs.discard(connection_id)
            
            del self.connections[connection_id]
    
    def subscribe(self, connection_id: str, topic: str):
        """Subscribe connection to topic"""
        if topic in self.topics and connection_id in self.connections:
            self.topics[topic].add(connection_id)
            self.connections[connection_id].subscribe(topic)
    
    async def broadcast(self, topic: str, data: Dict):
        """Broadcast to all subscribers of a topic"""
        if topic not in self.topics:
            return
        
        message = {
            "topic": topic,
            "data": data,
            "timestamp": time.time()
        }
        
        for conn_id in self.topics[topic]:
            conn = self.connections.get(conn_id)
            if conn and conn.connected:
                await conn.send(message)
    
    async def broadcast_metrics(self):
        """Broadcast current metrics to subscribers"""
        metrics = self.metrics.get_metrics()
        await self.broadcast("metrics", asdict(metrics))
    
    async def broadcast_alert(self, alert: Dict):
        """Broadcast new alert"""
        await self.broadcast("alerts", alert)
        self.metrics.record_alert()
    
    async def broadcast_vote(self, vote_data: Dict):
        """Broadcast vote (anonymized)"""
        jurisdiction = vote_data.get("jurisdiction", "unknown")
        self.metrics.record_vote(jurisdiction)
        self.metrics.record_fraud_score(vote_data.get("fraud_score", 0.0))
        
        await self.broadcast("votes", {
            "jurisdiction": jurisdiction,
            "timestamp": time.time(),
            "fraud_score": vote_data.get("fraud_score", 0.0)
        })
    
    async def start_metrics_broadcast(self, interval: float = 1.0):
        """Start periodic metrics broadcast"""
        self.running = True
        while self.running:
            await self.broadcast_metrics()
            await asyncio.sleep(interval)
    
    def stop(self):
        """Stop the service"""
        self.running = False
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.connections),
            "by_topic": {topic: len(subs) for topic, subs in self.topics.items()},
            "active": sum(1 for c in self.connections.values() if c.connected)
        }


class DashboardWidgetData:
    """
    Pre-computed widget data for dashboard.
    """
    
    def __init__(self, metrics: MetricsAggregator):
        self.metrics = metrics
    
    def get_vote_map_data(self) -> Dict:
        """Get data for geographic vote map"""
        return {
            "jurisdictions": [
                {"id": j, "votes": v, "color": self._get_heat_color(v)}
                for j, v in self.metrics.vote_counts.items()
            ],
            "total": sum(self.metrics.vote_counts.values())
        }
    
    def get_velocity_chart_data(self, points: int = 60) -> List[Dict]:
        """Get data for velocity chart (votes per minute)"""
        # Simulated time series
        now = time.time()
        return [
            {
                "timestamp": now - (points - i) * 60,
                "value": random.randint(100, 500)
            }
            for i in range(points)
        ]
    
    def get_fraud_distribution_data(self) -> Dict:
        """Get fraud score distribution for histogram"""
        scores = self.metrics.fraud_scores
        if not scores:
            return {"buckets": [], "counts": []}
        
        buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts = [0] * (len(buckets) - 1)
        
        for score in scores:
            for i in range(len(buckets) - 1):
                if buckets[i] <= score < buckets[i + 1]:
                    counts[i] += 1
                    break
        
        return {
            "buckets": [f"{buckets[i]:.1f}-{buckets[i+1]:.1f}" for i in range(len(buckets)-1)],
            "counts": counts
        }
    
    def get_alert_timeline_data(self, hours: int = 24) -> List[Dict]:
        """Get alert timeline data"""
        now = time.time()
        return [
            {
                "hour": i,
                "timestamp": now - (hours - i) * 3600,
                "count": random.randint(0, 10)
            }
            for i in range(hours)
        ]
    
    def _get_heat_color(self, value: int) -> str:
        """Get heat map color based on value"""
        if value > 100000:
            return "#ff4444"
        elif value > 50000:
            return "#ff8844"
        elif value > 10000:
            return "#ffcc44"
        else:
            return "#44cc44"


if __name__ == "__main__":
    async def main():
        print("=== Live Dashboard Service ===\n")
        
        service = LiveDashboardService()
        
        # Simulate connections
        conn1 = await service.connect("conn1", "admin1")
        conn2 = await service.connect("conn2", "admin2")
        
        # Subscribe to topics
        service.subscribe("conn1", "metrics")
        service.subscribe("conn1", "alerts")
        service.subscribe("conn2", "votes")
        
        print(f"Connections: {service.get_connection_stats()}")
        
        # Simulate activity
        for i in range(20):
            await service.broadcast_vote({
                "jurisdiction": random.choice(["WA", "OR", "CA"]),
                "fraud_score": random.uniform(0.1, 0.4)
            })
        
        # Broadcast metrics
        await service.broadcast_metrics()
        
        # Get widget data
        widgets = DashboardWidgetData(service.metrics)
        print(f"\nVote Map: {widgets.get_vote_map_data()}")
        print(f"Fraud Distribution: {widgets.get_fraud_distribution_data()}")
        
        # Check queued messages
        print(f"\nConn1 has {conn1.message_queue.qsize()} messages queued")
    
    asyncio.run(main())
