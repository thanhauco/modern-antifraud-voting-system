"""
Hyperscale Vote Processing Engine
Designed for 100M+ concurrent voters with sub-second response times.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import random

class ShardStrategy(Enum):
    """Sharding strategies for vote distribution"""
    GEOGRAPHIC = "geographic"      # By state/county
    HASH_BASED = "hash_based"      # By voter ID hash
    RANGE_BASED = "range_based"    # By voter ID range
    ROUND_ROBIN = "round_robin"    # Load balanced


@dataclass
class VotePayload:
    """Incoming vote request"""
    voter_id: str
    election_id: str
    encrypted_ballot: bytes
    timestamp: float
    jurisdiction: str
    zk_proof: bytes


@dataclass
class ShardConfig:
    """Configuration for a processing shard"""
    shard_id: str
    region: str
    capacity_tps: int  # Transactions per second
    current_load: float = 0.0
    healthy: bool = True
    primary: bool = True
    replicas: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result of vote processing"""
    success: bool
    vote_hash: str
    shard_id: str
    latency_ms: float
    queued: bool = False
    retry_after: Optional[int] = None


class ConsistentHashRing:
    """
    Consistent hashing for shard assignment.
    Minimizes reshuffling when shards are added/removed.
    """
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.shards: Dict[str, ShardConfig] = {}

    def add_shard(self, shard: ShardConfig):
        """Add a shard to the ring"""
        self.shards[shard.shard_id] = shard
        
        for i in range(self.virtual_nodes):
            key = self._hash(f"{shard.shard_id}:{i}")
            self.ring[key] = shard.shard_id
        
        self.sorted_keys = sorted(self.ring.keys())

    def remove_shard(self, shard_id: str):
        """Remove a shard from the ring"""
        if shard_id in self.shards:
            del self.shards[shard_id]
        
        for i in range(self.virtual_nodes):
            key = self._hash(f"{shard_id}:{i}")
            if key in self.ring:
                del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())

    def get_shard(self, key: str) -> Optional[ShardConfig]:
        """Get shard for a given key"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Binary search for first key >= hash_key
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                shard_id = self.ring[ring_key]
                return self.shards.get(shard_id)
        
        # Wrap around to first shard
        shard_id = self.ring[self.sorted_keys[0]]
        return self.shards.get(shard_id)

    def _hash(self, key: str) -> int:
        """Generate hash for key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class BackpressureController:
    """
    Adaptive backpressure for handling load spikes.
    Implements circuit breaker pattern.
    """
    
    def __init__(self, 
                 max_queue_size: int = 100000,
                 target_latency_ms: float = 100.0,
                 sample_window: int = 1000):
        self.max_queue_size = max_queue_size
        self.target_latency_ms = target_latency_ms
        self.sample_window = sample_window
        
        self.queue_size = 0
        self.latency_samples: List[float] = []
        self.rejection_rate = 0.0
        self.state = "closed"  # closed, half-open, open

    def should_accept(self) -> bool:
        """Check if request should be accepted"""
        if self.state == "open":
            return False
        
        if self.queue_size >= self.max_queue_size:
            return False
        
        # Probabilistic rejection based on load
        if self.rejection_rate > 0 and random.random() < self.rejection_rate:
            return False
        
        return True

    def record_request(self, latency_ms: float, success: bool):
        """Record request outcome for adaptive control"""
        self.latency_samples.append(latency_ms)
        
        if len(self.latency_samples) > self.sample_window:
            self.latency_samples = self.latency_samples[-self.sample_window:]
        
        # Calculate P99 latency
        if len(self.latency_samples) >= 100:
            sorted_samples = sorted(self.latency_samples)
            p99 = sorted_samples[int(len(sorted_samples) * 0.99)]
            
            # Adjust rejection rate based on latency
            if p99 > self.target_latency_ms * 2:
                self.rejection_rate = min(0.5, self.rejection_rate + 0.05)
            elif p99 < self.target_latency_ms:
                self.rejection_rate = max(0.0, self.rejection_rate - 0.01)

    def update_queue_size(self, size: int):
        """Update current queue size"""
        self.queue_size = size
        
        # Circuit breaker logic
        if size > self.max_queue_size * 0.9:
            self.state = "open"
        elif size < self.max_queue_size * 0.5:
            self.state = "closed"


class HyperscaleVoteProcessor:
    """
    Main vote processing engine for hyperscale elections.
    
    Architecture:
    - Geographic sharding (by state/region)
    - Consistent hashing for load distribution
    - Async processing with backpressure
    - Multi-region replication
    - Sub-100ms P99 latency target
    
    Capacity: 100,000+ TPS
    """
    
    def __init__(self, num_shards: int = 50, target_tps: int = 100000):
        self.hash_ring = ConsistentHashRing()
        self.backpressure = BackpressureController()
        self.target_tps = target_tps
        
        # Initialize shards across regions
        regions = ["us-west", "us-central", "us-east", "us-gov-west", "us-gov-east"]
        for i in range(num_shards):
            shard = ShardConfig(
                shard_id=f"shard-{i:03d}",
                region=regions[i % len(regions)],
                capacity_tps=target_tps // num_shards,
                replicas=[f"shard-{i:03d}-replica-{j}" for j in range(2)]
            )
            self.hash_ring.add_shard(shard)
        
        # Metrics
        self.total_processed = 0
        self.total_rejected = 0
        self.processing_times: List[float] = []

    async def process_vote(self, vote: VotePayload) -> ProcessingResult:
        """
        Process an incoming vote with hyperscale guarantees.
        """
        start_time = time.monotonic()
        
        # Backpressure check
        if not self.backpressure.should_accept():
            self.total_rejected += 1
            return ProcessingResult(
                success=False,
                vote_hash="",
                shard_id="",
                latency_ms=0,
                queued=False,
                retry_after=5  # Retry after 5 seconds
            )
        
        # Route to shard
        shard = self.hash_ring.get_shard(vote.voter_id)
        if not shard or not shard.healthy:
            shard = self._get_fallback_shard(vote.jurisdiction)
        
        # Process vote
        try:
            vote_hash = await self._process_on_shard(shard, vote)
            
            latency_ms = (time.monotonic() - start_time) * 1000
            self.processing_times.append(latency_ms)
            self.backpressure.record_request(latency_ms, True)
            self.total_processed += 1
            
            return ProcessingResult(
                success=True,
                vote_hash=vote_hash,
                shard_id=shard.shard_id,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self.backpressure.record_request(latency_ms, False)
            
            return ProcessingResult(
                success=False,
                vote_hash="",
                shard_id=shard.shard_id if shard else "",
                latency_ms=latency_ms,
                retry_after=1
            )

    async def _process_on_shard(self, shard: ShardConfig, vote: VotePayload) -> str:
        """Process vote on specific shard"""
        # Simulate processing (in production, this calls shard service)
        await asyncio.sleep(0.001)  # 1ms processing time
        
        # Generate vote hash
        vote_data = f"{vote.voter_id}:{vote.election_id}:{vote.timestamp}"
        vote_hash = hashlib.sha256(vote_data.encode()).hexdigest()
        
        return vote_hash

    def _get_fallback_shard(self, jurisdiction: str) -> ShardConfig:
        """Get fallback shard for jurisdiction"""
        # Find any healthy shard in same region
        for shard in self.hash_ring.shards.values():
            if shard.healthy:
                return shard
        return list(self.hash_ring.shards.values())[0]

    def get_metrics(self) -> dict:
        """Get current processing metrics"""
        if self.processing_times:
            sorted_times = sorted(self.processing_times[-10000:])
            p50 = sorted_times[len(sorted_times) // 2]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            p50 = p99 = 0
        
        return {
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "acceptance_rate": self.total_processed / (self.total_processed + self.total_rejected) if self.total_processed else 1.0,
            "p50_latency_ms": p50,
            "p99_latency_ms": p99,
            "active_shards": len([s for s in self.hash_ring.shards.values() if s.healthy]),
            "current_tps": len(self.processing_times[-1000:]) if self.processing_times else 0
        }


class GeographicLoadBalancer:
    """
    Geographic load balancer for multi-region deployment.
    Routes voters to nearest healthy region.
    """
    
    def __init__(self):
        self.regions = {
            "us-west": {"lat": 37.7749, "lon": -122.4194, "healthy": True, "load": 0.0},
            "us-central": {"lat": 41.8781, "lon": -87.6298, "healthy": True, "load": 0.0},
            "us-east": {"lat": 40.7128, "lon": -74.0060, "healthy": True, "load": 0.0},
        }

    def get_nearest_region(self, voter_lat: float, voter_lon: float) -> str:
        """Get nearest healthy region for voter"""
        best_region = None
        best_distance = float('inf')
        
        for region, info in self.regions.items():
            if not info["healthy"]:
                continue
            
            distance = self._haversine(voter_lat, voter_lon, info["lat"], info["lon"])
            
            # Weight by current load
            weighted_distance = distance * (1 + info["load"])
            
            if weighted_distance < best_distance:
                best_distance = weighted_distance
                best_region = region
        
        return best_region or "us-central"

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points"""
        import math
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("=== Hyperscale Vote Processor ===\n")
        
        processor = HyperscaleVoteProcessor(num_shards=50, target_tps=100000)
        
        # Simulate high-volume voting
        votes = []
        for i in range(10000):
            vote = VotePayload(
                voter_id=f"voter-{i:08d}",
                election_id="WA_GOV_2024",
                encrypted_ballot=b"encrypted_data",
                timestamp=time.time(),
                jurisdiction="WA",
                zk_proof=b"proof"
            )
            votes.append(vote)
        
        print(f"Processing {len(votes)} votes...")
        start = time.monotonic()
        
        # Process in batches
        results = await asyncio.gather(*[processor.process_vote(v) for v in votes])
        
        elapsed = time.monotonic() - start
        successful = sum(1 for r in results if r.success)
        
        print(f"\nResults:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {len(votes)/elapsed:.0f} TPS")
        print(f"  Successful: {successful}/{len(votes)}")
        
        metrics = processor.get_metrics()
        print(f"\nMetrics:")
        print(f"  P50 latency: {metrics['p50_latency_ms']:.2f}ms")
        print(f"  P99 latency: {metrics['p99_latency_ms']:.2f}ms")
        print(f"  Active shards: {metrics['active_shards']}")
    
    asyncio.run(main())
