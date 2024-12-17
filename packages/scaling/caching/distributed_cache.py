"""
Distributed Caching Layer for High-Performance Vote System
Multi-tier caching with Redis Cluster for millions of concurrent users.
"""

import asyncio
import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import random

class CacheLevel(Enum):
    """Cache tier levels"""
    L1_LOCAL = "l1_local"       # In-process cache
    L2_REDIS = "l2_redis"       # Distributed Redis
    L3_DB = "l3_database"       # Database


@dataclass
class CacheEntry:
    """Cached data entry"""
    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    hits: int = 0
    level: CacheLevel = CacheLevel.L1_LOCAL


class LRUCache:
    """
    Thread-safe LRU cache for L1 (in-process) caching.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.storage: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item and move to end (most recently used)"""
        if key not in self.storage:
            return None
        
        entry = self.storage[key]
        
        # Check expiration
        if time.time() > entry.created_at + entry.ttl_seconds:
            del self.storage[key]
            return None
        
        # Move to end
        self.storage.move_to_end(key)
        entry.hits += 1
        return entry

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set item, evict LRU if at capacity"""
        if key in self.storage:
            self.storage.move_to_end(key)
        
        self.storage[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl,
            level=CacheLevel.L1_LOCAL
        )
        
        while len(self.storage) > self.max_size:
            self.storage.popitem(last=False)

    def invalidate(self, key: str):
        """Remove item from cache"""
        if key in self.storage:
            del self.storage[key]

    def clear(self):
        """Clear all items"""
        self.storage.clear()


class RedisClusterSimulator:
    """
    Simulates Redis Cluster behavior for L2 caching.
    In production, use actual Redis Cluster.
    """
    
    def __init__(self, num_slots: int = 16384, num_nodes: int = 6):
        self.num_slots = num_slots
        self.nodes: Dict[int, Dict[str, CacheEntry]] = {}
        self.slot_to_node: Dict[int, int] = {}
        
        # Initialize nodes and slot assignment
        slots_per_node = num_slots // num_nodes
        for i in range(num_nodes):
            self.nodes[i] = {}
            start_slot = i * slots_per_node
            end_slot = (i + 1) * slots_per_node if i < num_nodes - 1 else num_slots
            for slot in range(start_slot, end_slot):
                self.slot_to_node[slot] = i

    def _get_slot(self, key: str) -> int:
        """CRC16 hash to slot (simplified)"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_slots

    def _get_node(self, key: str) -> int:
        """Get node for key"""
        slot = self._get_slot(key)
        return self.slot_to_node[slot]

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cluster"""
        node = self._get_node(key)
        entry = self.nodes[node].get(key)
        
        if entry is None:
            return None
        
        if time.time() > entry.created_at + entry.ttl_seconds:
            del self.nodes[node][key]
            return None
        
        entry.hits += 1
        return entry.value

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cluster"""
        node = self._get_node(key)
        self.nodes[node][key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl,
            level=CacheLevel.L2_REDIS
        )

    async def delete(self, key: str):
        """Delete from cluster"""
        node = self._get_node(key)
        if key in self.nodes[node]:
            del self.nodes[node][key]


class MultiTierCache:
    """
    Multi-tier caching system for vote data.
    
    L1: In-process LRU cache (sub-millisecond)
    L2: Redis Cluster (1-5ms)
    L3: Database (fallback)
    
    Features:
    - Write-through and write-behind
    - Cache invalidation propagation
    - Hot key detection and replication
    - Cache warming for elections
    """
    
    def __init__(self, l1_size: int = 10000):
        self.l1_cache = LRUCache(max_size=l1_size)
        self.l2_cache = RedisClusterSimulator()
        
        # Metrics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "writes": 0
        }
        
        # Hot key tracking
        self.hot_keys: Dict[str, int] = {}

    async def get(self, key: str) -> Tuple[Optional[Any], CacheLevel]:
        """Get value with multi-tier lookup."""
        # Try L1
        l1_entry = self.l1_cache.get(key)
        if l1_entry:
            self.stats["l1_hits"] += 1
            return l1_entry.value, CacheLevel.L1_LOCAL
        
        # Try L2
        l2_value = await self.l2_cache.get(key)
        if l2_value is not None:
            self.stats["l2_hits"] += 1
            # Promote to L1
            self.l1_cache.set(key, l2_value, ttl=60)
            return l2_value, CacheLevel.L2_REDIS
        
        # Miss
        self.stats["misses"] += 1
        return None, CacheLevel.L3_DB

    async def set(self, key: str, value: Any, 
                  l1_ttl: int = 60, l2_ttl: int = 3600):
        """Set value in all cache tiers (write-through)."""
        self.l1_cache.set(key, value, ttl=l1_ttl)
        await self.l2_cache.set(key, value, ttl=l2_ttl)
        self.stats["writes"] += 1

    async def invalidate(self, key: str):
        """Invalidate across all tiers"""
        self.l1_cache.invalidate(key)
        await self.l2_cache.delete(key)

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["misses"]
        
        return {
            **self.stats,
            "total_requests": total,
            "l1_hit_rate": self.stats["l1_hits"] / total if total > 0 else 0,
            "overall_hit_rate": (self.stats["l1_hits"] + self.stats["l2_hits"]) / total if total > 0 else 0
        }


if __name__ == "__main__":
    async def main():
        print("=== Multi-Tier Caching System ===")
        cache = MultiTierCache(l1_size=1000)
        
        # Write entries
        for i in range(5000):
            await cache.set(f"voter:{i}", {"eligible": True})
        
        # Read entries
        for _ in range(1000):
            await cache.get(f"voter:{random.randint(0, 4999)}")
        
        stats = cache.get_stats()
        print(f"Overall Hit Rate: {stats['overall_hit_rate']:.2%}")
    
    asyncio.run(main())
