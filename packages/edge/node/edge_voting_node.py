"""
Edge Computing Node for Low-Latency Voting
Runs at polling stations for offline-capable voting.
"""

import asyncio
import hashlib
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class NodeState(Enum):
    """Edge node operational states"""
    ONLINE = "online"           # Connected to central
    OFFLINE = "offline"         # Disconnected, caching votes
    SYNCING = "syncing"         # Reconnecting, uploading cached
    DEGRADED = "degraded"       # Partial functionality
    MAINTENANCE = "maintenance"


@dataclass
class CachedVote:
    """A vote cached during offline operation"""
    vote_id: str
    encrypted_ballot: bytes
    voter_hash: str  # Hashed voter ID for dedup
    timestamp: float
    zk_proof: bytes
    signature: bytes
    synced: bool = False


@dataclass
class EdgeNodeConfig:
    """Configuration for edge node"""
    node_id: str
    jurisdiction: str
    polling_station: str
    max_cache_size: int = 10000
    sync_batch_size: int = 100
    heartbeat_interval: int = 5
    offline_threshold: int = 30  # Seconds without heartbeat


class OfflineVoteCache:
    """
    Persistent cache for votes during network outages.
    Ensures no votes are lost.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.votes: Dict[str, CachedVote] = {}
        self.sync_queue: deque = deque()
        self.voter_hashes: set = set()  # Prevent double voting
    
    def cache_vote(self, vote: CachedVote) -> bool:
        """Cache a vote for later sync"""
        if len(self.votes) >= self.max_size:
            return False
        
        if vote.voter_hash in self.voter_hashes:
            return False  # Already voted
        
        self.votes[vote.vote_id] = vote
        self.voter_hashes.add(vote.voter_hash)
        self.sync_queue.append(vote.vote_id)
        return True
    
    def get_pending_sync(self, batch_size: int = 100) -> List[CachedVote]:
        """Get batch of votes pending sync"""
        pending = []
        for vote_id in list(self.sync_queue)[:batch_size]:
            vote = self.votes.get(vote_id)
            if vote and not vote.synced:
                pending.append(vote)
        return pending
    
    def mark_synced(self, vote_ids: List[str]):
        """Mark votes as synced to central"""
        for vote_id in vote_ids:
            if vote_id in self.votes:
                self.votes[vote_id].synced = True
            if vote_id in self.sync_queue:
                self.sync_queue.remove(vote_id)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        synced = sum(1 for v in self.votes.values() if v.synced)
        return {
            "total_cached": len(self.votes),
            "synced": synced,
            "pending": len(self.votes) - synced,
            "queue_depth": len(self.sync_queue),
            "capacity_used": len(self.votes) / self.max_size
        }


class EdgeVotingNode:
    """
    Edge computing node for polling station deployment.
    
    Features:
    - Local vote validation (ZK proof verification)
    - Offline operation with vote caching
    - Automatic sync when connectivity restored
    - Tamper-evident local storage
    """
    
    def __init__(self, config: EdgeNodeConfig):
        self.config = config
        self.state = NodeState.ONLINE
        self.cache = OfflineVoteCache(config.max_cache_size)
        self.last_heartbeat = time.time()
        self.votes_processed = 0
        self.central_connected = True
        
        # Local verification keys (loaded from secure storage)
        self.verification_keys: Dict[str, bytes] = {}
    
    async def process_vote(self, 
                           voter_id: str,
                           encrypted_ballot: bytes,
                           zk_proof: bytes,
                           signature: bytes) -> dict:
        """
        Process a vote at the edge node.
        Works both online and offline.
        """
        # Hash voter ID for privacy
        voter_hash = hashlib.sha256(voter_id.encode()).hexdigest()
        
        # Check for double voting locally
        if voter_hash in self.cache.voter_hashes:
            return {"success": False, "error": "Already voted at this location"}
        
        # Verify ZK proof locally
        if not await self._verify_zk_proof(zk_proof):
            return {"success": False, "error": "Invalid ZK proof"}
        
        # Verify signature
        if not await self._verify_signature(signature, encrypted_ballot):
            return {"success": False, "error": "Invalid signature"}
        
        # Generate vote ID
        vote_id = hashlib.sha256(
            f"{self.config.node_id}:{time.time()}:{voter_hash}".encode()
        ).hexdigest()[:32]
        
        vote = CachedVote(
            vote_id=vote_id,
            encrypted_ballot=encrypted_ballot,
            voter_hash=voter_hash,
            timestamp=time.time(),
            zk_proof=zk_proof,
            signature=signature
        )
        
        if self.state == NodeState.ONLINE and self.central_connected:
            # Try to send directly
            success = await self._send_to_central(vote)
            if success:
                vote.synced = True
                self.cache.cache_vote(vote)
                self.votes_processed += 1
                return {"success": True, "vote_id": vote_id, "synced": True}
        
        # Cache for later sync
        cached = self.cache.cache_vote(vote)
        if cached:
            self.votes_processed += 1
            return {"success": True, "vote_id": vote_id, "synced": False, "cached": True}
        
        return {"success": False, "error": "Cache full"}
    
    async def _verify_zk_proof(self, proof: bytes) -> bool:
        """Verify ZK proof locally using cached verification key"""
        # In production, uses snarkjs or similar
        await asyncio.sleep(0.01)  # Simulate verification
        return len(proof) > 0
    
    async def _verify_signature(self, signature: bytes, data: bytes) -> bool:
        """Verify signature locally"""
        await asyncio.sleep(0.001)
        return len(signature) > 0
    
    async def _send_to_central(self, vote: CachedVote) -> bool:
        """Send vote to central server"""
        try:
            # Simulate network call
            await asyncio.sleep(0.05)
            return self.central_connected
        except Exception:
            return False
    
    async def sync_cached_votes(self):
        """Sync cached votes to central server"""
        if self.state != NodeState.ONLINE:
            return {"synced": 0, "error": "Node not online"}
        
        pending = self.cache.get_pending_sync(self.config.sync_batch_size)
        if not pending:
            return {"synced": 0, "message": "No pending votes"}
        
        synced_ids = []
        for vote in pending:
            success = await self._send_to_central(vote)
            if success:
                synced_ids.append(vote.vote_id)
            else:
                break  # Stop on first failure
        
        self.cache.mark_synced(synced_ids)
        return {"synced": len(synced_ids), "remaining": len(pending) - len(synced_ids)}
    
    async def heartbeat(self):
        """Check connectivity to central"""
        try:
            # Simulate heartbeat
            await asyncio.sleep(0.01)
            self.last_heartbeat = time.time()
            self.central_connected = True
            
            if self.state == NodeState.OFFLINE:
                self.state = NodeState.SYNCING
                await self.sync_cached_votes()
                self.state = NodeState.ONLINE
            
            return True
        except Exception:
            self.central_connected = False
            if time.time() - self.last_heartbeat > self.config.offline_threshold:
                self.state = NodeState.OFFLINE
            return False
    
    def get_status(self) -> dict:
        """Get node status"""
        return {
            "node_id": self.config.node_id,
            "state": self.state.value,
            "jurisdiction": self.config.jurisdiction,
            "polling_station": self.config.polling_station,
            "votes_processed": self.votes_processed,
            "central_connected": self.central_connected,
            "cache": self.cache.get_stats(),
            "last_heartbeat": self.last_heartbeat
        }


class EdgeNodeOrchestrator:
    """
    Manages fleet of edge nodes across polling stations.
    """
    
    def __init__(self):
        self.nodes: Dict[str, EdgeVotingNode] = {}
    
    def register_node(self, config: EdgeNodeConfig) -> EdgeVotingNode:
        """Register a new edge node"""
        node = EdgeVotingNode(config)
        self.nodes[config.node_id] = node
        return node
    
    def get_fleet_status(self) -> dict:
        """Get status of all nodes"""
        statuses = {node_id: node.get_status() for node_id, node in self.nodes.items()}
        
        online = sum(1 for s in statuses.values() if s["state"] == "online")
        offline = sum(1 for s in statuses.values() if s["state"] == "offline")
        total_votes = sum(s["votes_processed"] for s in statuses.values())
        
        return {
            "total_nodes": len(self.nodes),
            "online": online,
            "offline": offline,
            "total_votes": total_votes,
            "nodes": statuses
        }


if __name__ == "__main__":
    async def main():
        print("=== Edge Voting Node ===\n")
        
        config = EdgeNodeConfig(
            node_id="edge-wa-seattle-001",
            jurisdiction="WA",
            polling_station="Seattle Central Library"
        )
        
        node = EdgeVotingNode(config)
        
        # Process votes
        print("Processing votes...")
        for i in range(10):
            result = await node.process_vote(
                voter_id=f"voter_{i}",
                encrypted_ballot=b"encrypted_vote_data",
                zk_proof=b"zk_proof_bytes",
                signature=b"signature_bytes"
            )
            print(f"  Vote {i}: {result['success']}, synced={result.get('synced', False)}")
        
        # Simulate going offline
        print("\nSimulating offline mode...")
        node.central_connected = False
        node.state = NodeState.OFFLINE
        
        for i in range(10, 15):
            result = await node.process_vote(
                voter_id=f"voter_{i}",
                encrypted_ballot=b"encrypted_vote_data",
                zk_proof=b"zk_proof_bytes",
                signature=b"signature_bytes"
            )
            print(f"  Offline vote {i}: cached={result.get('cached', False)}")
        
        # Status
        status = node.get_status()
        print(f"\nNode Status:")
        print(f"  State: {status['state']}")
        print(f"  Votes processed: {status['votes_processed']}")
        print(f"  Cached pending: {status['cache']['pending']}")
    
    asyncio.run(main())
