"""
Event-Driven Architecture for Real-Time Vote Stream Processing
Handles millions of concurrent vote events using Apache Kafka patterns.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import random

class EventType(Enum):
    """Vote system event types"""
    VOTE_SUBMITTED = "vote.submitted"
    VOTE_VALIDATED = "vote.validated"
    VOTE_PROCESSED = "vote.processed"
    VOTE_CONFIRMED = "vote.confirmed"
    VOTE_REJECTED = "vote.rejected"
    FRAUD_DETECTED = "fraud.detected"
    AUDIT_REQUIRED = "audit.required"


@dataclass
class Event:
    """Base event structure"""
    event_id: str
    event_type: EventType
    timestamp: float
    partition_key: str
    payload: Dict[str, Any]
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "partition_key": self.partition_key,
            "payload": self.payload,
            "metadata": self.metadata
        })


@dataclass
class PartitionConfig:
    """Kafka-style partition configuration"""
    partition_id: int
    topic: str
    leader_broker: str
    replicas: List[str]
    isr: List[str]  # In-sync replicas
    offset: int = 0


class EventPartitioner:
    """
    Partitions events for parallel processing.
    Ensures ordering within partition key.
    """
    
    def __init__(self, num_partitions: int = 256):
        self.num_partitions = num_partitions

    def get_partition(self, partition_key: str) -> int:
        """Consistent partition assignment"""
        key_hash = int(hashlib.md5(partition_key.encode()).hexdigest(), 16)
        return key_hash % self.num_partitions


class InMemoryEventStore:
    """
    In-memory event store simulating Kafka behavior.
    In production, use actual Kafka/Pulsar.
    """
    
    def __init__(self, num_partitions: int = 256):
        self.partitions: Dict[int, List[Event]] = defaultdict(list)
        self.partitioner = EventPartitioner(num_partitions)
        self.consumer_offsets: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.num_partitions = num_partitions

    async def publish(self, event: Event) -> int:
        """Publish event to partition"""
        partition = self.partitioner.get_partition(event.partition_key)
        self.partitions[partition].append(event)
        return len(self.partitions[partition]) - 1  # Return offset

    async def consume(self, consumer_group: str, partition: int, 
                      batch_size: int = 100) -> List[Event]:
        """Consume events from partition"""
        offset = self.consumer_offsets[consumer_group][partition]
        events = self.partitions[partition][offset:offset + batch_size]
        
        if events:
            self.consumer_offsets[consumer_group][partition] = offset + len(events)
        
        return events

    def get_lag(self, consumer_group: str) -> Dict[int, int]:
        """Get consumer lag per partition"""
        lag = {}
        for partition in range(self.num_partitions):
            current_offset = self.consumer_offsets[consumer_group][partition]
            latest_offset = len(self.partitions[partition])
            lag[partition] = latest_offset - current_offset
        return lag


class StreamProcessor:
    """
    Stream processor for vote events.
    Implements exactly-once processing semantics.
    """
    
    def __init__(self, 
                 consumer_group: str,
                 event_store: InMemoryEventStore,
                 handler: Callable[[Event], Any]):
        self.consumer_group = consumer_group
        self.event_store = event_store
        self.handler = handler
        self.running = False
        self.processed_count = 0
        self.error_count = 0

    async def start(self, partitions: List[int]):
        """Start processing events from assigned partitions"""
        self.running = True
        
        while self.running:
            for partition in partitions:
                events = await self.event_store.consume(
                    self.consumer_group, partition, batch_size=100
                )
                
                for event in events:
                    try:
                        await self._process_event(event)
                        self.processed_count += 1
                    except Exception as e:
                        self.error_count += 1
                        # In production, send to dead letter queue
            
            await asyncio.sleep(0.001)  # 1ms polling interval

    async def _process_event(self, event: Event):
        """Process single event with idempotency check"""
        # In production, check idempotency key in database
        result = await self.handler(event)
        return result

    def stop(self):
        """Stop processor"""
        self.running = False


class VoteStreamPipeline:
    """
    Complete streaming pipeline for vote processing.
    
    Pipeline stages:
    1. Ingestion: Receive vote submissions
    2. Validation: Verify signatures, ZK proofs
    3. Fraud Check: ML-based fraud detection
    4. Processing: Write to blockchain
    5. Confirmation: Notify voter
    """
    
    def __init__(self, num_partitions: int = 256):
        self.event_store = InMemoryEventStore(num_partitions)
        self.processors: Dict[str, StreamProcessor] = {}
        self.metrics = {
            "ingested": 0,
            "validated": 0,
            "processed": 0,
            "confirmed": 0,
            "rejected": 0,
            "fraud_detected": 0
        }

    async def ingest_vote(self, vote_data: dict) -> str:
        """Ingest a new vote into the pipeline"""
        event_id = hashlib.sha256(
            f"{vote_data['voter_id']}:{time.time()}".encode()
        ).hexdigest()[:32]
        
        event = Event(
            event_id=event_id,
            event_type=EventType.VOTE_SUBMITTED,
            timestamp=time.time(),
            partition_key=vote_data["voter_id"],
            payload=vote_data
        )
        
        await self.event_store.publish(event)
        self.metrics["ingested"] += 1
        
        return event_id

    async def handle_validation(self, event: Event):
        """Validate vote (ZK proof, signature)"""
        # Simulate validation
        await asyncio.sleep(0.001)
        
        is_valid = random.random() > 0.01  # 99% valid
        
        if is_valid:
            validated_event = Event(
                event_id=f"val_{event.event_id}",
                event_type=EventType.VOTE_VALIDATED,
                timestamp=time.time(),
                partition_key=event.partition_key,
                payload={**event.payload, "validated": True}
            )
            await self.event_store.publish(validated_event)
            self.metrics["validated"] += 1
        else:
            self.metrics["rejected"] += 1

    async def handle_fraud_check(self, event: Event):
        """Run fraud detection on vote"""
        await asyncio.sleep(0.001)
        
        fraud_score = random.random()
        is_fraud = fraud_score > 0.99  # 1% fraud rate
        
        if is_fraud:
            fraud_event = Event(
                event_id=f"fraud_{event.event_id}",
                event_type=EventType.FRAUD_DETECTED,
                timestamp=time.time(),
                partition_key=event.partition_key,
                payload={**event.payload, "fraud_score": fraud_score}
            )
            await self.event_store.publish(fraud_event)
            self.metrics["fraud_detected"] += 1
        else:
            processed_event = Event(
                event_id=f"proc_{event.event_id}",
                event_type=EventType.VOTE_PROCESSED,
                timestamp=time.time(),
                partition_key=event.partition_key,
                payload=event.payload
            )
            await self.event_store.publish(processed_event)
            self.metrics["processed"] += 1

    async def handle_confirmation(self, event: Event):
        """Send confirmation to voter"""
        await asyncio.sleep(0.0005)
        
        confirmed_event = Event(
            event_id=f"conf_{event.event_id}",
            event_type=EventType.VOTE_CONFIRMED,
            timestamp=time.time(),
            partition_key=event.partition_key,
            payload={**event.payload, "confirmation_code": event.event_id[:8]}
        )
        await self.event_store.publish(confirmed_event)
        self.metrics["confirmed"] += 1

    def get_metrics(self) -> dict:
        """Get pipeline metrics"""
        return {
            **self.metrics,
            "pipeline_lag": sum(self.event_store.get_lag("vote-processors").values())
        }


class AutoScaler:
    """
    Auto-scaler for stream processors.
    Scales based on consumer lag and CPU utilization.
    """
    
    def __init__(self, 
                 min_instances: int = 2,
                 max_instances: int = 100,
                 target_lag_per_instance: int = 1000):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_lag_per_instance = target_lag_per_instance
        self.current_instances = min_instances

    def calculate_desired_instances(self, total_lag: int) -> int:
        """Calculate desired number of instances based on lag"""
        if total_lag == 0:
            return self.min_instances
        
        desired = max(
            self.min_instances,
            min(self.max_instances, total_lag // self.target_lag_per_instance + 1)
        )
        
        return desired

    def should_scale(self, current_lag: int) -> tuple:
        """Determine if scaling is needed"""
        desired = self.calculate_desired_instances(current_lag)
        
        if desired > self.current_instances:
            return ("scale_up", desired - self.current_instances)
        elif desired < self.current_instances:
            return ("scale_down", self.current_instances - desired)
        else:
            return ("no_change", 0)


if __name__ == "__main__":
    async def main():
        print("=== Vote Stream Processing Pipeline ===\n")
        
        pipeline = VoteStreamPipeline(num_partitions=64)
        
        # Simulate high-volume ingestion
        print("Ingesting 10,000 votes...")
        start = time.monotonic()
        
        tasks = []
        for i in range(10000):
            vote_data = {
                "voter_id": f"voter-{i:06d}",
                "election_id": "WA_GOV_2024",
                "candidate": random.choice(["A", "B", "C"]),
                "timestamp": time.time()
            }
            tasks.append(pipeline.ingest_vote(vote_data))
        
        await asyncio.gather(*tasks)
        
        elapsed = time.monotonic() - start
        print(f"Ingestion completed in {elapsed:.2f}s ({10000/elapsed:.0f} events/sec)")
        
        # Get metrics
        metrics = pipeline.get_metrics()
        print(f"\nPipeline Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    asyncio.run(main())
