"""
OpenTelemetry Distributed Tracing for Vote System
End-to-end observability across all microservices.
"""

import time
import random
import hashlib
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import json

class SpanKind(Enum):
    """OpenTelemetry span kinds"""
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


@dataclass
class SpanContext:
    """Distributed tracing context"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # Sampled
    
    def to_traceparent(self) -> str:
        """W3C Trace Context format"""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"
    
    @classmethod
    def from_traceparent(cls, header: str) -> "SpanContext":
        parts = header.split("-")
        return cls(
            trace_id=parts[1],
            span_id=parts[2],
            trace_flags=int(parts[3], 16)
        )


@dataclass
class Span:
    """Tracing span representing a unit of work"""
    name: str
    context: SpanContext
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Dict = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_status(self, status: str, description: str = None):
        self.status = status
        if description:
            self.attributes["status.description"] = description
    
    def end(self):
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


class Tracer:
    """
    OpenTelemetry-compatible tracer for vote system.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: List[Span] = []
        self.current_span: Optional[Span] = None
        self.exporters: List["SpanExporter"] = []
    
    def _generate_id(self, length: int = 16) -> str:
        return hashlib.sha256(
            f"{time.time()}{random.random()}".encode()
        ).hexdigest()[:length]
    
    @contextmanager
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL, 
                   parent: SpanContext = None):
        """Start a new span, optionally as child of parent"""
        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
        elif self.current_span:
            trace_id = self.current_span.context.trace_id
            parent_span_id = self.current_span.context.span_id
        else:
            trace_id = self._generate_id(32)
            parent_span_id = None
        
        context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_id(16),
            parent_span_id=parent_span_id
        )
        
        span = Span(
            name=name,
            context=context,
            kind=kind,
            start_time=time.time()
        )
        span.set_attribute("service.name", self.service_name)
        
        previous_span = self.current_span
        self.current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_status("ERROR", str(e))
            raise
        finally:
            span.end()
            self.spans.append(span)
            self.current_span = previous_span
            
            # Export span
            for exporter in self.exporters:
                exporter.export([span])
    
    def add_exporter(self, exporter: "SpanExporter"):
        self.exporters.append(exporter)


class SpanExporter:
    """Base span exporter"""
    def export(self, spans: List[Span]):
        raise NotImplementedError


class ConsoleExporter(SpanExporter):
    """Export spans to console"""
    def export(self, spans: List[Span]):
        for span in spans:
            print(f"[TRACE] {span.name} | {span.duration_ms:.2f}ms | {span.status}")


class OTLPExporter(SpanExporter):
    """
    Export spans to OTLP collector (simulated).
    In production, sends to Jaeger/Tempo/etc.
    """
    def __init__(self, endpoint: str = "http://localhost:4317"):
        self.endpoint = endpoint
        self.buffer: List[Dict] = []
    
    def export(self, spans: List[Span]):
        for span in spans:
            self.buffer.append({
                "traceId": span.context.trace_id,
                "spanId": span.context.span_id,
                "parentSpanId": span.context.parent_span_id,
                "name": span.name,
                "kind": span.kind.value,
                "startTimeUnixNano": int(span.start_time * 1e9),
                "endTimeUnixNano": int(span.end_time * 1e9) if span.end_time else 0,
                "attributes": span.attributes,
                "status": {"code": span.status}
            })


class VoteSystemTracer:
    """
    Pre-configured tracer for vote system services.
    """
    _instances: Dict[str, Tracer] = {}
    
    @classmethod
    def get_tracer(cls, service_name: str) -> Tracer:
        if service_name not in cls._instances:
            tracer = Tracer(service_name)
            tracer.add_exporter(OTLPExporter())
            cls._instances[service_name] = tracer
        return cls._instances[service_name]


# Decorators for easy instrumentation
def trace(name: str = None, kind: SpanKind = SpanKind.INTERNAL):
    """Decorator to trace a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = VoteSystemTracer.get_tracer("vote-api")
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_span(span_name, kind) as span:
                span.set_attribute("function.name", func.__name__)
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


# Vote-specific tracing
class VoteTracing:
    """
    Vote-specific tracing with semantic conventions.
    """
    
    def __init__(self):
        self.tracer = VoteSystemTracer.get_tracer("vote-processor")
    
    @contextmanager
    def trace_vote_submission(self, voter_id: str, election_id: str):
        """Trace complete vote submission flow"""
        with self.tracer.start_span("vote.submission", SpanKind.SERVER) as span:
            span.set_attribute("voter.id_hash", hashlib.sha256(voter_id.encode()).hexdigest()[:16])
            span.set_attribute("election.id", election_id)
            span.add_event("vote.received")
            yield span
            span.add_event("vote.processed")
    
    @contextmanager
    def trace_zk_verification(self):
        """Trace ZK proof verification"""
        with self.tracer.start_span("crypto.zk_verify", SpanKind.INTERNAL) as span:
            span.set_attribute("crypto.algorithm", "groth16")
            yield span
    
    @contextmanager
    def trace_blockchain_write(self, tx_hash: str = None):
        """Trace blockchain transaction"""
        with self.tracer.start_span("blockchain.write", SpanKind.CLIENT) as span:
            span.set_attribute("blockchain.network", "besu")
            if tx_hash:
                span.set_attribute("tx.hash", tx_hash)
            yield span


if __name__ == "__main__":
    # Demo tracing
    tracer = VoteSystemTracer.get_tracer("vote-api")
    tracer.add_exporter(ConsoleExporter())
    
    vote_tracing = VoteTracing()
    
    with vote_tracing.trace_vote_submission("voter123", "WA_GOV_2024") as root:
        time.sleep(0.01)
        
        with vote_tracing.trace_zk_verification():
            time.sleep(0.005)
        
        with vote_tracing.trace_blockchain_write("0xabc123"):
            time.sleep(0.02)
    
    print("\nTracing demo complete!")
