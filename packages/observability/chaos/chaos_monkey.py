"""
Chaos Engineering Framework for Vote System Resilience Testing
Implements fault injection and resilience verification.
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

class FaultType(Enum):
    """Types of faults to inject"""
    LATENCY = "latency"           # Add artificial latency
    ERROR = "error"               # Return error response
    TIMEOUT = "timeout"           # Simulate timeout
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    CRASH = "crash"


@dataclass
class FaultConfig:
    """Configuration for a fault injection"""
    fault_type: FaultType
    probability: float  # 0.0 to 1.0
    duration_ms: int = 0
    target_service: str = "*"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a chaos experiment"""
    experiment_id: str
    hypothesis: str
    passed: bool
    metrics: Dict[str, float]
    observations: List[str]
    duration_seconds: float


class FaultInjector:
    """
    Fault injection engine for chaos experiments.
    """
    
    def __init__(self):
        self.active_faults: Dict[str, FaultConfig] = {}
        self.injection_count = 0
        self.fault_log: List[Dict] = []
    
    def register_fault(self, name: str, config: FaultConfig):
        """Register a fault configuration"""
        self.active_faults[name] = config
    
    def remove_fault(self, name: str):
        """Remove a fault configuration"""
        if name in self.active_faults:
            del self.active_faults[name]
    
    async def maybe_inject(self, service: str) -> Optional[FaultType]:
        """
        Possibly inject a fault based on registered configs.
        Returns the fault type if injected, None otherwise.
        """
        for name, config in self.active_faults.items():
            if not config.enabled:
                continue
            
            if config.target_service != "*" and config.target_service != service:
                continue
            
            if random.random() < config.probability:
                await self._apply_fault(config)
                self._log_injection(name, config, service)
                return config.fault_type
        
        return None
    
    async def _apply_fault(self, config: FaultConfig):
        """Apply the actual fault"""
        if config.fault_type == FaultType.LATENCY:
            await asyncio.sleep(config.duration_ms / 1000)
        
        elif config.fault_type == FaultType.ERROR:
            raise Exception("Chaos: Injected error")
        
        elif config.fault_type == FaultType.TIMEOUT:
            await asyncio.sleep(30)  # Long delay to trigger timeout
        
        elif config.fault_type == FaultType.CRASH:
            raise SystemExit("Chaos: Process crash simulation")
    
    def _log_injection(self, name: str, config: FaultConfig, service: str):
        self.injection_count += 1
        self.fault_log.append({
            "timestamp": time.time(),
            "fault_name": name,
            "fault_type": config.fault_type.value,
            "service": service
        })


class ChaosExperiment:
    """
    A chaos experiment with hypothesis-driven testing.
    """
    
    def __init__(self, 
                 name: str,
                 hypothesis: str,
                 fault_injector: FaultInjector):
        self.name = name
        self.hypothesis = hypothesis
        self.fault_injector = fault_injector
        self.start_time: Optional[float] = None
        self.metrics: Dict[str, List[float]] = {}
        self.observations: List[str] = []
    
    def record_metric(self, name: str, value: float):
        """Record a metric during experiment"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def observe(self, message: str):
        """Record an observation"""
        self.observations.append(f"[{time.time() - self.start_time:.2f}s] {message}")
    
    async def run(self, 
                  workload: Callable,
                  duration_seconds: float,
                  verify: Callable[[Dict], bool]) -> ExperimentResult:
        """
        Run the chaos experiment.
        
        Args:
            workload: Async function to run as workload
            duration_seconds: How long to run
            verify: Function to verify hypothesis
        """
        self.start_time = time.time()
        
        # Run workload for duration
        end_time = self.start_time + duration_seconds
        while time.time() < end_time:
            try:
                result = await workload()
                self.record_metric("success", 1)
                if hasattr(result, 'latency_ms'):
                    self.record_metric("latency_ms", result.latency_ms)
            except Exception as e:
                self.record_metric("failure", 1)
                self.observe(f"Error: {str(e)}")
            
            await asyncio.sleep(0.01)
        
        # Calculate aggregate metrics
        agg_metrics = {}
        for name, values in self.metrics.items():
            agg_metrics[f"{name}_count"] = len(values)
            agg_metrics[f"{name}_sum"] = sum(values)
            if values:
                agg_metrics[f"{name}_avg"] = sum(values) / len(values)
        
        # Verify hypothesis
        passed = verify(agg_metrics)
        
        return ExperimentResult(
            experiment_id=f"exp_{int(self.start_time)}",
            hypothesis=self.hypothesis,
            passed=passed,
            metrics=agg_metrics,
            observations=self.observations,
            duration_seconds=duration_seconds
        )


class VoteChaosMonkey:
    """
    Chaos Monkey for vote system resilience testing.
    
    Experiments:
    1. API latency injection
    2. Database connection failure
    3. Blockchain node crash
    4. Redis cache unavailable
    """
    
    def __init__(self):
        self.injector = FaultInjector()
        self.experiments_run = 0
        self.results: List[ExperimentResult] = []
    
    async def run_api_latency_experiment(self, workload: Callable) -> ExperimentResult:
        """Test: System handles increased latency gracefully"""
        
        # Register latency fault
        self.injector.register_fault("api_latency", FaultConfig(
            fault_type=FaultType.LATENCY,
            probability=0.3,
            duration_ms=500,
            target_service="vote-api"
        ))
        
        experiment = ChaosExperiment(
            name="API Latency Resilience",
            hypothesis="System maintains > 95% success rate under 500ms latency injection",
            fault_injector=self.injector
        )
        
        def verify(metrics: Dict) -> bool:
            success = metrics.get("success_count", 0)
            failure = metrics.get("failure_count", 0)
            total = success + failure
            success_rate = success / total if total > 0 else 0
            return success_rate >= 0.95
        
        result = await experiment.run(workload, duration_seconds=5, verify=verify)
        
        self.injector.remove_fault("api_latency")
        self.results.append(result)
        return result
    
    async def run_database_failure_experiment(self, workload: Callable) -> ExperimentResult:
        """Test: System handles database failures with failover"""
        
        self.injector.register_fault("db_error", FaultConfig(
            fault_type=FaultType.ERROR,
            probability=0.1,
            target_service="postgres"
        ))
        
        experiment = ChaosExperiment(
            name="Database Failure Resilience",
            hypothesis="System recovers from 10% database failures via retry",
            fault_injector=self.injector
        )
        
        def verify(metrics: Dict) -> bool:
            return metrics.get("success_count", 0) > metrics.get("failure_count", 0)
        
        result = await experiment.run(workload, duration_seconds=5, verify=verify)
        
        self.injector.remove_fault("db_error")
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all experiments"""
        passed = sum(1 for r in self.results if r.passed)
        return {
            "total_experiments": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results) if self.results else 0
        }


if __name__ == "__main__":
    async def main():
        print("=== Chaos Engineering Framework ===\n")
        
        chaos = VoteChaosMonkey()
        
        # Mock workload
        async def mock_vote_workload():
            await asyncio.sleep(0.01)
            if random.random() < 0.02:  # 2% natural failure rate
                raise Exception("Natural failure")
            return type('Result', (), {'latency_ms': random.uniform(10, 50)})()
        
        print("Running API latency experiment...")
        result = await chaos.run_api_latency_experiment(mock_vote_workload)
        print(f"  Hypothesis: {result.hypothesis}")
        print(f"  Passed: {result.passed}")
        print(f"  Success rate: {result.metrics.get('success_avg', 0):.2%}")
        
        summary = chaos.get_summary()
        print(f"\nSummary: {summary['passed']}/{summary['total_experiments']} passed")
    
    asyncio.run(main())
