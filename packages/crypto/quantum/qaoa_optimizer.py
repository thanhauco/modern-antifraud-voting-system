"""
Hybrid Quantum-Classical Optimization using QAOA
For solving combinatorial optimization problems in election logistics.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import math

class QuantumBackend(Enum):
    """Available quantum backends"""
    SIMULATOR = "simulator"
    IBM_BRISBANE = "ibm_brisbane"
    GOOGLE_SYCAMORE = "google_sycamore"
    AWS_IONQ = "aws_ionq"


@dataclass
class QAOAResult:
    """Result from QAOA optimization"""
    optimal_solution: List[int]
    optimal_value: float
    iterations: int
    final_gamma: List[float]
    final_beta: List[float]
    expectation_history: List[float]


class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm (QAOA)
    
    Hybrid quantum-classical algorithm for combinatorial optimization.
    
    Applications in voting system:
    - Optimal polling station placement
    - Ballot distribution logistics
    - Resource allocation optimization
    - Election official scheduling
    """
    
    def __init__(self, 
                 num_qubits: int,
                 depth: int = 2,
                 backend: QuantumBackend = QuantumBackend.SIMULATOR):
        self.num_qubits = num_qubits
        self.depth = depth  # Number of QAOA layers
        self.backend = backend
        
        # Initialize parameters
        self.gamma = np.random.uniform(0, 2 * np.pi, depth)
        self.beta = np.random.uniform(0, np.pi, depth)

    def _simulate_qaoa_circuit(self, 
                                gamma: np.ndarray, 
                                beta: np.ndarray,
                                cost_hamiltonian: np.ndarray) -> np.ndarray:
        """
        Simulate QAOA circuit to get final state.
        
        In production, this would execute on real quantum hardware.
        """
        n = self.num_qubits
        dim = 2 ** n
        
        # Initial state: uniform superposition |+⟩^n
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)
        
        for layer in range(self.depth):
            # Apply cost unitary: exp(-i * gamma * C)
            phase = np.exp(-1j * gamma[layer] * np.diag(cost_hamiltonian))
            state = phase * state
            
            # Apply mixer unitary: exp(-i * beta * X^n)
            # Simplified: apply single-qubit X rotations
            state = self._apply_mixer(state, beta[layer])
        
        return state

    def _apply_mixer(self, state: np.ndarray, beta: float) -> np.ndarray:
        """Apply mixer Hamiltonian (sum of X operators)"""
        n = self.num_qubits
        dim = 2 ** n
        new_state = np.zeros_like(state)
        
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        for i in range(dim):
            # Diagonal term
            new_state[i] += cos_b ** n * state[i]
            
            # Off-diagonal terms (bit flips)
            for bit in range(n):
                j = i ^ (1 << bit)  # Flip bit
                mix_factor = cos_b ** (n - 1) * (-1j * sin_b)
                new_state[i] += mix_factor * state[j]
        
        return new_state / np.linalg.norm(new_state)

    def _compute_expectation(self, 
                              state: np.ndarray, 
                              cost_hamiltonian: np.ndarray) -> float:
        """Compute expectation value ⟨ψ|C|ψ⟩"""
        probabilities = np.abs(state) ** 2
        expectation = np.sum(probabilities * np.diag(cost_hamiltonian))
        return expectation.real

    def _sample_solution(self, state: np.ndarray, num_shots: int = 1000) -> List[int]:
        """Sample from the quantum state to get classical solution"""
        probabilities = np.abs(state) ** 2
        samples = np.random.choice(len(state), size=num_shots, p=probabilities)
        
        # Find most frequent sample
        counts = np.bincount(samples)
        best_state = np.argmax(counts)
        
        # Convert to bit string
        return [int(b) for b in format(best_state, f'0{self.num_qubits}b')]

    def optimize(self,
                 cost_function: Callable[[List[int]], float],
                 max_iterations: int = 100,
                 learning_rate: float = 0.1) -> QAOAResult:
        """
        Run QAOA optimization.
        
        Args:
            cost_function: Classical cost function to minimize
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for parameter update
            
        Returns:
            QAOAResult with optimal solution
        """
        # Build cost Hamiltonian matrix
        dim = 2 ** self.num_qubits
        cost_hamiltonian = np.zeros((dim, dim))
        
        for i in range(dim):
            bits = [int(b) for b in format(i, f'0{self.num_qubits}b')]
            cost_hamiltonian[i, i] = cost_function(bits)
        
        gamma = self.gamma.copy()
        beta = self.beta.copy()
        
        expectation_history = []
        
        for iteration in range(max_iterations):
            # Simulate QAOA circuit
            state = self._simulate_qaoa_circuit(gamma, beta, cost_hamiltonian)
            
            # Compute expectation
            expectation = self._compute_expectation(state, cost_hamiltonian)
            expectation_history.append(expectation)
            
            # Parameter-shift rule for gradients (simplified)
            grad_gamma = np.zeros_like(gamma)
            grad_beta = np.zeros_like(beta)
            
            shift = np.pi / 2
            for k in range(self.depth):
                # Gamma gradient
                gamma_plus = gamma.copy()
                gamma_plus[k] += shift
                state_plus = self._simulate_qaoa_circuit(gamma_plus, beta, cost_hamiltonian)
                exp_plus = self._compute_expectation(state_plus, cost_hamiltonian)
                
                gamma_minus = gamma.copy()
                gamma_minus[k] -= shift
                state_minus = self._simulate_qaoa_circuit(gamma_minus, beta, cost_hamiltonian)
                exp_minus = self._compute_expectation(state_minus, cost_hamiltonian)
                
                grad_gamma[k] = (exp_plus - exp_minus) / 2
                
                # Beta gradient
                beta_plus = beta.copy()
                beta_plus[k] += shift
                state_plus = self._simulate_qaoa_circuit(gamma, beta_plus, cost_hamiltonian)
                exp_plus = self._compute_expectation(state_plus, cost_hamiltonian)
                
                beta_minus = beta.copy()
                beta_minus[k] -= shift
                state_minus = self._simulate_qaoa_circuit(gamma, beta_minus, cost_hamiltonian)
                exp_minus = self._compute_expectation(state_minus, cost_hamiltonian)
                
                grad_beta[k] = (exp_plus - exp_minus) / 2
            
            # Update parameters (gradient descent for minimization)
            gamma -= learning_rate * grad_gamma
            beta -= learning_rate * grad_beta
        
        # Final solution
        final_state = self._simulate_qaoa_circuit(gamma, beta, cost_hamiltonian)
        optimal_solution = self._sample_solution(final_state)
        optimal_value = cost_function(optimal_solution)
        
        return QAOAResult(
            optimal_solution=optimal_solution,
            optimal_value=optimal_value,
            iterations=max_iterations,
            final_gamma=gamma.tolist(),
            final_beta=beta.tolist(),
            expectation_history=expectation_history
        )


class PollingStationOptimizer:
    """
    Use QAOA to optimize polling station placement.
    """
    
    def __init__(self, num_locations: int, num_stations: int):
        self.num_locations = num_locations
        self.num_stations = num_stations
        self.distances = np.random.rand(num_locations, num_locations) * 100
        self.populations = np.random.randint(1000, 50000, num_locations)
        
        self.qaoa = QAOAOptimizer(num_qubits=num_locations, depth=2)

    def cost_function(self, x: List[int]) -> float:
        """
        Cost function for polling station placement.
        
        Minimize:
        - Total distance from population centers
        - Penalty for wrong number of stations
        """
        selected = [i for i, b in enumerate(x) if b == 1]
        
        # Penalty for wrong number of stations
        station_penalty = abs(len(selected) - self.num_stations) * 1000
        
        if not selected:
            return float('inf')
        
        # Distance cost: each location's distance to nearest station
        distance_cost = 0
        for i in range(self.num_locations):
            min_dist = min(self.distances[i, j] for j in selected)
            distance_cost += min_dist * self.populations[i]
        
        return distance_cost / 1000000 + station_penalty

    def optimize(self) -> Tuple[List[int], float]:
        """Find optimal polling station locations"""
        result = self.qaoa.optimize(self.cost_function, max_iterations=50)
        return result.optimal_solution, result.optimal_value


if __name__ == "__main__":
    print("=== QAOA Polling Station Optimization ===\n")
    
    # Small example: 6 locations, select 2 for polling stations
    optimizer = PollingStationOptimizer(num_locations=6, num_stations=2)
    
    solution, cost = optimizer.optimize()
    selected_stations = [i for i, b in enumerate(solution) if b == 1]
    
    print(f"Optimal solution: {solution}")
    print(f"Selected locations: {selected_stations}")
    print(f"Cost: {cost:.4f}")
