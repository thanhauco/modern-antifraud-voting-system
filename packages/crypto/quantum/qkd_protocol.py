"""
Quantum Key Distribution (QKD) Simulation
Implements BB84 protocol for quantum-secure key exchange.
"""

import secrets
import hashlib
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

class Basis(Enum):
    """Measurement basis for qubits"""
    RECTILINEAR = "rectilinear"  # |0⟩, |1⟩
    DIAGONAL = "diagonal"        # |+⟩, |-⟩

class Qubit:
    """Simulated qubit for QKD"""
    def __init__(self, value: int, basis: Basis):
        self.value = value  # 0 or 1
        self.basis = basis
        self._measured = False

    def measure(self, measurement_basis: Basis) -> int:
        """
        Measure qubit in given basis.
        If bases match, returns original value.
        If bases differ, returns random value (50% each).
        """
        self._measured = True
        
        if measurement_basis == self.basis:
            return self.value
        else:
            # Wrong basis - random outcome
            return random.randint(0, 1)

    def __repr__(self):
        return f"Qubit(value={self.value}, basis={self.basis.value})"


@dataclass
class QKDResult:
    """Result of QKD key exchange"""
    shared_key: bytes
    key_length_bits: int
    qber: float  # Quantum Bit Error Rate
    secure: bool
    raw_key_length: int
    final_key_length: int


class BB84Protocol:
    """
    BB84 Quantum Key Distribution Protocol Implementation
    
    This is a simulation that models the behavior of a real QKD system.
    In production, this would interface with actual quantum hardware.
    
    Protocol steps:
    1. Alice generates random bits and random bases
    2. Alice encodes qubits and sends to Bob
    3. Bob measures with random bases
    4. Alice and Bob compare bases (public)
    5. Discard bits where bases differ
    6. Perform error estimation and privacy amplification
    """
    
    def __init__(self, eavesdrop_probability: float = 0.0):
        self.eavesdrop_probability = eavesdrop_probability
        self.qber_threshold = 0.11  # Above 11% QBER, abort (theoretical limit)

    def generate_alice_bits(self, n: int) -> Tuple[List[int], List[Basis]]:
        """Alice generates random bits and bases"""
        bits = [secrets.randbelow(2) for _ in range(n)]
        bases = [random.choice(list(Basis)) for _ in range(n)]
        return bits, bases

    def encode_qubits(self, bits: List[int], bases: List[Basis]) -> List[Qubit]:
        """Alice encodes bits into qubits using her bases"""
        return [Qubit(bit, basis) for bit, basis in zip(bits, bases)]

    def simulate_channel(self, qubits: List[Qubit]) -> List[Qubit]:
        """
        Simulate quantum channel (with optional eavesdropper).
        Eve's measurement disturbs the qubits.
        """
        if self.eavesdrop_probability > 0:
            for qubit in qubits:
                if random.random() < self.eavesdrop_probability:
                    # Eve measures in random basis (disturbs qubit)
                    eve_basis = random.choice(list(Basis))
                    eve_measurement = qubit.measure(eve_basis)
                    # Eve retransmits (creates new qubit)
                    qubit.value = eve_measurement
                    qubit.basis = eve_basis
        
        return qubits

    def bob_measure(self, qubits: List[Qubit]) -> Tuple[List[int], List[Basis]]:
        """Bob measures qubits in random bases"""
        bob_bases = [random.choice(list(Basis)) for _ in range(len(qubits))]
        measurements = [q.measure(b) for q, b in zip(qubits, bob_bases)]
        return measurements, bob_bases

    def sift_keys(
        self,
        alice_bits: List[int],
        alice_bases: List[Basis],
        bob_bits: List[int],
        bob_bases: List[Basis]
    ) -> Tuple[List[int], List[int]]:
        """
        Sift keys - keep only bits where bases matched.
        This step is done over classical channel.
        """
        alice_sifted = []
        bob_sifted = []
        
        for a_bit, a_basis, b_bit, b_basis in zip(alice_bits, alice_bases, bob_bits, bob_bases):
            if a_basis == b_basis:
                alice_sifted.append(a_bit)
                bob_sifted.append(b_bit)
        
        return alice_sifted, bob_sifted

    def estimate_qber(self, alice_bits: List[int], bob_bits: List[int], sample_size: int = None) -> float:
        """
        Estimate Quantum Bit Error Rate using a sample.
        High QBER indicates eavesdropping.
        """
        if sample_size is None:
            sample_size = min(len(alice_bits) // 4, 100)
        
        if len(alice_bits) == 0:
            return 1.0
        
        # Sample random positions to check
        check_positions = random.sample(range(len(alice_bits)), min(sample_size, len(alice_bits)))
        
        errors = sum(1 for i in check_positions if alice_bits[i] != bob_bits[i])
        qber = errors / len(check_positions) if check_positions else 0.0
        
        return qber

    def privacy_amplification(self, key_bits: List[int], security_parameter: int = 128) -> bytes:
        """
        Privacy amplification using hash function.
        Reduces key length but increases security margin.
        """
        # Convert bits to bytes
        key_bytes = bytes(
            sum(bit << (7 - i % 8) for i, bit in enumerate(key_bits[j*8:(j+1)*8]))
            for j in range(len(key_bits) // 8)
        )
        
        # Apply hash for privacy amplification
        h = hashlib.sha256(key_bytes)
        final_key = h.digest()[:security_parameter // 8]
        
        return final_key

    def exchange_key(self, num_qubits: int = 1024) -> QKDResult:
        """
        Complete BB84 key exchange.
        
        Args:
            num_qubits: Number of qubits to exchange
            
        Returns:
            QKDResult with shared key and statistics
        """
        # Step 1: Alice generates random bits and bases
        alice_bits, alice_bases = self.generate_alice_bits(num_qubits)
        
        # Step 2: Alice encodes qubits
        qubits = self.encode_qubits(alice_bits, alice_bases)
        
        # Step 3: Simulate quantum channel (with potential eavesdropping)
        received_qubits = self.simulate_channel(qubits)
        
        # Step 4: Bob measures
        bob_bits, bob_bases = self.bob_measure(received_qubits)
        
        # Step 5: Sift keys
        alice_sifted, bob_sifted = self.sift_keys(alice_bits, alice_bases, bob_bits, bob_bases)
        
        # Step 6: Estimate QBER
        qber = self.estimate_qber(alice_sifted, bob_sifted)
        
        # Check if channel is secure
        secure = qber < self.qber_threshold
        
        if not secure:
            return QKDResult(
                shared_key=b"",
                key_length_bits=0,
                qber=qber,
                secure=False,
                raw_key_length=len(alice_sifted),
                final_key_length=0
            )
        
        # Step 7: Privacy amplification (using remaining bits after QBER check)
        remaining_bits = [b for i, b in enumerate(alice_sifted) if i >= len(alice_sifted) // 4]
        final_key = self.privacy_amplification(remaining_bits)
        
        return QKDResult(
            shared_key=final_key,
            key_length_bits=len(final_key) * 8,
            qber=qber,
            secure=True,
            raw_key_length=len(alice_sifted),
            final_key_length=len(final_key) * 8
        )


class QKDKeyManager:
    """
    Manages quantum-derived keys for the voting system.
    """
    
    def __init__(self):
        self.protocol = BB84Protocol()
        self.key_cache = {}
        
    def generate_election_master_key(self, election_id: str, num_qubits: int = 2048) -> Optional[bytes]:
        """
        Generate quantum-secure master key for an election.
        """
        result = self.protocol.exchange_key(num_qubits)
        
        if not result.secure:
            raise SecurityError(f"QKD exchange failed - QBER {result.qber:.2%} exceeds threshold")
        
        self.key_cache[election_id] = result.shared_key
        return result.shared_key
    
    def derive_ballot_key(self, election_id: str, ballot_id: str) -> bytes:
        """
        Derive ballot-specific key from election master key.
        """
        master_key = self.key_cache.get(election_id)
        if not master_key:
            raise KeyError(f"No master key for election {election_id}")
        
        # HKDF-like derivation
        h = hashlib.sha256(master_key + ballot_id.encode())
        return h.digest()


class SecurityError(Exception):
    """Raised when security checks fail"""
    pass


if __name__ == "__main__":
    print("=== BB84 QKD Protocol Simulation ===\n")
    
    # Test without eavesdropper
    print("Test 1: Normal exchange (no eavesdropper)")
    protocol = BB84Protocol(eavesdrop_probability=0.0)
    result = protocol.exchange_key(1024)
    print(f"  Secure: {result.secure}")
    print(f"  QBER: {result.qber:.2%}")
    print(f"  Raw key length: {result.raw_key_length} bits")
    print(f"  Final key length: {result.final_key_length} bits")
    print(f"  Key (hex): {result.shared_key.hex()[:32]}...")
    
    # Test with eavesdropper
    print("\nTest 2: Exchange with eavesdropper (25% interception)")
    protocol_eve = BB84Protocol(eavesdrop_probability=0.25)
    result_eve = protocol_eve.exchange_key(1024)
    print(f"  Secure: {result_eve.secure}")
    print(f"  QBER: {result_eve.qber:.2%}")
    if not result_eve.secure:
        print("  ⚠️ Eavesdropping detected! Key exchange aborted.")
