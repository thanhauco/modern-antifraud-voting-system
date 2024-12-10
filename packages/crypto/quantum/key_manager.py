"""
Quantum-Safe Key Manager
Handles key lifecycle with crypto-agility for quantum migration.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import secrets

class KeyAlgorithm(Enum):
    """Supported key algorithms with quantum safety level"""
    # Classical (vulnerable to quantum)
    RSA_2048 = ("RSA-2048", False, "legacy")
    RSA_4096 = ("RSA-4096", False, "legacy")
    ECDSA_P256 = ("ECDSA-P256", False, "legacy")
    
    # Post-Quantum (NIST approved)
    ML_KEM_768 = ("ML-KEM-768", True, "pqc")
    ML_KEM_1024 = ("ML-KEM-1024", True, "pqc")
    ML_DSA_65 = ("ML-DSA-65", True, "pqc")
    ML_DSA_87 = ("ML-DSA-87", True, "pqc")
    SLH_DSA_256 = ("SLH-DSA-256", True, "pqc")
    
    # Hybrid (classical + PQC)
    HYBRID_RSA_MLKEM = ("RSA+ML-KEM", True, "hybrid")
    HYBRID_ECDSA_MLDSA = ("ECDSA+ML-DSA", True, "hybrid")


@dataclass
class KeyMetadata:
    """Metadata for managed keys"""
    key_id: str
    algorithm: KeyAlgorithm
    created_at: datetime
    expires_at: datetime
    purpose: str  # "encryption", "signing", "key_agreement"
    quantum_safe: bool
    status: str = "active"  # active, expired, revoked, pending_rotation
    rotation_policy: str = "annual"
    predecessor_id: Optional[str] = None
    successor_id: Optional[str] = None


@dataclass
class ManagedKey:
    """A managed cryptographic key"""
    metadata: KeyMetadata
    public_key: bytes
    encrypted_private_key: bytes  # Encrypted with master key
    key_encryption_key_id: str


class QuantumSafeKeyManager:
    """
    Enterprise key manager with quantum safety and crypto-agility.
    
    Features:
    - Automatic key rotation
    - Crypto-agility for algorithm migration
    - Hybrid classical/PQC mode during transition
    - HSM integration support
    - Key versioning and rollback
    - Quantum-safe key derivation
    """
    
    def __init__(self, master_key: bytes = None):
        self.master_key = master_key or secrets.token_bytes(32)
        self.keys: Dict[str, ManagedKey] = {}
        self.algorithm_policy = KeyAlgorithm.HYBRID_RSA_MLKEM
        self.key_rotation_days = 365
        self.migration_mode = "hybrid"  # "classical", "hybrid", "pqc_only"

    def generate_key(self, 
                     algorithm: KeyAlgorithm,
                     purpose: str,
                     validity_days: int = 365) -> str:
        """
        Generate a new managed key.
        
        Args:
            algorithm: Key algorithm to use
            purpose: Key purpose (encryption, signing, key_agreement)
            validity_days: Key validity in days
            
        Returns:
            Key ID
        """
        key_id = self._generate_key_id()
        
        now = datetime.utcnow()
        expires = now + timedelta(days=validity_days)
        
        # Generate key material (simplified)
        public_key, private_key = self._generate_key_pair(algorithm)
        
        # Encrypt private key
        encrypted_private = self._encrypt_with_master(private_key)
        
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            created_at=now,
            expires_at=expires,
            purpose=purpose,
            quantum_safe=algorithm.value[1],
            rotation_policy=f"{validity_days}d"
        )
        
        managed_key = ManagedKey(
            metadata=metadata,
            public_key=public_key,
            encrypted_private_key=encrypted_private,
            key_encryption_key_id="master"
        )
        
        self.keys[key_id] = managed_key
        return key_id

    def rotate_key(self, key_id: str, new_algorithm: KeyAlgorithm = None) -> str:
        """
        Rotate a key, optionally upgrading algorithm.
        
        This enables seamless migration from classical to PQC.
        """
        old_key = self.keys.get(key_id)
        if not old_key:
            raise KeyError(f"Key {key_id} not found")
        
        # Use new algorithm or keep existing
        algorithm = new_algorithm or old_key.metadata.algorithm
        
        # Generate replacement
        new_key_id = self.generate_key(
            algorithm=algorithm,
            purpose=old_key.metadata.purpose,
            validity_days=self.key_rotation_days
        )
        
        # Link keys
        old_key.metadata.successor_id = new_key_id
        old_key.metadata.status = "pending_rotation"
        self.keys[new_key_id].metadata.predecessor_id = key_id
        
        return new_key_id

    def migrate_to_quantum_safe(self) -> Dict[str, str]:
        """
        Migrate all classical keys to quantum-safe algorithms.
        
        Returns mapping of old_key_id -> new_key_id
        """
        migrations = {}
        
        for key_id, key in list(self.keys.items()):
            if not key.metadata.quantum_safe and key.metadata.status == "active":
                # Determine target algorithm
                if key.metadata.purpose == "encryption":
                    target = KeyAlgorithm.ML_KEM_1024
                else:
                    target = KeyAlgorithm.ML_DSA_87
                
                # If in hybrid mode, use hybrid algorithm
                if self.migration_mode == "hybrid":
                    if key.metadata.purpose == "encryption":
                        target = KeyAlgorithm.HYBRID_RSA_MLKEM
                    else:
                        target = KeyAlgorithm.HYBRID_ECDSA_MLDSA
                
                new_key_id = self.rotate_key(key_id, target)
                migrations[key_id] = new_key_id
        
        return migrations

    def get_active_keys(self, purpose: str = None) -> List[ManagedKey]:
        """Get all active keys, optionally filtered by purpose"""
        active = [k for k in self.keys.values() if k.metadata.status == "active"]
        if purpose:
            active = [k for k in active if k.metadata.purpose == purpose]
        return active

    def check_expiring_keys(self, days_threshold: int = 30) -> List[KeyMetadata]:
        """Find keys expiring within threshold"""
        now = datetime.utcnow()
        threshold = now + timedelta(days=days_threshold)
        
        expiring = []
        for key in self.keys.values():
            if key.metadata.status == "active" and key.metadata.expires_at < threshold:
                expiring.append(key.metadata)
        
        return expiring

    def revoke_key(self, key_id: str, reason: str = "unspecified"):
        """Revoke a key"""
        key = self.keys.get(key_id)
        if key:
            key.metadata.status = "revoked"

    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        random_bytes = secrets.token_bytes(16)
        return f"key_{hashlib.sha256(random_bytes).hexdigest()[:24]}"

    def _generate_key_pair(self, algorithm: KeyAlgorithm) -> tuple:
        """Generate key pair for algorithm (simplified)"""
        # In production, would use actual crypto libraries
        if algorithm.value[2] == "pqc":
            # PQC key sizes
            public_key = secrets.token_bytes(1952)
            private_key = secrets.token_bytes(4032)
        elif algorithm.value[2] == "hybrid":
            # Hybrid has both classical and PQC components
            public_key = secrets.token_bytes(2500)
            private_key = secrets.token_bytes(5000)
        else:
            # Classical
            public_key = secrets.token_bytes(256)
            private_key = secrets.token_bytes(512)
        
        return public_key, private_key

    def _encrypt_with_master(self, data: bytes) -> bytes:
        """Encrypt with master key (simplified)"""
        # In production, use AES-GCM with the master key
        # This is a placeholder
        h = hashlib.sha256(self.master_key + data)
        return data + h.digest()

    def get_quantum_readiness_report(self) -> dict:
        """Generate quantum readiness assessment"""
        total_keys = len(self.keys)
        quantum_safe = sum(1 for k in self.keys.values() if k.metadata.quantum_safe)
        classical = total_keys - quantum_safe
        
        return {
            "total_keys": total_keys,
            "quantum_safe_keys": quantum_safe,
            "classical_keys": classical,
            "quantum_readiness_pct": (quantum_safe / total_keys * 100) if total_keys > 0 else 0,
            "migration_mode": self.migration_mode,
            "recommended_action": "Complete" if classical == 0 else f"Migrate {classical} classical keys"
        }


if __name__ == "__main__":
    print("=== Quantum-Safe Key Manager ===\n")
    
    manager = QuantumSafeKeyManager()
    
    # Generate classical keys (legacy)
    print("Creating legacy keys...")
    rsa_key = manager.generate_key(KeyAlgorithm.RSA_4096, "encryption")
    ecdsa_key = manager.generate_key(KeyAlgorithm.ECDSA_P256, "signing")
    
    # Generate PQC keys
    print("Creating quantum-safe keys...")
    pqc_enc = manager.generate_key(KeyAlgorithm.ML_KEM_768, "encryption")
    pqc_sig = manager.generate_key(KeyAlgorithm.ML_DSA_65, "signing")
    
    # Check readiness
    report = manager.get_quantum_readiness_report()
    print(f"\nQuantum Readiness: {report['quantum_readiness_pct']:.1f}%")
    print(f"  Quantum-safe: {report['quantum_safe_keys']}")
    print(f"  Classical: {report['classical_keys']}")
    print(f"  Recommendation: {report['recommended_action']}")
    
    # Migrate to quantum-safe
    print("\nMigrating to quantum-safe...")
    migrations = manager.migrate_to_quantum_safe()
    print(f"  Migrated {len(migrations)} keys")
    
    # Final report
    report = manager.get_quantum_readiness_report()
    print(f"\nFinal Quantum Readiness: {report['quantum_readiness_pct']:.1f}%")
