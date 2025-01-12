"""
Homomorphic Encryption for Privacy-Preserving Vote Tallying
Compute on encrypted votes without decryption.
"""

import random
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass
from functools import reduce

@dataclass
class PaillierPublicKey:
    """Paillier public key"""
    n: int       # n = p * q
    g: int       # generator
    n_squared: int
    
    def encrypt(self, plaintext: int) -> int:
        """Encrypt a value"""
        r = random.randint(1, self.n - 1)
        while math.gcd(r, self.n) != 1:
            r = random.randint(1, self.n - 1)
        
        c = (pow(self.g, plaintext, self.n_squared) * 
             pow(r, self.n, self.n_squared)) % self.n_squared
        return c
    
    def add_encrypted(self, c1: int, c2: int) -> int:
        """Homomorphic addition: E(m1) * E(m2) = E(m1 + m2)"""
        return (c1 * c2) % self.n_squared
    
    def multiply_constant(self, c: int, k: int) -> int:
        """Scalar multiplication: E(m)^k = E(m * k)"""
        return pow(c, k, self.n_squared)


@dataclass
class PaillierPrivateKey:
    """Paillier private key"""
    lambda_: int  # lcm(p-1, q-1)
    mu: int       # modular multiplicative inverse
    n: int
    n_squared: int
    
    def decrypt(self, ciphertext: int) -> int:
        """Decrypt a ciphertext"""
        x = pow(ciphertext, self.lambda_, self.n_squared)
        l = (x - 1) // self.n
        return (l * self.mu) % self.n


class PaillierCrypto:
    """
    Paillier Cryptosystem for homomorphic encryption.
    Supports addition of encrypted values.
    """
    
    @staticmethod
    def generate_keypair(bits: int = 512) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
        """Generate Paillier key pair"""
        # For demo, use small primes. Production needs cryptographic primes.
        p = PaillierCrypto._generate_prime(bits // 2)
        q = PaillierCrypto._generate_prime(bits // 2)
        
        n = p * q
        n_squared = n * n
        g = n + 1  # Simplified generator
        
        lambda_ = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
        
        # Calculate mu
        x = pow(g, lambda_, n_squared)
        l = (x - 1) // n
        mu = pow(l, -1, n)
        
        public_key = PaillierPublicKey(n=n, g=g, n_squared=n_squared)
        private_key = PaillierPrivateKey(lambda_=lambda_, mu=mu, n=n, n_squared=n_squared)
        
        return public_key, private_key
    
    @staticmethod
    def _generate_prime(bits: int) -> int:
        """Generate a prime number (simplified for demo)"""
        # In production, use cryptographic prime generation
        candidates = [i for i in range(2**(bits-1), 2**bits) if PaillierCrypto._is_prime(i)]
        return random.choice(candidates[:100]) if candidates else 257
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True


class HomomorphicVoteTally:
    """
    Privacy-preserving vote tallying using homomorphic encryption.
    
    Each candidate's vote count is encrypted.
    Votes can be added without decryption.
    Only election officials with the private key can decrypt final tally.
    """
    
    def __init__(self, num_candidates: int):
        self.num_candidates = num_candidates
        self.public_key, self.private_key = PaillierCrypto.generate_keypair(bits=64)
        
        # Encrypted tallies (initialized to E(0))
        self.encrypted_tallies: Dict[int, int] = {}
        for i in range(num_candidates):
            self.encrypted_tallies[i] = self.public_key.encrypt(0)
        
        self.total_votes = 0
    
    def cast_vote(self, candidate_id: int) -> int:
        """
        Cast an encrypted vote for a candidate.
        Returns encrypted ballot for verification.
        """
        if candidate_id < 0 or candidate_id >= self.num_candidates:
            raise ValueError(f"Invalid candidate ID: {candidate_id}")
        
        # Create encrypted ballot: E(1) for chosen candidate, E(0) for others
        encrypted_vote = self.public_key.encrypt(1)
        
        # Homomorphically add to running tally
        self.encrypted_tallies[candidate_id] = self.public_key.add_encrypted(
            self.encrypted_tallies[candidate_id],
            encrypted_vote
        )
        
        self.total_votes += 1
        return encrypted_vote
    
    def get_encrypted_tally(self, candidate_id: int) -> int:
        """Get encrypted tally for a candidate (can be verified without decryption)"""
        return self.encrypted_tallies.get(candidate_id, 0)
    
    def decrypt_tally(self, candidate_id: int) -> int:
        """Decrypt final tally (requires private key)"""
        encrypted = self.encrypted_tallies.get(candidate_id)
        if encrypted is None:
            return 0
        return self.private_key.decrypt(encrypted)
    
    def get_all_results(self) -> Dict[int, int]:
        """Decrypt and return all results"""
        return {i: self.decrypt_tally(i) for i in range(self.num_candidates)}
    
    def verify_total(self) -> bool:
        """Verify that sum of decrypted tallies equals total votes"""
        decrypted_sum = sum(self.decrypt_tally(i) for i in range(self.num_candidates))
        return decrypted_sum == self.total_votes


class ThresholdDecryption:
    """
    Threshold decryption: requires n-of-m key holders to decrypt.
    Prevents single point of compromise.
    """
    
    def __init__(self, threshold: int, total_holders: int):
        self.threshold = threshold
        self.total_holders = total_holders
        self.key_shares: List[int] = []
        self.partial_decryptions: Dict[int, int] = {}
    
    def split_key(self, private_key: int) -> List[Tuple[int, int]]:
        """
        Split private key into shares using Shamir's Secret Sharing.
        Returns list of (holder_id, share) tuples.
        """
        # Generate random polynomial coefficients
        prime = 2**61 - 1  # Large prime
        coefficients = [private_key] + [random.randint(1, prime-1) for _ in range(self.threshold - 1)]
        
        shares = []
        for i in range(1, self.total_holders + 1):
            # Evaluate polynomial at point i
            share = sum(c * pow(i, j, prime) for j, c in enumerate(coefficients)) % prime
            shares.append((i, share))
        
        return shares
    
    def submit_partial_decryption(self, holder_id: int, partial: int):
        """Submit partial decryption from a key holder"""
        self.partial_decryptions[holder_id] = partial
    
    def can_decrypt(self) -> bool:
        """Check if enough shares collected"""
        return len(self.partial_decryptions) >= self.threshold
    
    def reconstruct_decryption(self) -> int:
        """Combine partial decryptions to get final result"""
        if not self.can_decrypt():
            raise ValueError(f"Need {self.threshold} shares, have {len(self.partial_decryptions)}")
        
        # Lagrange interpolation at x=0
        prime = 2**61 - 1
        result = 0
        
        points = list(self.partial_decryptions.items())[:self.threshold]
        
        for i, (xi, yi) in enumerate(points):
            numerator = denominator = 1
            for j, (xj, _) in enumerate(points):
                if i != j:
                    numerator = (numerator * (-xj)) % prime
                    denominator = (denominator * (xi - xj)) % prime
            
            lagrange = (yi * numerator * pow(denominator, -1, prime)) % prime
            result = (result + lagrange) % prime
        
        return result


if __name__ == "__main__":
    print("=== Homomorphic Vote Tallying ===\n")
    
    # Create election with 3 candidates
    election = HomomorphicVoteTally(num_candidates=3)
    
    # Cast votes (voters don't see each other's votes)
    votes = [0, 1, 0, 2, 1, 0, 0, 1, 2, 0]  # 5 for 0, 3 for 1, 2 for 2
    
    print(f"Casting {len(votes)} votes...")
    for vote in votes:
        election.cast_vote(vote)
    
    # Before decryption, tallies are encrypted
    print("\nEncrypted tallies (meaningless without key):")
    for i in range(3):
        print(f"  Candidate {i}: {election.get_encrypted_tally(i) % 10000}...")
    
    # Decrypt results
    print("\nDecrypted results:")
    results = election.get_all_results()
    for candidate, count in results.items():
        print(f"  Candidate {candidate}: {count} votes")
    
    # Verify integrity
    print(f"\nIntegrity check: {election.verify_total()}")
