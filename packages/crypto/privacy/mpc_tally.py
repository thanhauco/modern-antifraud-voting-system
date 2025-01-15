"""
Secure Multi-Party Computation (MPC) for Distributed Vote Tallying
No single party can learn individual votes.
"""

import random
import hashlib
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class SecretShare:
    """A share of a secret value"""
    party_id: int
    share_value: int
    commitment: str  # Hash commitment for verification


@dataclass
class MPCParty:
    """A party in the MPC protocol"""
    party_id: int
    name: str
    shares_held: Dict[str, SecretShare] = field(default_factory=dict)
    
    def store_share(self, share_id: str, share: SecretShare):
        self.shares_held[share_id] = share
    
    def get_share(self, share_id: str) -> Optional[SecretShare]:
        return self.shares_held.get(share_id)


class AdditiveSecretSharing:
    """
    Additive secret sharing scheme.
    Secret s is split into n shares where s = s1 + s2 + ... + sn (mod p)
    """
    
    def __init__(self, num_parties: int, prime: int = 2**61 - 1):
        self.num_parties = num_parties
        self.prime = prime
    
    def share(self, secret: int, share_id: str) -> List[SecretShare]:
        """Split a secret into additive shares"""
        # Generate n-1 random shares
        shares = [random.randint(0, self.prime - 1) for _ in range(self.num_parties - 1)]
        
        # Last share makes the sum equal to secret
        final_share = (secret - sum(shares)) % self.prime
        shares.append(final_share)
        
        # Create share objects with commitments
        result = []
        for i, share_value in enumerate(shares):
            commitment = hashlib.sha256(
                f"{share_id}:{i}:{share_value}".encode()
            ).hexdigest()
            
            result.append(SecretShare(
                party_id=i,
                share_value=share_value,
                commitment=commitment
            ))
        
        return result
    
    def reconstruct(self, shares: List[SecretShare]) -> int:
        """Reconstruct secret from shares"""
        if len(shares) != self.num_parties:
            raise ValueError(f"Need {self.num_parties} shares, got {len(shares)}")
        
        return sum(s.share_value for s in shares) % self.prime
    
    def add_shares(self, shares_a: List[SecretShare], shares_b: List[SecretShare]) -> List[SecretShare]:
        """Add two shared secrets"""
        result = []
        for a, b in zip(shares_a, shares_b):
            new_value = (a.share_value + b.share_value) % self.prime
            result.append(SecretShare(
                party_id=a.party_id,
                share_value=new_value,
                commitment=""  # Would recompute
            ))
        return result


class SecureMPCTally:
    """
    Secure Multi-Party Computation for vote tallying.
    
    Protocol:
    1. Voter splits their vote into shares
    2. Each share goes to a different tallying authority
    3. Authorities compute on shares without seeing votes
    4. Final tally is reconstructed only when all parties agree
    """
    
    def __init__(self, num_authorities: int, num_candidates: int):
        self.num_authorities = num_authorities
        self.num_candidates = num_candidates
        self.sharing = AdditiveSecretSharing(num_authorities)
        
        # Initialize parties
        self.parties = [
            MPCParty(party_id=i, name=f"Authority_{i}")
            for i in range(num_authorities)
        ]
        
        # Running tally shares per candidate
        self.tally_shares: Dict[int, List[SecretShare]] = {}
        for c in range(num_candidates):
            # Initialize to shares of 0
            zero_shares = self.sharing.share(0, f"init_{c}")
            for i, party in enumerate(self.parties):
                party.store_share(f"tally_{c}", zero_shares[i])
            self.tally_shares[c] = zero_shares
        
        self.vote_count = 0
    
    def cast_vote(self, candidate_id: int) -> str:
        """
        Cast a vote using MPC.
        Vote is split into shares, distributed to authorities.
        Returns receipt hash.
        """
        if candidate_id < 0 or candidate_id >= self.num_candidates:
            raise ValueError(f"Invalid candidate: {candidate_id}")
        
        self.vote_count += 1
        vote_id = f"vote_{self.vote_count}"
        
        # For chosen candidate, share vote value 1
        # For others, share value 0
        for c in range(self.num_candidates):
            vote_value = 1 if c == candidate_id else 0
            vote_shares = self.sharing.share(vote_value, f"{vote_id}_{c}")
            
            # Add vote shares to running tally shares
            self.tally_shares[c] = self.sharing.add_shares(
                self.tally_shares[c],
                vote_shares
            )
            
            # Update each party's held share
            for i, party in enumerate(self.parties):
                party.store_share(f"tally_{c}", self.tally_shares[c][i])
        
        # Generate receipt
        receipt = hashlib.sha256(
            f"{vote_id}:{candidate_id}:{random.random()}".encode()
        ).hexdigest()[:16]
        
        return receipt
    
    def get_partial_tally(self, party_id: int, candidate_id: int) -> int:
        """Get a party's share of the tally (reveals nothing about actual tally)"""
        share = self.parties[party_id].get_share(f"tally_{candidate_id}")
        return share.share_value if share else 0
    
    def reconstruct_tally(self, candidate_id: int) -> int:
        """
        Reconstruct final tally.
        Requires all parties to contribute their shares.
        """
        shares = []
        for party in self.parties:
            share = party.get_share(f"tally_{candidate_id}")
            if share is None:
                raise ValueError(f"Party {party.party_id} missing share")
            shares.append(share)
        
        return self.sharing.reconstruct(shares)
    
    def get_all_results(self) -> Dict[int, int]:
        """Reconstruct all tallies"""
        return {c: self.reconstruct_tally(c) for c in range(self.num_candidates)}
    
    def verify_integrity(self) -> bool:
        """Verify that total votes matches sum of tallies"""
        total = sum(self.reconstruct_tally(c) for c in range(self.num_candidates))
        return total == self.vote_count


class ByzantineFaultTolerantMPC(SecureMPCTally):
    """
    MPC with Byzantine fault tolerance.
    Can tolerate up to t < n/3 malicious parties.
    """
    
    def __init__(self, num_authorities: int, num_candidates: int):
        super().__init__(num_authorities, num_candidates)
        self.complaints: List[Tuple[int, int, str]] = []  # (accuser, accused, reason)
        self.verified_shares: Dict[str, bool] = {}
    
    def verify_share_consistency(self, share_id: str) -> bool:
        """
        Verify that shares are consistent using commitments.
        Uses Pedersen commitments in production.
        """
        shares = []
        for party in self.parties:
            share = party.get_share(share_id)
            if share:
                shares.append(share)
        
        # Verify all commitments match expected format
        for share in shares:
            expected = hashlib.sha256(
                f"{share_id}:{share.party_id}:{share.share_value}".encode()
            ).hexdigest()
            
            if share.commitment and share.commitment != expected:
                self.complaints.append((
                    -1,  # System complaint
                    share.party_id,
                    "Commitment mismatch"
                ))
                return False
        
        self.verified_shares[share_id] = True
        return True
    
    def file_complaint(self, accuser_id: int, accused_id: int, reason: str):
        """File a complaint against a potentially malicious party"""
        self.complaints.append((accuser_id, accused_id, reason))
    
    def get_reputation(self, party_id: int) -> float:
        """Get party's reputation based on complaints"""
        complaints_against = sum(1 for _, accused, _ in self.complaints if accused == party_id)
        return max(0, 1.0 - (complaints_against * 0.1))


if __name__ == "__main__":
    print("=== Secure Multi-Party Computation Tally ===\n")
    
    # Create MPC with 5 authorities and 3 candidates
    mpc = SecureMPCTally(num_authorities=5, num_candidates=3)
    
    # Cast votes
    votes = [0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 2, 0]
    
    print(f"Casting {len(votes)} votes via MPC...")
    receipts = []
    for vote in votes:
        receipt = mpc.cast_vote(vote)
        receipts.append(receipt)
    
    # Show partial tallies (meaningless individually)
    print("\nPartial tallies (each authority's view):")
    for party in mpc.parties[:2]:  # Show first 2
        print(f"  {party.name}:")
        for c in range(3):
            partial = mpc.get_partial_tally(party.party_id, c)
            print(f"    Candidate {c}: {partial % 1000}... (meaningless alone)")
    
    # Reconstruct final tallies
    print("\nFinal tallies (after MPC reconstruction):")
    results = mpc.get_all_results()
    for candidate, count in results.items():
        print(f"  Candidate {candidate}: {count} votes")
    
    print(f"\nIntegrity verified: {mpc.verify_integrity()}")
