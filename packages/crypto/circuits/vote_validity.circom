pragma circom 2.1.6;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/bitify.circom";

/**
 * VoteValidity Circuit
 * Proves that a vote is valid without revealing the vote choice
 * 
 * Public inputs:
 * - voteCommitment: Poseidon hash of (voterSecret, candidateId, electionId)
 * - electionId: The election being voted in
 * - nullifierHash: Prevents double voting
 * 
 * Private inputs:
 * - voterSecret: Secret known only to the voter
 * - candidateId: The chosen candidate (0 to numCandidates-1)
 * - numCandidates: Total number of candidates
 */
template VoteValidity(maxCandidates) {
    // Public inputs
    signal input voteCommitment;
    signal input electionId;
    signal input nullifierHash;
    
    // Private inputs
    signal input voterSecret;
    signal input candidateId;
    signal input numCandidates;
    
    // Output
    signal output valid;
    
    // 1. Verify candidateId is within valid range [0, numCandidates)
    component ltNumCandidates = LessThan(8);
    ltNumCandidates.in[0] <== candidateId;
    ltNumCandidates.in[1] <== numCandidates;
    ltNumCandidates.out === 1;
    
    // 2. Verify candidateId is less than maxCandidates
    component ltMaxCandidates = LessThan(8);
    ltMaxCandidates.in[0] <== candidateId;
    ltMaxCandidates.in[1] <== maxCandidates;
    ltMaxCandidates.out === 1;
    
    // 3. Compute vote commitment = Poseidon(voterSecret, candidateId, electionId)
    component commitmentHasher = Poseidon(3);
    commitmentHasher.inputs[0] <== voterSecret;
    commitmentHasher.inputs[1] <== candidateId;
    commitmentHasher.inputs[2] <== electionId;
    
    // 4. Verify commitment matches public input
    voteCommitment === commitmentHasher.out;
    
    // 5. Compute nullifier = Poseidon(voterSecret, electionId)
    // This prevents double voting while maintaining anonymity
    component nullifierHasher = Poseidon(2);
    nullifierHasher.inputs[0] <== voterSecret;
    nullifierHasher.inputs[1] <== electionId;
    
    // 6. Verify nullifier matches public input
    nullifierHash === nullifierHasher.out;
    
    // Vote is valid
    valid <== 1;
}

// Main component with max 100 candidates
component main {public [voteCommitment, electionId, nullifierHash]} = VoteValidity(100);
