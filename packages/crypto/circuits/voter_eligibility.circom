pragma circom 2.1.6;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/eddsaposeidon.circom";
include "circomlib/circuits/comparators.circom";

/**
 * VoterEligibility Circuit
 * Proves voter is eligible without revealing identity
 * 
 * Uses EdDSA signature to prove possession of valid credentials
 * issued by election authority
 * 
 * Public inputs:
 * - merkleRoot: Root of voter registry Merkle tree
 * - electionId: The election being voted in
 * - authorityPubKey: Election authority's public key
 * 
 * Private inputs:
 * - voterCredential: Voter's credential hash
 * - voterSecret: Voter's secret key
 * - merkleProof: Proof of inclusion in voter registry
 * - authSignature: Authority's signature on voter credential
 */
template VoterEligibility(levels) {
    // Public inputs
    signal input merkleRoot;
    signal input electionId;
    signal input authorityPubKeyX;
    signal input authorityPubKeyY;
    
    // Private inputs
    signal input voterCredential;
    signal input voterSecret;
    signal input merklePathElements[levels];
    signal input merklePathIndices[levels];
    signal input authSignatureR8x;
    signal input authSignatureR8y;
    signal input authSignatureS;
    
    // Output
    signal output eligible;
    
    // 1. Compute voter leaf = Poseidon(voterCredential)
    component leafHasher = Poseidon(1);
    leafHasher.inputs[0] <== voterCredential;
    signal voterLeaf;
    voterLeaf <== leafHasher.out;
    
    // 2. Verify Merkle proof
    component merkleVerifier[levels];
    signal currentHash[levels + 1];
    currentHash[0] <== voterLeaf;
    
    for (var i = 0; i < levels; i++) {
        merkleVerifier[i] = Poseidon(2);
        
        // If index is 0, hash(current, sibling), else hash(sibling, current)
        component isZero = IsZero();
        isZero.in <== merklePathIndices[i];
        
        // Select order based on path index
        signal left;
        signal right;
        left <== isZero.out * currentHash[i] + (1 - isZero.out) * merklePathElements[i];
        right <== isZero.out * merklePathElements[i] + (1 - isZero.out) * currentHash[i];
        
        merkleVerifier[i].inputs[0] <== left;
        merkleVerifier[i].inputs[1] <== right;
        currentHash[i + 1] <== merkleVerifier[i].out;
    }
    
    // 3. Verify computed root matches public merkle root
    merkleRoot === currentHash[levels];
    
    // 4. Verify authority signature on credential
    component sigVerifier = EdDSAPoseidonVerifier();
    sigVerifier.enabled <== 1;
    sigVerifier.Ax <== authorityPubKeyX;
    sigVerifier.Ay <== authorityPubKeyY;
    sigVerifier.R8x <== authSignatureR8x;
    sigVerifier.R8y <== authSignatureR8y;
    sigVerifier.S <== authSignatureS;
    sigVerifier.M <== voterCredential;
    
    // Voter is eligible
    eligible <== 1;
}

// Main component with 20 levels (supports ~1M voters)
component main {public [merkleRoot, electionId, authorityPubKeyX, authorityPubKeyY]} = VoterEligibility(20);
