import { groth16 } from "snarkjs";
import { readFileSync } from "fs";
import path from "path";

export interface VerificationKey {
  protocol: string;
  curve: string;
  nPublic: number;
  vk_alpha_1: string[];
  vk_beta_2: string[][];
  vk_gamma_2: string[][];
  vk_delta_2: string[][];
  vk_alphabeta_12: string[][][];
  IC: string[][];
}

export interface VerifierConfig {
  verificationKeyPath: string;
}

/**
 * ZK Proof verifier for vote validity
 * Can be used both off-chain and to generate on-chain verifier
 */
export class VoteVerifier {
  private verificationKey: VerificationKey | null = null;
  private verificationKeyPath: string;

  constructor(config: VerifierConfig) {
    this.verificationKeyPath = config.verificationKeyPath;
  }

  /**
   * Loads the verification key from file
   */
  async loadVerificationKey(): Promise<void> {
    const vkeyJson = readFileSync(this.verificationKeyPath, "utf-8");
    this.verificationKey = JSON.parse(vkeyJson);
  }

  /**
   * Verifies a ZK proof
   * @returns true if the proof is valid
   */
  async verifyProof(
    proof: {
      pi_a: string[];
      pi_b: string[][];
      pi_c: string[];
    },
    publicSignals: string[]
  ): Promise<boolean> {
    if (!this.verificationKey) {
      await this.loadVerificationKey();
    }

    try {
      const isValid = await groth16.verify(
        this.verificationKey,
        publicSignals,
        proof
      );
      return isValid;
    } catch (error) {
      console.error("Proof verification failed:", error);
      return false;
    }
  }

  /**
   * Extracts vote commitment from public signals
   */
  extractVoteCommitment(publicSignals: string[]): string {
    // Public signals order: [voteCommitment, electionId, nullifierHash]
    return publicSignals[0] ?? "";
  }

  /**
   * Extracts election ID from public signals
   */
  extractElectionId(publicSignals: string[]): string {
    return publicSignals[1] ?? "";
  }

  /**
   * Extracts nullifier hash from public signals
   */
  extractNullifierHash(publicSignals: string[]): string {
    return publicSignals[2] ?? "";
  }

  /**
   * Generates Solidity verifier contract code
   * Uses snarkjs to export verifier
   */
  async exportSolidityVerifier(zkeyPath: string): Promise<string> {
    const { zKey } = await import("snarkjs");
    const verifierCode = await zKey.exportSolidityVerifier(zkeyPath, {
      groth16,
    });
    return verifierCode;
  }
}

/**
 * Factory function to create a configured verifier
 */
export function createVoteVerifier(buildDir: string = "./build"): VoteVerifier {
  return new VoteVerifier({
    verificationKeyPath: path.join(buildDir, "vote_validity_verification_key.json"),
  });
}

/**
 * Verify a vote proof against expected values
 */
export async function verifyVoteProof(
  verifier: VoteVerifier,
  proof: { pi_a: string[]; pi_b: string[][]; pi_c: string[] },
  publicSignals: string[],
  expectedElectionId: string,
  usedNullifiers: Set<string>
): Promise<{ valid: boolean; reason?: string }> {
  // 1. Verify the ZK proof cryptographically
  const cryptoValid = await verifier.verifyProof(proof, publicSignals);
  if (!cryptoValid) {
    return { valid: false, reason: "Invalid cryptographic proof" };
  }

  // 2. Check election ID matches
  const proofElectionId = verifier.extractElectionId(publicSignals);
  if (proofElectionId !== expectedElectionId) {
    return { valid: false, reason: "Election ID mismatch" };
  }

  // 3. Check nullifier hasn't been used (double-vote prevention)
  const nullifier = verifier.extractNullifierHash(publicSignals);
  if (usedNullifiers.has(nullifier)) {
    return { valid: false, reason: "Vote already cast (duplicate nullifier)" };
  }

  return { valid: true };
}
