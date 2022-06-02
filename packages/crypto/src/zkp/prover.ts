import { groth16 } from "snarkjs";
import { readFileSync } from "fs";
import path from "path";

export interface VoteProofInput {
  voterSecret: bigint;
  candidateId: number;
  electionId: bigint;
  numCandidates: number;
}

export interface VoteProof {
  proof: {
    pi_a: string[];
    pi_b: string[][];
    pi_c: string[];
    protocol: string;
    curve: string;
  };
  publicSignals: string[];
}

export interface ProverConfig {
  wasmPath: string;
  zkeyPath: string;
}

/**
 * ZK Proof generator for vote validity
 */
export class VoteProver {
  private wasmPath: string;
  private zkeyPath: string;

  constructor(config: ProverConfig) {
    this.wasmPath = config.wasmPath;
    this.zkeyPath = config.zkeyPath;
  }

  /**
   * Generates a ZK proof for a vote
   */
  async generateProof(input: VoteProofInput): Promise<VoteProof> {
    // Compute vote commitment: Poseidon(voterSecret, candidateId, electionId)
    const voteCommitment = await this.computePoseidonHash([
      input.voterSecret,
      BigInt(input.candidateId),
      input.electionId,
    ]);

    // Compute nullifier: Poseidon(voterSecret, electionId)
    const nullifierHash = await this.computePoseidonHash([
      input.voterSecret,
      input.electionId,
    ]);

    const circuitInput = {
      voteCommitment: voteCommitment.toString(),
      electionId: input.electionId.toString(),
      nullifierHash: nullifierHash.toString(),
      voterSecret: input.voterSecret.toString(),
      candidateId: input.candidateId,
      numCandidates: input.numCandidates,
    };

    const { proof, publicSignals } = await groth16.fullProve(
      circuitInput,
      this.wasmPath,
      this.zkeyPath
    );

    return {
      proof,
      publicSignals,
    };
  }

  /**
   * Computes Poseidon hash (placeholder - actual implementation uses circomlibjs)
   */
  private async computePoseidonHash(inputs: bigint[]): Promise<bigint> {
    // In production, use circomlibjs Poseidon
    // This is a placeholder that would be replaced with actual Poseidon implementation
    const { buildPoseidon } = await import("circomlibjs");
    const poseidon = await buildPoseidon();
    const hash = poseidon(inputs);
    return poseidon.F.toObject(hash);
  }

  /**
   * Generates the vote commitment for on-chain verification
   */
  async getVoteCommitment(input: VoteProofInput): Promise<string> {
    const commitment = await this.computePoseidonHash([
      input.voterSecret,
      BigInt(input.candidateId),
      input.electionId,
    ]);
    return commitment.toString();
  }

  /**
   * Generates nullifier hash for double-vote prevention
   */
  async getNullifierHash(voterSecret: bigint, electionId: bigint): Promise<string> {
    const nullifier = await this.computePoseidonHash([voterSecret, electionId]);
    return nullifier.toString();
  }
}

/**
 * Factory function to create a configured prover
 */
export function createVoteProver(buildDir: string = "./build"): VoteProver {
  return new VoteProver({
    wasmPath: path.join(buildDir, "vote_validity_js", "vote_validity.wasm"),
    zkeyPath: path.join(buildDir, "vote_validity.zkey"),
  });
}
