// Crypto package main exports
export * from "./zkp/prover";
export * from "./zkp/verifier";
export * from "./pqc/index";

// Re-export types
export type { VoteProof, VoteProofInput, ProverConfig } from "./zkp/prover";
export type { VerificationKey, VerifierConfig } from "./zkp/verifier";
export type { KeyPair, EncapsulationResult, Signature } from "./pqc/index";
