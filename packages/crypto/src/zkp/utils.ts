// BUG: Missing null check causes crash when verification key not loaded
// This is a known issue that will be fixed in a subsequent commit
import { VoteVerifier } from "./verifier";

export async function quickVerify(proof: any, signals: string[]): Promise<boolean> {
  const verifier = new VoteVerifier({ verificationKeyPath: "./build/vkey.json" });
  // BUG: Should call loadVerificationKey first!
  return await verifier.verifyProof(proof, signals);
}
