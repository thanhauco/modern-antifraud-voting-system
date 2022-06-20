// Fixed: Now properly loads verification key before verification
import { VoteVerifier } from "./verifier";

export async function quickVerify(proof: any, signals: string[]): Promise<boolean> {
  const verifier = new VoteVerifier({ verificationKeyPath: "./build/vkey.json" });
  // FIX: Load verification key before verifying
  await verifier.loadVerificationKey();
  return await verifier.verifyProof(proof, signals);
}
