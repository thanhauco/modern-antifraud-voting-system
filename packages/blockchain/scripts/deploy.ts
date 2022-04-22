import { ethers } from "hardhat";

async function main() {
  console.log("Deploying Modern Anti-Fraud Voting System contracts...");
  
  // Deploy VoteRegistry
  const VoteRegistry = await ethers.getContractFactory("VoteRegistry");
  const voteRegistry = await VoteRegistry.deploy();
  await voteRegistry.waitForDeployment();
  console.log(`VoteRegistry deployed to: ${await voteRegistry.getAddress()}`);
  
  // Deploy ElectionManager
  const ElectionManager = await ethers.getContractFactory("ElectionManager");
  const electionManager = await ElectionManager.deploy();
  await electionManager.waitForDeployment();
  console.log(`ElectionManager deployed to: ${await electionManager.getAddress()}`);
  
  console.log("\nDeployment complete!");
  console.log("---");
  console.log("Contract Addresses:");
  console.log(`  VoteRegistry: ${await voteRegistry.getAddress()}`);
  console.log(`  ElectionManager: ${await electionManager.getAddress()}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
