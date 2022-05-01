import { expect } from "chai";
import { ethers } from "hardhat";
import { VoteRegistry } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("VoteRegistry", function () {
  let voteRegistry: VoteRegistry;
  let owner: SignerWithAddress;
  let admin: SignerWithAddress;
  let voter: SignerWithAddress;
  
  const ELECTION_ID = ethers.encodeBytes32String("WA_GOV_2024");
  const JURISDICTION_ID = ethers.encodeBytes32String("WA");
  const VOTER_ID = ethers.encodeBytes32String("voter_001");
  
  beforeEach(async function () {
    [owner, admin, voter] = await ethers.getSigners();
    
    const VoteRegistryFactory = await ethers.getContractFactory("VoteRegistry");
    voteRegistry = await VoteRegistryFactory.deploy();
    await voteRegistry.waitForDeployment();
  });
  
  describe("Election Creation", function () {
    it("should create an election successfully", async function () {
      const now = Math.floor(Date.now() / 1000);
      const startTime = now + 3600; // 1 hour from now
      const endTime = now + 86400; // 24 hours from now
      
      await expect(
        voteRegistry.createElection(
          ELECTION_ID,
          "Washington Governor 2024",
          startTime,
          endTime,
          JURISDICTION_ID
        )
      ).to.emit(voteRegistry, "ElectionCreated");
      
      const election = await voteRegistry.elections(ELECTION_ID);
      expect(election.name).to.equal("Washington Governor 2024");
      expect(election.isActive).to.be.true;
    });
    
    it("should reject duplicate election creation", async function () {
      const now = Math.floor(Date.now() / 1000);
      const startTime = now + 3600;
      const endTime = now + 86400;
      
      await voteRegistry.createElection(
        ELECTION_ID,
        "Washington Governor 2024",
        startTime,
        endTime,
        JURISDICTION_ID
      );
      
      await expect(
        voteRegistry.createElection(
          ELECTION_ID,
          "Duplicate",
          startTime,
          endTime,
          JURISDICTION_ID
        )
      ).to.be.revertedWith("Election already exists");
    });
  });
  
  describe("Vote Casting", function () {
    beforeEach(async function () {
      const now = Math.floor(Date.now() / 1000);
      // Start immediately for testing
      await voteRegistry.createElection(
        ELECTION_ID,
        "Washington Governor 2024",
        now - 1,
        now + 86400,
        JURISDICTION_ID
      );
    });
    
    it("should cast a vote successfully", async function () {
      const voteHash = ethers.keccak256(ethers.toUtf8Bytes("encrypted_vote"));
      const zkProofHash = ethers.keccak256(ethers.toUtf8Bytes("zk_proof"));
      
      await expect(
        voteRegistry.castVote(ELECTION_ID, VOTER_ID, voteHash, zkProofHash)
      ).to.emit(voteRegistry, "VoteCast");
      
      const hasVoted = await voteRegistry.checkVoterStatus(ELECTION_ID, VOTER_ID);
      expect(hasVoted).to.be.true;
    });
    
    it("should prevent double voting", async function () {
      const voteHash = ethers.keccak256(ethers.toUtf8Bytes("encrypted_vote"));
      const zkProofHash = ethers.keccak256(ethers.toUtf8Bytes("zk_proof"));
      
      await voteRegistry.castVote(ELECTION_ID, VOTER_ID, voteHash, zkProofHash);
      
      await expect(
        voteRegistry.castVote(ELECTION_ID, VOTER_ID, voteHash, zkProofHash)
      ).to.be.revertedWith("Already voted in this election");
    });
  });
  
  describe("Vote Count", function () {
    it("should track vote count correctly", async function () {
      const now = Math.floor(Date.now() / 1000);
      await voteRegistry.createElection(
        ELECTION_ID,
        "Washington Governor 2024",
        now - 1,
        now + 86400,
        JURISDICTION_ID
      );
      
      const voteHash = ethers.keccak256(ethers.toUtf8Bytes("vote"));
      const zkProofHash = ethers.keccak256(ethers.toUtf8Bytes("proof"));
      
      await voteRegistry.castVote(ELECTION_ID, VOTER_ID, voteHash, zkProofHash);
      
      const count = await voteRegistry.getVoteCount(ELECTION_ID);
      expect(count).to.equal(1);
    });
  });
});
