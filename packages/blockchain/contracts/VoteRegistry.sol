// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title VoteRegistry
 * @dev Core voting smart contract for the Modern Anti-Fraud Voting System
 * @notice Manages vote casting, storage, and verification on the blockchain
 * @author Thanh Vu <thanhauco@gmail.com>
 */
contract VoteRegistry is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant ELECTION_ADMIN_ROLE = keccak256("ELECTION_ADMIN_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");
    
    struct Vote {
        bytes32 voteHash;          // Hash of encrypted vote
        bytes32 zkProofHash;       // Zero-knowledge proof hash
        uint256 timestamp;         // Block timestamp
        bytes32 jurisdictionId;    // Jurisdiction identifier
        bool isVerified;           // ZK proof verified
    }
    
    struct Election {
        bytes32 electionId;
        string name;
        uint256 startTime;
        uint256 endTime;
        bytes32 jurisdictionId;
        bool isActive;
        uint256 totalVotes;
    }
    
    // Mappings
    mapping(bytes32 => Election) public elections;
    mapping(bytes32 => mapping(bytes32 => Vote)) public votes; // electionId => voterId => Vote
    mapping(bytes32 => mapping(bytes32 => bool)) public hasVoted; // electionId => voterId => bool
    mapping(bytes32 => bytes32[]) public electionVoterIds; // electionId => voterIds[]
    
    // Events
    event ElectionCreated(bytes32 indexed electionId, string name, uint256 startTime, uint256 endTime);
    event VoteCast(bytes32 indexed electionId, bytes32 indexed voterId, bytes32 voteHash, uint256 timestamp);
    event VoteVerified(bytes32 indexed electionId, bytes32 indexed voterId, bool success);
    event ElectionEnded(bytes32 indexed electionId, uint256 totalVotes);
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ELECTION_ADMIN_ROLE, msg.sender);
    }
    
    /**
     * @dev Creates a new election
     * @param electionId Unique identifier for the election
     * @param name Human-readable name for the election
     * @param startTime Unix timestamp when voting begins
     * @param endTime Unix timestamp when voting ends
     * @param jurisdictionId Jurisdiction identifier (e.g., state code)
     */
    function createElection(
        bytes32 electionId,
        string calldata name,
        uint256 startTime,
        uint256 endTime,
        bytes32 jurisdictionId
    ) external onlyRole(ELECTION_ADMIN_ROLE) {
        require(elections[electionId].electionId == bytes32(0), "Election already exists");
        require(startTime < endTime, "Invalid time range");
        require(startTime > block.timestamp, "Start time must be in future");
        
        elections[electionId] = Election({
            electionId: electionId,
            name: name,
            startTime: startTime,
            endTime: endTime,
            jurisdictionId: jurisdictionId,
            isActive: true,
            totalVotes: 0
        });
        
        emit ElectionCreated(electionId, name, startTime, endTime);
    }
    
    /**
     * @dev Casts a vote for an election
     * @param electionId Election identifier
     * @param voterId Anonymized voter identifier (derived from ZKP)
     * @param voteHash Hash of the encrypted vote
     * @param zkProofHash Hash of the zero-knowledge proof
     */
    function castVote(
        bytes32 electionId,
        bytes32 voterId,
        bytes32 voteHash,
        bytes32 zkProofHash
    ) external nonReentrant whenNotPaused {
        Election storage election = elections[electionId];
        
        require(election.isActive, "Election is not active");
        require(block.timestamp >= election.startTime, "Voting has not started");
        require(block.timestamp <= election.endTime, "Voting has ended");
        require(!hasVoted[electionId][voterId], "Already voted in this election");
        require(voteHash != bytes32(0), "Invalid vote hash");
        require(zkProofHash != bytes32(0), "Invalid ZK proof hash");
        
        votes[electionId][voterId] = Vote({
            voteHash: voteHash,
            zkProofHash: zkProofHash,
            timestamp: block.timestamp,
            jurisdictionId: election.jurisdictionId,
            isVerified: false
        });
        
        hasVoted[electionId][voterId] = true;
        electionVoterIds[electionId].push(voterId);
        election.totalVotes++;
        
        emit VoteCast(electionId, voterId, voteHash, block.timestamp);
    }
    
    /**
     * @dev Marks a vote as verified after ZK proof validation
     * @param electionId Election identifier
     * @param voterId Voter identifier
     * @param verified Whether the ZK proof was validated successfully
     */
    function verifyVote(
        bytes32 electionId,
        bytes32 voterId,
        bool verified
    ) external onlyRole(AUDITOR_ROLE) {
        require(hasVoted[electionId][voterId], "Vote does not exist");
        
        votes[electionId][voterId].isVerified = verified;
        
        emit VoteVerified(electionId, voterId, verified);
    }
    
    /**
     * @dev Ends an election (prevents further voting)
     * @param electionId Election identifier
     */
    function endElection(bytes32 electionId) external onlyRole(ELECTION_ADMIN_ROLE) {
        Election storage election = elections[electionId];
        require(election.isActive, "Election already ended");
        
        election.isActive = false;
        
        emit ElectionEnded(electionId, election.totalVotes);
    }
    
    /**
     * @dev Gets the total number of votes in an election
     * @param electionId Election identifier
     * @return Total number of votes cast
     */
    function getVoteCount(bytes32 electionId) external view returns (uint256) {
        return elections[electionId].totalVotes;
    }
    
    /**
     * @dev Checks if a voter has voted in an election
     * @param electionId Election identifier
     * @param voterId Voter identifier
     * @return Whether the voter has voted
     */
    function checkVoterStatus(bytes32 electionId, bytes32 voterId) external view returns (bool) {
        return hasVoted[electionId][voterId];
    }
    
    /**
     * @dev Pauses all voting (emergency only)
     */
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpauses voting
     */
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
}
