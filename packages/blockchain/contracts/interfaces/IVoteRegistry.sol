// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

/**
 * @title IVoteRegistry
 * @dev Interface for VoteRegistry contract
 */
interface IVoteRegistry {
    function createElection(
        bytes32 electionId,
        string calldata name,
        uint256 startTime,
        uint256 endTime,
        bytes32 jurisdictionId
    ) external;
    
    function castVote(
        bytes32 electionId,
        bytes32 voterId,
        bytes32 voteHash,
        bytes32 zkProofHash
    ) external;
    
    function verifyVote(
        bytes32 electionId,
        bytes32 voterId,
        bool verified
    ) external;
    
    function endElection(bytes32 electionId) external;
    
    function getVoteCount(bytes32 electionId) external view returns (uint256);
    
    function checkVoterStatus(bytes32 electionId, bytes32 voterId) external view returns (bool);
}
