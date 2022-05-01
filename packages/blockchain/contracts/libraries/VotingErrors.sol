// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

/**
 * @title VotingErrors
 * @dev Custom errors for gas-efficient reverts
 */
library VotingErrors {
    error ElectionNotFound(bytes32 electionId);
    error ElectionAlreadyExists(bytes32 electionId);
    error ElectionNotActive(bytes32 electionId);
    error VotingNotStarted(bytes32 electionId, uint256 startTime);
    error VotingEnded(bytes32 electionId, uint256 endTime);
    error AlreadyVoted(bytes32 electionId, bytes32 voterId);
    error InvalidVoteHash();
    error InvalidZKProofHash();
    error InvalidTimeRange(uint256 startTime, uint256 endTime);
    error Unauthorized(address caller);
    error CandidateNotFound(bytes32 candidateId);
    error CandidateNotActive(bytes32 candidateId);
    error RegistrationClosed(bytes32 electionId);
}
