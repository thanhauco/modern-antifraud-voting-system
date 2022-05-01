// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title ElectionManager
 * @dev Manages election lifecycle, candidates, and results aggregation
 * @author Thanh Vu <thanhauco@gmail.com>
 */
contract ElectionManager is AccessControl {
    bytes32 public constant ELECTION_ADMIN_ROLE = keccak256("ELECTION_ADMIN_ROLE");
    
    struct Candidate {
        bytes32 candidateId;
        string name;
        string party;
        string position;
        bool isActive;
    }
    
    struct ElectionConfig {
        bytes32 electionId;
        string electionType; // "governor", "senate", "house", etc.
        bytes32 jurisdictionId;
        bytes32[] candidateIds;
        uint256 registrationDeadline;
        bool allowWriteIn;
    }
    
    // Mappings
    mapping(bytes32 => ElectionConfig) public electionConfigs;
    mapping(bytes32 => Candidate) public candidates;
    mapping(bytes32 => mapping(bytes32 => bool)) public electionCandidates; // electionId => candidateId => registered
    
    // Events
    event CandidateRegistered(bytes32 indexed candidateId, string name, string party);
    event CandidateAddedToElection(bytes32 indexed electionId, bytes32 indexed candidateId);
    event ElectionConfigured(bytes32 indexed electionId, string electionType);
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ELECTION_ADMIN_ROLE, msg.sender);
    }
    
    /**
     * @dev Registers a new candidate in the system
     */
    function registerCandidate(
        bytes32 candidateId,
        string calldata name,
        string calldata party,
        string calldata position
    ) external onlyRole(ELECTION_ADMIN_ROLE) {
        require(candidates[candidateId].candidateId == bytes32(0), "Candidate exists");
        
        candidates[candidateId] = Candidate({
            candidateId: candidateId,
            name: name,
            party: party,
            position: position,
            isActive: true
        });
        
        emit CandidateRegistered(candidateId, name, party);
    }
    
    /**
     * @dev Configures an election with candidates
     */
    function configureElection(
        bytes32 electionId,
        string calldata electionType,
        bytes32 jurisdictionId,
        uint256 registrationDeadline,
        bool allowWriteIn
    ) external onlyRole(ELECTION_ADMIN_ROLE) {
        require(electionConfigs[electionId].electionId == bytes32(0), "Election configured");
        
        electionConfigs[electionId] = ElectionConfig({
            electionId: electionId,
            electionType: electionType,
            jurisdictionId: jurisdictionId,
            candidateIds: new bytes32[](0),
            registrationDeadline: registrationDeadline,
            allowWriteIn: allowWriteIn
        });
        
        emit ElectionConfigured(electionId, electionType);
    }
    
    /**
     * @dev Adds a candidate to an election
     */
    function addCandidateToElection(
        bytes32 electionId,
        bytes32 candidateId
    ) external onlyRole(ELECTION_ADMIN_ROLE) {
        require(electionConfigs[electionId].electionId != bytes32(0), "Election not found");
        require(candidates[candidateId].isActive, "Candidate not active");
        require(!electionCandidates[electionId][candidateId], "Already added");
        require(block.timestamp < electionConfigs[electionId].registrationDeadline, "Registration closed");
        
        electionConfigs[electionId].candidateIds.push(candidateId);
        electionCandidates[electionId][candidateId] = true;
        
        emit CandidateAddedToElection(electionId, candidateId);
    }
    
    /**
     * @dev Gets all candidates for an election
     */
    function getElectionCandidates(bytes32 electionId) external view returns (bytes32[] memory) {
        return electionConfigs[electionId].candidateIds;
    }
    
    /**
     * @dev Gets candidate details
     */
    function getCandidateInfo(bytes32 candidateId) external view returns (
        string memory name,
        string memory party,
        string memory position,
        bool isActive
    ) {
        Candidate storage c = candidates[candidateId];
        return (c.name, c.party, c.position, c.isActive);
    }
}
