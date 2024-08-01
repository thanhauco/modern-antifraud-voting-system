"""
Graph Neural Network for Fraud Ring Detection
Uses Graph Attention Networks (GAT) to identify coordinated voting clusters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VoterNode:
    id: str
    features: List[float]  # [voting_history, device_risk, ip_risk, ...]
    
@dataclass
class RelationshipEdge:
    source: str
    target: str
    type: str  # 'shared_ip', 'shared_device', 'social_connection'
    weight: float

class FraudRingGAT(nn.Module):
    """
    Graph Attention Network for detecting fraud rings.
    Learns embeddings for voters and classifies them as 'legitimate', 'suspicious', or 'sybil'.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        
        # First Graph Attention Layer
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        
        # Second Graph Attention Layer
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, x, edge_index):
        # x: Node feature matrix [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]
        
        # Layer 1
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Layer 2
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class RingDetector:
    def __init__(self):
        self.model = FraudRingGAT(in_channels=16, hidden_channels=32, out_channels=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def build_graph(self, nodes: List[VoterNode], edges: List[RelationshipEdge]):
        """Convert raw data to PyTorch Geometric Data format"""
        # Mapping IDs to indices
        node_map = {node.id: i for i, node in enumerate(nodes)}
        
        # Node Features
        x = torch.tensor([n.features for n in nodes], dtype=torch.float).to(self.device)
        
        # Edge Index
        edge_indices = []
        for edge in edges:
            if edge.source in node_map and edge.target in node_map:
                src = node_map[edge.source]
                dst = node_map[edge.target]
                edge_indices.append([src, dst])
                # Undirected graph implies bidirectional edges usually
                edge_indices.append([dst, src])
                
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        
        return x, edge_index

    def detect_communities(self, x, edge_index):
        """Identify dense subgraphs indicating potential rings"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, edge_index)
            probs = torch.exp(out)
            
            # Simple threshold for suspicious classification
            suspicious_mask = probs[:, 1] > 0.8  # Class 1 is 'fraud'
            
        return suspicious_mask

# Example Usage Stub
if __name__ == "__main__":
    detector = RingDetector()
    print("Ring Detector initialized with GATv2 architecture")
