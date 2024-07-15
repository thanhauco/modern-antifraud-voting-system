"""
Federated Learning Node Client
Handles local model training and secure aggregation for privacy-preserving fraud detection.
"""

import numpy as np
from typing import Dict, List, Any
import copy

class FederatedNode:
    def __init__(self, node_id: str, local_data: List[Any], model: Any):
        self.node_id = node_id
        self.local_data = local_data
        self.local_model = copy.deepcopy(model)
        self.privacy_budget = 1.0  # Epsilon for Differential Privacy
        
    def train_local(self, epochs: int = 5, learning_rate: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Train the model locally on jurisdiction-specific data.
        Returns model updates (gradients/weights) with added noise for privacy.
        """
        print(f"[{self.node_id}] Starting local training on {len(self.local_data)} records...")
        
        # Simulating training process
        # In reality, this would run PyTorch/TensorFlow training loop
        
        # Extract model weights
        weights = self._get_weights()
        
        # Add Differential Privacy Noise
        noisy_weights = self._apply_dp_noise(weights)
        
        return noisy_weights
    
    def update_global_model(self, global_weights: Dict[str, np.ndarray]):
        """Update local model with aggregated global weights"""
        self._set_weights(global_weights)
        print(f"[{self.node_id}] Synced with global model.")

    def _get_weights(self) -> Dict[str, np.ndarray]:
        # Mock weight extraction
        return {"layer1": np.random.rand(10, 10), "layer2": np.random.rand(5, 10)}
        
    def _set_weights(self, weights: Dict[str, np.ndarray]):
        # Mock weight update
        pass

    def _apply_dp_noise(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Laplacian noise for Differential Privacy"""
        noisy_weights = {}
        sensitivity = 0.1 # Sensitivity of the model update function
        scale = sensitivity / self.privacy_budget
        
        for k, v in weights.items():
            noise = np.random.laplace(0, scale, v.shape)
            noisy_weights[k] = v + noise
            
        return noisy_weights

class SecureAggregator:
    def __init__(self):
        self.updates = []
        
    def collect_update(self, node_id: str, weights: Dict[str, np.ndarray]):
        self.updates.append(weights)
        
    def aggregate(self) -> Dict[str, np.ndarray]:
        """Federated Averaging (FedAvg)"""
        if not self.updates:
            return {}
            
        keys = self.updates[0].keys()
        avg_weights = {}
        n = len(self.updates)
        
        for k in keys:
            stacked = np.stack([u[k] for u in self.updates])
            avg_weights[k] = np.mean(stacked, axis=0)
            
        self.updates = [] # Reset for next round
        return avg_weights

if __name__ == "__main__":
    # Simulate Federated Learning Round
    server = SecureAggregator()
    node1 = FederatedNode("WA_Seattle", [], None)
    node2 = FederatedNode("WA_Spokane", [], None)
    
    w1 = node1.train_local()
    w2 = node2.train_local()
    
    server.collect_update(node1.node_id, w1)
    server.collect_update(node2.node_id, w2)
    
    new_global_model = server.aggregate()
    print("Global model aggregated securely.")
