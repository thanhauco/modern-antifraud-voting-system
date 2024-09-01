"""
Distributed Fraud Inference Service using Ray Serve
Enables ultra-low latency, scalable API for real-time vote scoring.
"""

import ray
from ray import serve
from starlette.requests import Request
import torch
import json
import numpy as np

# Bind to the Ray cluster
ray.init(address="auto", ignore_reinit_error=True)
serve.start(detached=True)

@serve.deployment(num_replicas=3, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5})
class FraudScoringDeployment:
    def __init__(self):
        print("Initializing Fraud Scoring Model...")
        # Laxy load models here usually
        # self.model = torch.load("s3://models/fraud_ensemble_v1.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def __call__(self, http_request: Request):
        data = await http_request.json()
        
        # Extract features
        features = self._preprocess(data)
        score = self._predict(features)
        
        return {"fraud_score": score, "status": "processed"}
    
    def _preprocess(self, data):
        # transform raw JSON to tensor
        return torch.tensor([0.1, 0.2, 0.9], device=self.device)
        
    def _predict(self, features):
        # Mock prediction
        return 0.15

@serve.deployment(route_prefix="/api/v2/anomaly")
class AnomalyDetectionDeployment:
    def __init__(self):
        # Load Isolation Forest or similar lightweight model
        pass
        
    async def __call__(self, http_request: Request):
        data = await http_request.json()
        return {"anomaly_detected": False, "confidence": 0.99}

# Define the deployment graph
fraud_node = FraudScoringDeployment.bind()
anomaly_node = AnomalyDetectionDeployment.bind()

if __name__ == "__main__":
    # Local test execution
    handle = serve.run(fraud_node)
    print("Deployment active. Sending test request...")
    import requests
    resp = requests.post("http://localhost:8000/", json={"event": "vote_cast", "voter_id": "123"})
    print(resp.json())
