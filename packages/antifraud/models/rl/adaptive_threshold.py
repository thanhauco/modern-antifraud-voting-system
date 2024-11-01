"""
Reinforcement Learning Agent for Adaptive Fraud Detection Thresholds
Uses Deep Q-Network to dynamically optimize detection sensitivity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class RLConfig:
    state_dim: int = 8        # State features
    action_dim: int = 5       # Number of threshold levels
    hidden_dim: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99       # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update: int = 100


class DQNetwork(nn.Module):
    """
    Deep Q-Network for threshold optimization.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for stable training"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class ThresholdAgent:
    """
    DQN Agent for adaptive fraud detection thresholds.
    
    State: [alert_rate, false_positive_rate, queue_backlog, 
            time_of_day, vote_velocity, recent_fraud_rate,
            manual_review_rate, system_load]
            
    Actions: [very_low, low, medium, high, very_high] sensitivity
    
    Reward: +1 for true positives, -2 for false positives, 
            -5 for missed fraud, -0.1 for each manual review
    """
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Q-Networks
        self.q_network = DQNetwork(
            self.config.state_dim, 
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        self.target_network = DQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        self.memory = ReplayBuffer(self.config.memory_size)
        
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        
        # Threshold mappings
        self.threshold_levels = {
            0: 0.3,   # very_low: catch almost everything
            1: 0.5,   # low
            2: 0.7,   # medium (default)
            3: 0.85,  # high
            4: 0.95   # very_high: only flag obvious fraud
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def train_step(self) -> float:
        """Single training step on replay buffer"""
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        # Update target network
        self.steps += 1
        if self.steps % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

    def get_threshold(self, action: int) -> float:
        """Convert discrete action to threshold value"""
        return self.threshold_levels.get(action, 0.7)


class AdaptiveThresholdService:
    """
    Production service for adaptive fraud thresholds.
    """
    def __init__(self, model_path: str = None):
        self.agent = ThresholdAgent()
        self.current_threshold = 0.7
        self.state_history = deque(maxlen=100)
        
        if model_path:
            self.agent.q_network.load_state_dict(
                torch.load(model_path, map_location=self.agent.device)
            )

    def get_current_state(self, metrics: Dict) -> np.ndarray:
        """Extract state from current system metrics"""
        return np.array([
            metrics.get("alert_rate", 0.1),
            metrics.get("false_positive_rate", 0.05),
            metrics.get("queue_backlog", 0) / 1000,
            metrics.get("hour", 12) / 24,
            metrics.get("vote_velocity", 100) / 10000,
            metrics.get("recent_fraud_rate", 0.01),
            metrics.get("manual_review_rate", 0.1),
            metrics.get("system_load", 0.5)
        ], dtype=np.float32)

    def update_threshold(self, metrics: Dict) -> Dict:
        """Update threshold based on current conditions"""
        state = self.get_current_state(metrics)
        action = self.agent.select_action(state, training=False)
        new_threshold = self.agent.get_threshold(action)
        
        old_threshold = self.current_threshold
        self.current_threshold = new_threshold
        
        return {
            "previous_threshold": old_threshold,
            "new_threshold": new_threshold,
            "action_taken": action,
            "action_name": ["very_low", "low", "medium", "high", "very_high"][action],
            "epsilon": self.agent.epsilon
        }

    def record_feedback(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Record feedback for online learning"""
        self.agent.memory.push(state, action, reward, next_state, done)

    def train(self) -> float:
        """Perform one training step"""
        return self.agent.train_step()


if __name__ == "__main__":
    # Test agent
    agent = ThresholdAgent()
    
    # Simulated training loop
    state = np.random.rand(8).astype(np.float32)
    
    for _ in range(5):
        action = agent.select_action(state)
        next_state = np.random.rand(8).astype(np.float32)
        reward = np.random.randn()
        done = False
        
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
    
    print(f"Selected action: {action}, Threshold: {agent.get_threshold(action)}")
    
    # Test service
    service = AdaptiveThresholdService()
    result = service.update_threshold({
        "alert_rate": 0.15,
        "false_positive_rate": 0.08,
        "hour": 14
    })
    print(f"Threshold update: {result}")
