"""
BNN (Bayesian Neural Network) module
===================================

This module provides a simple Bayesian neural network implementation based
on Monte Carlo dropout.  The network is used to estimate the dynamics of
our grid‑world environment, predicting the change in state given the
current state and action.  By sampling multiple forward passes with
stochastic dropout enabled at inference time, we approximate the model's
predictive uncertainty.  The TSUM reward penalty uses this predicted
variance to discourage actions in regions where the model is uncertain.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DropoutBNN(nn.Module):
    """Bayesian neural network using dropout to estimate predictive variance.

    The network takes a state vector consisting of continuous features
    (position x, position y, heading, linear velocity v and angular velocity
    omega) and outputs a prediction of the change in the agent's pose
    (delta x, delta y, delta heading).  Dropout layers remain active at
    inference time to simulate sampling from the approximate posterior.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 3, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def predict_variance(self, state: np.ndarray, num_samples: int = 10) -> float:
        """Estimate predictive variance of the network's output for a given state.

        Parameters
        ----------
        state : np.ndarray
            Input state vector of shape (5,).
        num_samples : int
            Number of Monte Carlo samples to draw.

        Returns
        -------
        float
            The trace of the empirical covariance of the sampled outputs.
        """
        self.eval()
        with torch.no_grad():
            # Prepare input tensor and replicate for sampling
            inp = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            outputs = []
            for _ in range(num_samples):
                # Use dropout in inference mode by calling forward() directly
                out = self(inp)
                outputs.append(out.squeeze(0).cpu().numpy())
            arr = np.stack(outputs, axis=0)
            # Compute covariance matrix and return its trace
            cov = np.cov(arr, rowvar=False)
            # cov might be 0‐dim for single output; handle scalar case
            if cov.ndim == 0:
                return float(cov)
            trace = np.trace(cov)
            return float(trace)


class BNNTrainer:
    """Utility to train a DropoutBNN on environment transition data."""

    def __init__(self, model: DropoutBNN, lr: float = 1e-3, batch_size: int = 128, device: str = 'cpu'):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, epochs: int = 10) -> None:
        """Train the BNN on collected transition samples.

        The network predicts the change in state (next_state - state) given
        the concatenated [state, action] vector.
        """
        x = np.concatenate([states, actions], axis=1).astype(np.float32)
        y = (next_states - states[:, :3]).astype(np.float32)  # delta pose: dx, dy, d_heading
        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        mse = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                preds = self.model(batch_x)
                loss = mse(preds, batch_y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


def generate_bnn_dataset(
    env_factory: callable,
    num_samples: int = 5000,
    max_episode_steps: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect a dataset of transitions for training the BNN.

    A random policy is used to explore the environment.  Each transition
    consists of (state, action, next_state) where the state vector
    includes x, y, heading, linear and angular velocities.
    """
    rng = np.random.RandomState(seed)
    states = []
    actions = []
    next_states = []
    while len(states) < num_samples:
        env = env_factory(seed=int(rng.randint(0, 1e6)))
        obs = env.reset()
        state = np.array([env.position[0], env.position[1], env.heading, 0.0, 0.0], dtype=np.float32)
        # Use random policy for a short episode
        for _ in range(max_episode_steps):
            # Random continuous actions within bounds
            action = np.array([
                rng.uniform(0.0, env.v_max),
                rng.uniform(-env.omega_max, env.omega_max)
            ], dtype=np.float32)
            # Save current state and action
            states.append(state.copy())
            actions.append(action.copy())
            # Step environment
            _, _, done, _ = env.step(action)
            next_state = np.array([env.position[0], env.position[1], env.heading], dtype=np.float32)
            next_states.append(next_state.copy())
            # Update state vector with action taken
            state = np.array([env.position[0], env.position[1], env.heading, action[0], action[1]], dtype=np.float32)
            if done or len(states) >= num_samples:
                break
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
    )

