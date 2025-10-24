"""
rl_training.py
===============

This script provides a reproducible entry point for training various
reinforcement learning agents on the grid‑world navigation problem
presented in the paper.  It supports standard baseline algorithms from
Stable Baselines 3 (DQN, PPO and SAC) as well as the proposed
TSUM+BNN Bayesian reinforcement learning approach.  The
TSUM+BNN variant first trains a Bayesian neural network (BNN) to
approximate the environment dynamics from random transition data and
then augments the reward with a variance‑based penalty derived from
a task‑specific uncertainty map (TSUM).  During training the TSUM is
recomputed at the beginning of each episode using the current
obstacle map and goal location.

Usage examples::

    # Train the TSUM+BNN agent for 50k steps
    python rl_training.py --algo tsum_bnn --timesteps 50000

    # Train a PPO baseline agent
    python rl_training.py --algo ppo --timesteps 50000

The resulting agent will be saved into the ``results/`` directory with
an informative filename.  You can then evaluate the agent using
``evaluate.py``.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch

import gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from environment import make_env, GridWorldEnv
from tsum import generate_tsum
from bnn_model import DropoutBNN, BNNTrainer, generate_bnn_dataset


class TSUMWrapper(gym.Wrapper):
    """Gym wrapper to update the TSUM map at each episode reset."""

    def __init__(self, env: GridWorldEnv, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3, tau_min: float = 0.1):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau_min = tau_min

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Compute TSUM for the current environment based on obstacles and goal
        goal_cell = (int(self.env.goal[0]), int(self.env.goal[1]))
        tsum_map = generate_tsum(self.env.obstacles, goal_cell, self.alpha, self.beta, self.gamma, self.tau_min)
        self.env.tsum_map = tsum_map
        return obs


ALGORITHMS = {
    'dqn': DQN,
    'ppo': PPO,
    'sac': SAC,
}


def train_tsumbnn_agent(args):
    """Train the TSUM+BNN agent."""
    # Create a temporary environment for dataset collection
    env_factory = lambda seed=None: make_env(seed=seed)
    print("Collecting transition data for BNN training…")
    states, actions, next_states = generate_bnn_dataset(
        env_factory=env_factory,
        num_samples=args.bnn_samples,
        max_episode_steps=args.bnn_max_episode_steps,
        seed=args.seed or 0,
    )
    print(f"Collected {len(states)} transitions.")
    # Train BNN
    input_dim = states.shape[1] + actions.shape[1]
    model = DropoutBNN(input_dim=input_dim, hidden_dim=args.bnn_hidden_dim, output_dim=3, dropout_rate=args.dropout_rate)
    trainer = BNNTrainer(model, lr=args.bnn_lr, batch_size=args.bnn_batch_size, device='cpu')
    print("Training BNN…")
    trainer.train(states, actions, next_states, epochs=args.bnn_epochs)
    # Create environment with BNN and TSUM wrapper
    base_env = make_env(seed=args.seed, tsum_map=None, bnn_model=model, lambda_penalty=args.lambda_penalty, obstacle_density=args.obstacle_density)
    env = TSUMWrapper(base_env, alpha=args.tsum_alpha, beta=args.tsum_beta, gamma=args.tsum_gamma, tau_min=args.tsum_tau_min)
    # Choose algorithm (we use SAC as base for continuous action space)
    algo_class = SAC
    # Determine log directory and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"tsum_bnn_{timestamp}"
    save_path = os.path.join('results', model_name)
    os.makedirs('results', exist_ok=True)
    # Save BNN model separately for later evaluation
    bnn_path = save_path + "_bnn.pt"
    torch.save(model.state_dict(), bnn_path)
    print(f"BNN model saved to {bnn_path}.")
    # Train RL agent
    print("Training TSUM+BNN policy using SAC…")
    model_rl = algo_class('MlpPolicy', env, verbose=1, seed=args.seed)
    model_rl.learn(total_timesteps=args.timesteps)
    model_rl.save(save_path)
    print(f"TSUM+BNN agent saved to {save_path}.")


def train_baseline_agent(algo_name: str, args):
    """Train a baseline RL agent (DQN, PPO or SAC)."""
    algo_class = ALGORITHMS[algo_name]
    # Create environment without BNN or TSUM penalty
    env = make_env(seed=args.seed, obstacle_density=args.obstacle_density)
    # Determine model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{algo_name}_{timestamp}"
    save_path = os.path.join('results', model_name)
    os.makedirs('results', exist_ok=True)
    # Instantiate and train agent
    print(f"Training {algo_name.upper()} agent…")
    # Choose appropriate policy
    if algo_name == 'dqn':
        policy = 'MlpPolicy'
    else:
        policy = 'MlpPolicy'
    model = algo_class(policy, env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save(save_path)
    print(f"{algo_name.upper()} agent saved to {save_path}.")


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for autonomous navigation with TSUM and BNN.")
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo', 'sac', 'tsum_bnn'], default='tsum_bnn', help='Algorithm to train')
    parser.add_argument('--timesteps', type=int, default=20000, help='Number of training timesteps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--obstacle_density', type=float, default=0.25, help='Obstacle density in the environment')
    # BNN hyperparameters
    parser.add_argument('--bnn_samples', type=int, default=5000, help='Number of transitions to collect for BNN')
    parser.add_argument('--bnn_max_episode_steps', type=int, default=50, help='Max steps per episode during data collection')
    parser.add_argument('--bnn_hidden_dim', type=int, default=64, help='Hidden dimension of the BNN')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for BNN')
    parser.add_argument('--bnn_lr', type=float, default=1e-3, help='Learning rate for BNN training')
    parser.add_argument('--bnn_batch_size', type=int, default=256, help='Batch size for BNN training')
    parser.add_argument('--bnn_epochs', type=int, default=10, help='Training epochs for BNN')
    # TSUM hyperparameters
    parser.add_argument('--tsum_alpha', type=float, default=0.4, help='Weight for task relevance')
    parser.add_argument('--tsum_beta', type=float, default=0.3, help='Weight for environmental complexity')
    parser.add_argument('--tsum_gamma', type=float, default=0.3, help='Weight for safety')
    parser.add_argument('--tsum_tau_min', type=float, default=0.1, help='Minimum permissible variance')
    parser.add_argument('--lambda_penalty', type=float, default=1.0, help='Penalty coefficient for risk violations')

    args = parser.parse_args()

    if args.algo == 'tsum_bnn':
        train_tsumbnn_agent(args)
    else:
        train_baseline_agent(args.algo, args)


if __name__ == '__main__':
    main()

