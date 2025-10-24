"""
evaluate.py
===========

This script evaluates a trained reinforcement learning agent on the grid‑world
navigation task and reports quantitative performance metrics such as
success rate, average path length, cumulative reward, collision count
and uncertainty violation ratio.  It supports baseline agents trained
with DQN, PPO or SAC as well as the TSUM+BNN Bayesian RL agent.

Examples::

    python evaluate.py --model_path results/tsum_bnn_20250101_123456.zip --algo tsum_bnn
    python evaluate.py --model_path results/ppo_20250101_101010.zip --algo ppo

"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import gym
from stable_baselines3 import PPO, DQN, SAC

from environment import make_env, GridWorldEnv
from tsum import generate_tsum
from bnn_model import DropoutBNN


def load_bnn_model(path: str, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 3, dropout_rate: float = 0.1) -> DropoutBNN:
    """Load a DropoutBNN from a saved state dict."""
    model = DropoutBNN(input_dim=input_dim + 0, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


ALGORITHMS = {
    'dqn': DQN,
    'ppo': PPO,
    'sac': SAC,
}


def evaluate_agent(algo_name: str, model_path: str, bnn_path: str | None, episodes: int = 100, seed: int | None = None, obstacle_density: float = 0.25) -> None:
    """Evaluate a trained agent and print performance metrics."""
    # Create environment.  For TSUM+BNN we need to load BNN and set penalty and wrapper.
    if algo_name == 'tsum_bnn':
        assert bnn_path is not None, "BNN path must be provided for TSUM+BNN evaluation"
        # Load BNN
        bnn_model = load_bnn_model(bnn_path)
        # Create base env with BNN and penalty (lambda=1.0) – will update TSUM at reset
        base_env = make_env(seed=seed, tsum_map=None, bnn_model=bnn_model, lambda_penalty=1.0, obstacle_density=obstacle_density)
        # Use wrapper that recomputes TSUM per episode
        class EvalTSUMWrapper(gym.Wrapper):
            def __init__(self, env: GridWorldEnv):
                super().__init__(env)
            def reset(self, **kwargs):
                obs = self.env.reset(**kwargs)
                goal_cell = (int(self.env.goal[0]), int(self.env.goal[1]))
                tsum_map = generate_tsum(self.env.obstacles, goal_cell)
                self.env.tsum_map = tsum_map
                return obs
        env = EvalTSUMWrapper(base_env)
        # Choose algorithm class (SAC)
        algo_class = SAC
    else:
        env = make_env(seed=seed, obstacle_density=obstacle_density)
        algo_class = ALGORITHMS[algo_name]
    # Load RL model
    model = algo_class.load(model_path)
    # Run evaluation episodes
    success_count = 0
    path_lengths = []
    total_reward = []
    collision_count = 0
    risk_violation_steps = 0
    total_steps = 0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        while not done and steps < 1000:  # safety cap
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            total_steps += 1
            if info.get('collision', False):
                collision_count += 1
            if info.get('risk_violation', False):
                risk_violation_steps += 1
            if info.get('goal', False):
                success_count += 1
        path_lengths.append(steps)
        total_reward.append(ep_reward)
    # Compute metrics
    episodes_success = success_count / episodes
    avg_path = float(np.mean([l for l in path_lengths if l is not None]))
    avg_reward = float(np.mean(total_reward))
    violation_ratio = risk_violation_steps / total_steps if total_steps > 0 else 0.0
    print(f"Evaluation over {episodes} episodes:")
    print(f"  Success rate: {episodes_success * 100:.1f}%")
    print(f"  Average path length: {avg_path:.1f} steps")
    print(f"  Average cumulative reward: {avg_reward:.3f}")
    print(f"  Collision count: {collision_count}")
    if algo_name == 'tsum_bnn':
        print(f"  Risk violation ratio: {violation_ratio * 100:.2f}%")



def main():
    parser = argparse.ArgumentParser(description="Evaluate trained navigation agents.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved RL model (.zip)')
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo', 'sac', 'tsum_bnn'], required=True, help='Agent type')
    parser.add_argument('--bnn_path', type=str, default=None, help='Path to saved BNN state dict (required for tsum_bnn)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for environment')
    parser.add_argument('--obstacle_density', type=float, default=0.25, help='Obstacle density during evaluation')
    args = parser.parse_args()
    evaluate_agent(args.algo, args.model_path, args.bnn_path, episodes=args.episodes, seed=args.seed, obstacle_density=args.obstacle_density)


if __name__ == '__main__':
    main()

