"""
This module defines a simple grid‑world environment for autonomous navigation
tasks.  The environment is inspired by the 30×30 maze described in the
provided paper.  It supports continuous actions (linear and angular
velocities) and produces observations that summarise local obstacle
information in eight directions.  The environment can optionally
incorporate task‑specific uncertainty maps (TSUMs) and a Bayesian neural
network (BNN) to penalise risky actions.  When a BNN and TSUM are
provided, the reward returned by ``step`` includes an additional penalty
for actions whose predicted uncertainty exceeds the local TSUM threshold.

The environment is implemented as an OpenAI Gym environment.  It uses
NumPy rather than image rendering to keep dependencies light.  Agents
control a point robot with a heading; their actions specify desired
forward velocity and change in heading.  Motion noise and obstacle
collisions are modelled to introduce uncertainty.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """A continuous‑action grid world with optional risk penalties.

    The environment consists of a square grid of size ``grid_size`` with
    randomly placed obstacles.  The agent is modelled as a point robot
    with position and heading.  At each time step the agent chooses a
    linear velocity ``v`` in ``[0, v_max]`` and an angular velocity
    ``omega`` in ``[−omega_max, omega_max]``.  The robot moves according
    to simple kinematic equations with optional Gaussian noise.  If the
    robot collides with an obstacle, it incurs a penalty and the episode
    terminates.

    Observations are 64‑dimensional vectors capturing obstacle distances
    along eight equally spaced directions.  Specifically, for each of
    eight directions the agent measures the distance to the nearest
    obstacle (or boundary) at eight discrete range bins.  These values
    are normalised to ``[0, 1]``.

    If a TSUM map and a BNN are provided, the environment penalises
    actions whose predicted state transition variance exceeds the TSUM
    threshold at the agent's current location.  The penalty is scaled
    by ``lambda_penalty`` and subtracted from the reward.
    """
    metadata = {"render.modes": []}
    def __init__(self,
                 grid_size: int = 30,
                 obstacle_density: float = 0.25,
                 max_steps: int = 500,
                 v_max: float = 1.0,
                 omega_max: float = 1.0,
                 seed: Optional[int] = None,
                 tsum_map: Optional[np.ndarray] = None,
                 bnn_model: Optional[object] = None,
                 lambda_penalty: float = 1.0,
                 tau_min: float = 0.1) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.v_max = v_max
        self.omega_max = omega_max
        self.random_state = np.random.RandomState(seed)
        # TSUM and BNN for risk penalty
        self.tsum_map = tsum_map
        self.bnn_model = bnn_model
        self.lambda_penalty = lambda_penalty
        self.tau_min = tau_min
        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([0.0, -self.omega_max], dtype=np.float32),
                                       high=np.array([self.v_max, self.omega_max], dtype=np.float32),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64,), dtype=np.float32)
        # Internal state
        self.position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.reset()

    def _generate_obstacles(self) -> np.ndarray:
        obstacles = self.random_state.rand(self.grid_size, self.grid_size) < self.obstacle_density
        # Clear borders
        obstacles[0, :] = False
        obstacles[:, 0] = False
        obstacles[-1, :] = False
        obstacles[:, -1] = False
        return obstacles

    def _sample_free_cell(self) -> np.ndarray:
        while True:
            x = self.random_state.randint(0, self.grid_size)
            y = self.random_state.randint(0, self.grid_size)
            if not self.obstacles[x, y]:
                return np.array([x + 0.5, y + 0.5], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.obstacles = self._generate_obstacles()
        self.position = self._sample_free_cell()
        self.goal = self._sample_free_cell()
        # ensure separation
        while np.linalg.norm(self.goal - self.position) < self.grid_size / 4:
            self.goal = self._sample_free_cell()
        self.heading = float(self.random_state.rand() * 2 * np.pi)
        self.step_count = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(64, dtype=np.float32)
        directions = [i * np.pi / 4 for i in range(8)]
        max_range = 8.0
        for d_idx, direction in enumerate(directions):
            for r_bin in range(8):
                distance = (r_bin + 1) / 8.0 * max_range
                dx = distance * math.cos(self.heading + direction)
                dy = distance * math.sin(self.heading + direction)
                num_steps = 5
                collided = False
                for t in range(1, num_steps + 1):
                    px = self.position[0] + dx * t / num_steps
                    py = self.position[1] + dy * t / num_steps
                    gx = int(px)
                    gy = int(py)
                    if (gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size
                        or self.obstacles[gx, gy]):
                        collided = True
                        break
                obs[d_idx * 8 + r_bin] = 0.0 if collided else 1.0
        return obs

    def _goal_reached(self) -> bool:
        return np.linalg.norm(self.position - self.goal) < 1.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        self.step_count += 1
        v = float(np.clip(action[0], 0.0, self.v_max))
        omega = float(np.clip(action[1], -self.omega_max, self.omega_max))
        old_heading = self.heading
        self.heading = (self.heading + omega) % (2 * np.pi)
        noise_pos = self.random_state.normal(0, 0.1, size=2)
        dx = (v * math.cos(old_heading)) + noise_pos[0]
        dy = (v * math.sin(old_heading)) + noise_pos[1]
        new_position = self.position + np.array([dx, dy], dtype=np.float32)
        reward = -0.01
        done = False
        info: Dict[str, object] = {}
        gx = int(new_position[0])
        gy = int(new_position[1])
        if (gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size
            or self.obstacles[gx, gy]):
            reward -= 1.0
            done = True
            info["collision"] = True
        else:
            self.position = new_position
            info["collision"] = False
        if not done and self._goal_reached():
            reward += 1.0
            done = True
            info["goal"] = True
        else:
            info["goal"] = False
        # risk penalty
        if self.bnn_model is not None and self.tsum_map is not None and not done:
            state_vec = np.array([self.position[0], self.position[1], self.heading, v, omega], dtype=np.float32)
            var = float(self.bnn_model.predict_variance(state_vec))
            gx = int(self.position[0])
            gy = int(self.position[1])
            tau = float(self.tsum_map[gx, gy])
            if var > tau:
                reward -= self.lambda_penalty * (var - tau)
                info["risk_violation"] = True
            else:
                info["risk_violation"] = False
        else:
            info["risk_violation"] = False
        if self.step_count >= self.max_steps:
            done = True
        obs = self._get_observation()
        return obs, reward, done, info

    def render(self, mode: str = 'human') -> None:
        # rendering is not implemented in this minimal environment
        return


def make_env(seed: Optional[int] = None,
             tsum_map: Optional[np.ndarray] = None,
             bnn_model: Optional[object] = None,
             lambda_penalty: float = 1.0,
             obstacle_density: float = 0.25) -> GridWorldEnv:
    return GridWorldEnv(obstacle_density=obstacle_density,
                        seed=seed,
                        tsum_map=tsum_map,
                        bnn_model=bnn_model,
                        lambda_penalty=lambda_penalty)

