"""
TSUM generation module
======================

This module implements generation of Task‑Specific Uncertainty Maps (TSUMs).
A TSUM assigns an acceptable prediction variance threshold to each cell in the
navigation grid.  Cells closer to the goal or in more complex regions are
assigned lower thresholds to encourage caution, while open areas allow higher
uncertainty.  The formulation used here follows the intuition described in
Shuang Yu et al.'s paper:

``tau(x) = max(alpha * kappa(x) + beta * rho(x) + gamma * phi(x), tau_min)``

where:

* ``kappa(x)`` reflects task relevance—proximity to the goal.
* ``rho(x)`` measures environmental complexity via local obstacle density.
* ``phi(x)`` encodes safety via distance to nearest obstacle.
* ``alpha``, ``beta``, and ``gamma`` are weighting coefficients that sum to 1.
* ``tau_min`` is the minimum variance the agent must tolerate.

Although the underlying paper considers natural language descriptions and
semantics, this implementation focuses on spatial factors that can be
computed directly from a grid and obstacle map.  It serves as a reference
implementation for reproducible experiments.
"""

from __future__ import annotations

import numpy as np


def compute_distance_to_goal(grid_shape: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    """Compute Euclidean distance from each grid cell to the goal cell."""
    gx, gy = goal
    xs, ys = np.indices(grid_shape)
    return np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2)


def compute_local_obstacle_density(obstacles: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute local obstacle density for each cell using a square window.

    The density is the fraction of obstacle cells in a (window×window) neighbourhood
    around each cell.  Borders are handled by clipping the window at the grid edges.
    """
    h, w = obstacles.shape
    density = np.zeros_like(obstacles, dtype=np.float32)
    pad = window // 2
    padded = np.pad(obstacles.astype(np.float32), pad_width=pad, mode='constant')
    for i in range(h):
        for j in range(w):
            sub = padded[i : i + window, j : j + window]
            density[i, j] = np.mean(sub)
    return density


def compute_distance_to_obstacle(obstacles: np.ndarray) -> np.ndarray:
    """Compute distance from each free cell to the nearest obstacle.

    Cells containing obstacles are assigned a distance of zero.  The distances
    are computed using a simple breadth‐first search from all obstacle cells.
    """
    from collections import deque

    h, w = obstacles.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    q: deque[tuple[int, int]] = deque()
    # Initialize queue with obstacle cells
    for i in range(h):
        for j in range(w):
            if obstacles[i, j]:
                dist[i, j] = 0.0
                q.append((i, j))
    # 4‐neighbour BFS to compute Manhattan distances to nearest obstacle
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        i, j = q.popleft()
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                if dist[ni, nj] > dist[i, j] + 1:
                    dist[ni, nj] = dist[i, j] + 1
                    q.append((ni, nj))
    return dist


def generate_tsum(
    obstacles: np.ndarray,
    goal_cell: tuple[int, int],
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    tau_min: float = 0.1,
) -> np.ndarray:
    """Generate a task‑specific uncertainty map for a given obstacle map and goal.

    Parameters
    ----------
    obstacles : np.ndarray
        Boolean array indicating obstacle locations; shape `(H, W)`.
    goal_cell : tuple[int, int]
        Coordinates of the goal cell.
    alpha, beta, gamma : float
        Weights for task relevance, complexity and safety.  They should sum
        to 1.  In practice the algorithm normalises them internally.
    tau_min : float
        Minimum variance allowed anywhere on the map.

    Returns
    -------
    np.ndarray
        A 2D array of the same shape as ``obstacles`` containing the TSUM
        threshold at each cell.  Higher values correspond to regions where
        larger predictive variance is tolerated.
    """
    # Normalise weights to sum to 1
    total = alpha + beta + gamma
    if total == 0:
        raise ValueError("At least one weight must be positive.")
    alpha, beta, gamma = alpha / total, beta / total, gamma / total
    h, w = obstacles.shape
    # Task relevance kappa(x): invert normalised distance to goal
    dist_to_goal = compute_distance_to_goal((h, w), goal_cell)
    max_dist = np.max(dist_to_goal)
    kappa = 1.0 - (dist_to_goal / (max_dist + 1e-8))
    # Environmental complexity rho(x): local obstacle density
    rho = compute_local_obstacle_density(obstacles, window=5)
    # Safety phi(x): normalised distance to nearest obstacle
    dist_to_obs = compute_distance_to_obstacle(obstacles)
    max_dobs = np.max(dist_to_obs)
    phi = dist_to_obs / (max_dobs + 1e-8)
    # Compute tau(x)
    tau = alpha * kappa + beta * rho + gamma * phi
    tau = np.maximum(tau, tau_min)
    return tau

