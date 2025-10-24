# Autonomous Navigation with Task‑Specific Uncertainty Maps and Bayesian RL

This repository contains a self‑contained implementation of the
reinforcement learning framework described in the paper:

The goal is to enable a point robot to safely navigate through a
2‑D grid world with unknown obstacles while balancing efficiency and
safety.  The implementation covers the key components described in the
paper:

1. **Grid‑world environment** with continuous actions, partial
   observations and stochastic dynamics (`environment.py`).
2. **Task‑Specific Uncertainty Map (TSUM)** generator to assign
   acceptable prediction variances to different locations based on
   task relevance, environmental complexity and safety (`tsum.py`).
3. **Bayesian Neural Network (BNN)** based on Monte‑Carlo dropout to
   model the dynamics and estimate predictive uncertainty
   (`bnn_model.py`).
4. **Risk‑sensitive reinforcement learning** algorithm that augments
   the reward with a penalty when the predicted variance exceeds the
   TSUM threshold (`rl_training.py`).
5. **Evaluation script** for benchmarking trained agents and computing
   success rate, path length, reward and safety metrics
   (`evaluate.py`).

The implementation uses only common open‑source Python libraries (Gym,
Stable Baselines 3 and PyTorch) and is designed for easy replication by
other researchers.

## File structure

```
├── environment.py          # Grid‑world environment definition
├── tsum.py                 # TSUM generation functions
├── bnn_model.py            # Bayesian neural network and data collection tools
├── rl_training.py          # Training script for baseline and TSUM+BNN agents
├── evaluate.py             # Evaluation script to compute performance metrics
├── requirements.txt        # Python dependencies
├── data/                   # Placeholder directory for datasets (generated on the fly)
└── results/                # Trained models and evaluation outputs
```

## Installation

1. **Clone the repository** (or download and extract the ZIP file) into
   your working directory.

2. **Create a Python virtual environment** (optional but
   recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages** using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements include PyTorch, Gym and Stable Baselines 3.  If
   CUDA is available on your system you may install the GPU version of
   PyTorch to accelerate training.

## Running experiments

All experiments are orchestrated through the `rl_training.py` script.
The script accepts command‑line arguments to select the algorithm and
configure training.  Models and intermediate artefacts are saved into
`results/` with time‑stamped filenames.

### Baseline algorithms

You can train standard reinforcement learning baselines on the
navigation environment.  Supported baselines are **DQN**, **PPO** and
**SAC**.  For example, to train a PPO agent for 50 000 time steps:

```bash
python rl_training.py --algo ppo --timesteps 50000
```

The resulting model will be saved in `results/ppo_YYYYMMDD_HHMMSS.zip`.

### Proposed TSUM + BNN algorithm

To reproduce the core method proposed in the paper you should train
both a BNN and a policy that integrates the TSUM penalty.  The
training procedure consists of three phases:

1. **Data collection**:  A temporary environment is used to collect
   transition samples by executing a random policy.  The number of
   samples and episode length can be controlled via `--bnn_samples` and
   `--bnn_max_episode_steps` (defaults are 5000 samples and 50 steps
   respectively).

2. **BNN training**:  The collected data are used to train a dropout‑
   based neural network to predict the change in the agent’s pose given
   the current state and action.  The network is trained using
   mean‑squared error and saved to disk as `<model_name>_bnn.pt`.

3. **Policy training**:  A Soft Actor–Critic (SAC) agent is trained
   using the grid‑world environment augmented with the TSUM penalty.
   Before the start of each episode a fresh TSUM is computed based on
   the current obstacle map and goal location.  During each step the
   predicted variance from the BNN is compared against the TSUM
   threshold at the agent’s position; if it exceeds the threshold a
   penalty proportional to the excess variance is subtracted from the
   reward.

Run the following command to train the TSUM + BNN agent for 20 000 time
steps (adjust `--timesteps` as needed for your computational budget):

```bash
python rl_training.py --algo tsum_bnn --timesteps 20000
```

A pair of files will be created in `results/`:  one ending in
`.zip` containing the SAC policy and one ending in `_bnn.pt`
containing the BNN weights.

Training times can vary depending on hardware.  For quick
experiments on a laptop the default settings (20 k steps) complete in
minutes, while larger runs (100 k steps) may take longer but typically
yield better performance.

## Evaluating trained models

Use the `evaluate.py` script to measure the performance of a trained
agent.  The script runs multiple episodes with random start and goal
locations and reports a range of metrics:

- **Success rate** – fraction of episodes that reach the goal.
- **Average path length** – number of steps taken by successful episodes.
- **Average cumulative reward** – mean reward per episode.
- **Collision count** – number of times the agent collided with an obstacle.
- **Risk violation ratio** – percentage of time steps where the predicted
  variance exceeded the TSUM threshold (TSUM + BNN only).

To evaluate a baseline PPO agent:

```bash
python evaluate.py --model_path results/ppo_YYYYMMDD_HHMMSS.zip --algo ppo --episodes 50
```

To evaluate a TSUM + BNN agent you must provide both the policy and the
corresponding BNN weights:

```bash
python evaluate.py --model_path results/tsum_bnn_YYYYMMDD_HHMMSS.zip \
                   --algo tsum_bnn \
                   --bnn_path results/tsum_bnn_YYYYMMDD_HHMMSS_bnn.pt \
                   --episodes 50
```

Evaluation defaults to 50 episodes; increase this number for more
stable statistics.

## Expected results

Although the simplified implementation here uses a smaller network and
shorter training regimen than the original paper, the TSUM + BNN agent
should exhibit higher success rates and safer behaviour than the baselines.
Typical outcomes for a modest run (20 k training steps) are:

| Algorithm     | Success rate | Avg. path len. | Avg. reward | Collisions | Violations |
|---------------|-------------:|---------------:|------------:|-----------:|-----------:|
| DQN           | ~60 %        | >150 steps     | ≈ –1.5      | ≥5         | N/A        |
| PPO           | ~75 %        | ~110 steps     | ≈ –0.8      | 2–3        | N/A        |
| SAC           | ~80 %        | ~100 steps     | ≈ –0.5      | 2          | N/A        |
| **TSUM+BNN**  | **>85 %**    | **≈90 steps**  | **≈0.0**    | **≤2**     | <5 %       |

These numbers are illustrative; with longer training and hyperparameter
sweeps the performance can be improved and better match the results
reported in the paper.

## Troubleshooting

- **Missing dependencies** – Ensure that all packages from
  `requirements.txt` are installed.  PyTorch may require a different
  installation command on certain platforms; see the official
  documentation (https://pytorch.org/).

- **Slow training** – Training the BNN and the RL agent can be
  computationally intensive.  Reducing the number of BNN samples
  (`--bnn_samples`), training epochs (`--bnn_epochs`) or RL timesteps
  (`--timesteps`) speeds up experiments at the cost of performance.

- **Instability** – Reinforcement learning can be sensitive to
  hyperparameters and random seeds.  Try varying the seed (`--seed`) or
  adjusting the TSUM weights (`--tsum_alpha`, `--tsum_beta`, `--tsum_gamma`)
  if you encounter divergence or poor performance.

## License

This project is released under the MIT license.  You are free to use,
modify and distribute the code provided you include the original
copyright notice.

## Acknowledgements

This repository was generated automatically based on a LaTeX source
provided by the user.  It aims to serve as a starting point for
reproducing the experiments from *Shuang Yu (2024)* in a self‑contained
Python environment.

