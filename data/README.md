This directory holds datasets used by the BNN training procedure.  The
BNN dataset consists of triplets of (state, action, next_state) collected
by executing a random policy in the environment.  For reproducibility a
small dataset may be stored here as a `.npz` file; however, the
training script (`rl_training.py`) automatically generates fresh data
when training the TSUM+BNN agent.
