"""Zadatak 3: istreniraj DQN nad one-hot FrozenLake opservacijama."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN

from common import EvaluationResult, RewardTracker, evaluate_model, make_frozen_lake

def solve() -> tuple[DQN, list[float], EvaluationResult]:
    """Pokreni trening i evaluaciju za zadatak 3."""
    pass


if __name__ == "__main__":
    _, rewards, result = solve()
    print(f"Training episodes: {len(rewards)}")
    print(f"Last 100 training rewards: {np.mean(rewards[-100:]):.3f}")
    print(f"Evaluation reward: {result.avg_reward:.3f}")
    print(f"Evaluation steps: {result.avg_steps:.1f}")
