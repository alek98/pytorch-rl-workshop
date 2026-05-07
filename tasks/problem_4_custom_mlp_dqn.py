"""Zadatak 4: koristi custom PyTorch mrežu unutar DQN modela."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common import EvaluationResult, RewardTracker, evaluate_model, make_frozen_lake



def solve() -> tuple[DQN, list[float], EvaluationResult]:
    """Pokreni trening i evaluaciju za zadatak 4."""
    pass

if __name__ == "__main__":
    _, rewards, result = solve()
    print(f"Training episodes: {len(rewards)}")
    print(f"Last 100 training rewards: {np.mean(rewards[-100:]):.3f}")
    print(f"Evaluation reward: {result.avg_reward:.3f}")
    print(f"Evaluation steps: {result.avg_steps:.1f}")
