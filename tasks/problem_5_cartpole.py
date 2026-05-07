"""Zadatak 5: reši CartPole pomoću DQN-a."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

from common import EvaluationResult, RewardTracker, evaluate_model


def make_cartpole() -> gym.Env:
    """Napravi CartPole okruženje i uključi praćenje nagrada po epizodi."""

    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def solve() -> tuple[DQN, list[float], EvaluationResult]:
    """Pokreni trening i evaluaciju za CartPole primer."""
    pass


if __name__ == "__main__":
    model, rewards, result = solve()
    print()
    print(f"Training episodes: {len(rewards)}")
    print(f"Last 100 training rewards: {np.mean(rewards[-100:]):.3f}")
    print(f"Evaluation reward: {result.avg_reward:.3f}")
    print(f"Evaluation steps: {result.avg_steps:.1f}")
