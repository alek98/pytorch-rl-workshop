"""Zajedničke pomoćne funkcije za zadatke."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EvaluationResult:
    """Jednostavan rezultat evaluacije: prosečna nagrada i broj koraka."""

    avg_reward: float
    avg_steps: float


class RewardTracker(BaseCallback):
    """Sačuvaj ukupnu nagradu iz svake završene epizode."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        # Kada se epizoda završi, iz info rečnika čitamo njenu ukupnu nagradu.
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True


def make_frozen_lake(is_slippery: bool = False) -> gym.Env:
    """Napravi FrozenLake okruženje i uključi praćenje nagrada po epizodi."""

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
    # Čuvamo statistiku epizoda kako bismo kasnije pratili nagrade tokom treninga.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def evaluate_model(model, env: gym.Env, n_episodes: int = 100) -> EvaluationResult:
    """Evaluiraj model deterministički kroz više epizoda."""

    rewards = []
    steps_list = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()
    return EvaluationResult(avg_reward=float(np.mean(rewards)), avg_steps=float(np.mean(steps_list)))
