"""Zadatak 4: koristi custom MLP politiku za aproksimaciju DQN Q-vrednosti."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from solutions.common import (
        EvaluationResult,
        RewardTracker,
        evaluate_model,
        make_frozen_lake,
    )
except ModuleNotFoundError:
    from common import EvaluationResult, RewardTracker, evaluate_model, make_frozen_lake


class OneHotObservation(gym.ObservationWrapper):
    """Pretvori diskretan broj stanja u one-hot vektor."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.n_states = env.observation_space.n
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.n_states,), dtype=np.float32)

    def observation(self, observation: int) -> np.ndarray:
        """Vrati one-hot vektor za dobijenu diskretnu opservaciju."""

        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[observation] = 1.0
        return one_hot


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Jednostavna PyTorch mreža koja uči reprezentaciju opservacije."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        n_inputs = observation_space.shape[0]

        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Izračunaj feature-e iz one-hot opservacije."""
        return self.network(observations)


def train_custom_mlp_dqn(total_timesteps: int = 20_000) -> tuple[DQN, list[float]]:
    """Treniraj DQN koristeći custom PyTorch feature extractor."""

    env = make_frozen_lake(is_slippery=False)
    # Zadržavamo one-hot ulaz iz zadatka 3 i menjamo samo policy mrežu.
    env = OneHotObservation(env)
    rewards = RewardTracker()

    policy_kwargs = {
        # Ovde ubacujemo svoju PyTorch mrežu sa forward metodom.
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        # QNetwork zatim dodaje samo linearni sloj od feature-a do Q-vrednosti.
        "net_arch": [],
    }

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        learning_starts=100,
        target_update_interval=200,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        seed=42,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=rewards)
    env.close()
    return model, rewards.episode_rewards


def solve() -> tuple[DQN, list[float], EvaluationResult]:
    """Pokreni trening i evaluaciju za zadatak 4."""

    model, rewards = train_custom_mlp_dqn()
    eval_env = make_frozen_lake(is_slippery=False)
    # Istrenirani model i tokom evaluacije očekuje one-hot opservacije.
    eval_env = OneHotObservation(eval_env)
    result = evaluate_model(model, eval_env)
    return model, rewards, result


if __name__ == "__main__":
    _, rewards, result = solve()
    print(f"Training episodes: {len(rewards)}")
    print(f"Last 100 training rewards: {np.mean(rewards[-100:]):.3f}")
    print(f"Evaluation reward: {result.avg_reward:.3f}")
    print(f"Evaluation steps: {result.avg_steps:.1f}")
