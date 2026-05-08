"""Zadatak 5: reši CartPole pomoću DQN-a.

CartPole ima kontinualne opservacije, ali diskretne akcije. To je dobar primer
za DQN: neuronska mreža prima vektor stanja, a izlaz daje Q-vrednost za svaku
moguću akciju.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

try:
    from solutions.common import EvaluationResult, RewardTracker, evaluate_model
except ModuleNotFoundError:
    from common import EvaluationResult, RewardTracker, evaluate_model


def make_cartpole() -> gym.Env:
    """Napravi CartPole okruženje i uključi praćenje nagrada po epizodi."""

    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def train_cartpole(total_timesteps: int = 20_000) -> tuple[DQN, list[float]]:
    """Treniraj DQN agenta za CartPole okruženje."""

    env = make_cartpole()
    rewards = RewardTracker()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        learning_starts=500,
        target_update_interval=250,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=42,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=rewards)
    env.close()
    return model, rewards.episode_rewards


def print_env_comparison() -> None:
    """Ispiši zašto CartPole odgovara DQN algoritmu."""

    env = gym.make("CartPole-v1")

    print("CartPole observation space:", env.observation_space)
    print("CartPole action space:", env.action_space)
    print()
    print("Opservacija je kontinualan vektor od 4 broja.")
    print("Akcije su diskretne: 0 ili 1.")
    print("Zato DQN može da vrati dve Q-vrednosti: po jednu za svaku akciju.")

    env.close()


def solve() -> tuple[DQN, list[float], EvaluationResult]:
    """Pokreni trening i evaluaciju za CartPole primer."""

    model, rewards = train_cartpole()
    eval_env = make_cartpole()
    result = evaluate_model(model, eval_env)
    return model, rewards, result


if __name__ == "__main__":
    print_env_comparison()
    model, rewards, result = solve()
    print()
    print(f"Training episodes: {len(rewards)}")
    print(f"Last 100 training rewards: {np.mean(rewards[-100:]):.3f}")
    print(f"Evaluation reward: {result.avg_reward:.3f}")
    print(f"Evaluation steps: {result.avg_steps:.1f}")
