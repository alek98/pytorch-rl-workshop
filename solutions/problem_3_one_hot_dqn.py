"""Zadatak 3: istreniraj DQN nad one-hot FrozenLake opservacijama."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

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
        # Originalna opservacija je ceo broj. Posle wrapper-a postaje vektor,
        # na primer [1, 0, 0, ..., 0].
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.n_states,), dtype=np.float32)

    def observation(self, observation: int) -> np.ndarray:
        """Vrati one-hot vektor za dobijenu diskretnu opservaciju."""

        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[observation] = 1.0
        return one_hot


def train_one_hot_dqn(total_timesteps: int = 20_000) -> tuple[DQN, list[float]]:
    """Treniraj DQN nad FrozenLake okruženjem sa one-hot opservacijama."""

    env = make_frozen_lake(is_slippery=False)
    # Model vidi one-hot vektor, a ne originalni ceo broj koji predstavlja stanje.
    env = OneHotObservation(env)
    rewards = RewardTracker()

    model = DQN(
        "MlpPolicy",
        env,
        # Ove vrednosti su male i stabilne za 4x4 mapu.
        learning_rate=1e-3,
        learning_starts=100,
        target_update_interval=200,
        # Epsilon počinje visoko zbog istraživanja, pa opada tokom treninga.
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
    """Pokreni trening i evaluaciju za zadatak 3."""

    model, rewards = train_one_hot_dqn()
    eval_env = make_frozen_lake(is_slippery=False)
    # Evaluacija mora da koristi isti format opservacija kao trening.
    eval_env = OneHotObservation(eval_env)
    result = evaluate_model(model, eval_env)
    return model, rewards, result


if __name__ == "__main__":
    _, rewards, result = solve()
    print(f"Training episodes: {len(rewards)}")
    print(f"Last 100 training rewards: {np.mean(rewards[-100:]):.3f}")
    print(f"Evaluation reward: {result.avg_reward:.3f}")
    print(f"Evaluation steps: {result.avg_steps:.1f}")
