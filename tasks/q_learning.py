import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class Q_Agent():
    def __init__(self, env: gym.Env, alpha:int, gamma:int, epsilon:int, n_episodes:int, seed:int=42):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.qtable = np.zeros([self.n_states, self.n_actions])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.seed = seed
        np.random.seed(seed)

    def learn(self) -> list:
        '''Returns a list of rewards from training.'''
        pass

    def predict(self, state: np.ndarray) -> int:
        '''Returns the policy action from a state. Implements Exploration Exploitation trade-off'''
        pass
    
    def update_table(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float):
        pass

    def n_actions(self) -> int:
        '''Returns number of actions.'''
        return self.n_actions

    def n_states(self) -> int:
        '''Returns number of states.'''
        return self.n_states

    def get_v_table(self) -> np.ndarray:
        '''Returns v-table from q-table.'''
        v_table = np.max(self.qtable, axis=1).reshape((4,4))
        return v_table
    
    def test_agent(self, n_episodes:int) -> tuple[int, int]:
        rewards = []
        n_steps = []
        for episode in range(n_episodes):
            state, _ = self.env.reset(seed=42)
            done, in_ep_n_steps, in_ep_reward = False, 0, 0
            while not done:
                # action = agent.predict(state)             # exploration-exploitation
                action = np.argmax(self.qtable[state,:])    # only exploitation
                state, reward, done, _, _ = self.env.step(action)
                in_ep_reward += reward
                in_ep_n_steps += 1

            rewards.append(in_ep_reward)
            n_steps.append(in_ep_n_steps)

        avg_n_steps = sum(n_steps) / n_episodes
        avg_reward = sum(rewards) / n_episodes
        return avg_n_steps, avg_reward

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='rgb_array')
    env.reset(seed=42)
    agent = Q_Agent(env, alpha=0.1, gamma=0.9, epsilon=0.6, n_episodes=1000)
    rewards = agent.learn()
    plt.plot(rewards)
    plt.show()
    
    # promeni render_mode na 'human' kako bi mogao da vizualizujes agenta u okruženju.
    agent.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
    avg_n_steps, avg_reward = agent.test_agent(2)
    print(f'{avg_n_steps=}')
    print(f'{avg_reward=}')
    env.close()