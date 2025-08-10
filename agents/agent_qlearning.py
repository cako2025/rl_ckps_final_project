import numpy as np
import gymnasium as gym

from collections import defaultdict
from tqdm import tqdm

from agents.agent_base import BaseAgent
from agents.agent_policies import EpsilonGreedyPolicy, SoftmaxPolicy


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with support for epsilon-greedy and softmax exploration strategies.

    This agent uses a tabular Q-learning algorithm to learn optimal policies
    in discrete action and state spaces. It supports epsilon-greedy and softmax
    exploration for action selection.

    Parameters
    ----------
    env : gym.Env
        The environment in which the agent interacts.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    alpha : float, optional
        Learning rate (alpha) for Q-value updates. Default is 0.1.
    gamma : float, optional
        Discount factor for future rewards. Default is 0.95.
    exploration : str, optional
        Exploration strategy to use. Must be 'epsilon_greedy' or 'softmax'.
        Default is 'epsilon_greedy'.
    epsilon_start : float, optional
        Initial epsilon value for epsilon-greedy exploration. Default is 1.0.
    epsilon_end : float, optional
        Minimum epsilon value for epsilon-greedy exploration. Default is 0.1.
    temperature : float, optional
        Temperature parameter for softmax exploration. Default is 1.0.
    """
    def __init__(
        self, env: gym.Env, seed: (int | None)=None, alpha=0.1, gamma=0.95, exploration='epsilon_greedy',
        epsilon_start=1.0, epsilon_end=0.1, temperature=1.0
    ) -> None:
        """
        Initialize the QLearningAgent.

        Parameters
        ----------
        env : gym.Env
            The environment to interact with.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        alpha : float, optional
            Learning rate for updating Q-values. Default is 0.1.
        gamma : float, optional
            Discount factor for future rewards. Default is 0.95.
        exploration : str, optional
            Exploration strategy: 'epsilon_greedy' or 'softmax'. Default is 'epsilon_greedy'.
        epsilon_start : float, optional
            Starting epsilon value for epsilon-greedy policy. Default is 1.0.
        epsilon_end : float, optional
            Ending epsilon value for epsilon-greedy policy. Default is 0.1.
        temperature : float, optional
            Temperature parameter for softmax policy. Default is 1.0.
        """
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.env = env
        self.seed = seed
        self.lr = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.temperature = temperature
        self.training_error = []

        if exploration == 'epsilon_greedy':
            self.policy = EpsilonGreedyPolicy(epsilon_start, epsilon_end)
        elif exploration == 'softmax':
            self.policy = SoftmaxPolicy(temperature)
        else:
            raise ValueError(f"Unknown exploration strategy: {exploration}")

        self.set_seed()

    def train(self, episodes=1000, eval_steps=None) -> dict:
        """
        Train the agent over a number of episodes using Q-learning.

        Parameters
        ----------
        episodes : int, optional
            Number of training episodes (default is 1000).
        eval_steps : int, optional
            Interval (in episodes) to evaluate the agent during training (default is None).

        Returns
        -------
        dict
            Dictionary containing:
            - "eval_results": List of evaluation results (if `eval_steps` is set).
            - "training_error": List of average TD errors per episode.
            - "return_queue": Episode returns recorded by the environment.
            - "length_queue": Episode lengths recorded by the environment.
            - "q_table": Q-table as a dictionary with list-converted values.

        Notes
        -----
        Applies epsilon decay if using epsilon-greedy exploration.
        Evaluation is performed periodically if `eval_steps` is provided.
        """
        self.eval_results = []
        self.training_error = []

        if self.exploration == 'epsilon_greedy':
            self.policy.reset_epsilon(episodes)

        for episode in tqdm(range(episodes), desc=f"qlearning - {self.exploration}"):
            state, _ = self.env.reset()
            done = False
            episode_td_errors = []

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                old_value = self.q_table[state][action]
                next_max = 0.0 if terminated or truncated else np.max(self.q_table[next_state])

                td_error = reward + self.gamma * next_max - old_value
                episode_td_errors.append(abs(td_error))
                self.q_table[state][action] = old_value + self.lr * td_error

                state = next_state
                done = terminated or truncated

            self.training_error.append(np.mean(episode_td_errors))

            if self.exploration == 'epsilon_greedy':
                self.policy.update_epsilon(episode)

            if eval_steps and episode % eval_steps == 0:
                self.eval_results.append(self.evaluate())

        return {
            "eval_rewards": self.eval_results,
            "training_error": self.training_error,
            "return_queue": self.env.return_queue,
            "length_queue": self.env.length_queue,
            "q_table": {
                state: values.tolist()
                for state, values in self.q_table.items()
            }
        }
