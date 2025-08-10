import numpy as np

from collections import defaultdict
from tqdm import tqdm

from agents.agent_base import BaseAgent
from agents.agent_policies import EpsilonGreedyPolicy, SoftmaxPolicy


class SARSAAgent(BaseAgent):
    """
    SARSA agent implementing on-policy temporal difference learning.

    This agent learns a policy by updating Q-values based on the state-action-reward-state-action
    sequence using an exploration strategy (epsilon-greedy or softmax).

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
        Exploration strategy to use. Must be 'epsilon_greedy' or 'softmax'. Default is 'epsilon_greedy'.
    epsilon_start : float, optional
        Initial epsilon value for epsilon-greedy exploration. Default is 1.0.
    epsilon_end : float, optional
        Minimum epsilon value for epsilon-greedy exploration. Default is 0.1.
    temperature : float, optional
        Temperature parameter for softmax exploration. Default is 1.0.
    """
    def __init__(
        self, env, seed: (int | None)=None, alpha=0.1, gamma=0.95, exploration='epsilon_greedy',
        epsilon_start=1.0, epsilon_end=0.1, temperature=1.0
    ) -> None:
        """
        Initialize the SARSA agent.

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
        Train the agent over a number of episodes using SARSA.

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

        for episode in tqdm(range(episodes), desc=f"sarsa - {self.exploration}"):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False
            episode_td_errors = []

            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.choose_action(next_state)

                old_value = self.q_table[state][action]
                next_value = 0.0 if terminated or truncated else self.q_table[next_state][next_action]
                td_error = reward + self.gamma * next_value - old_value
                episode_td_errors.append(abs(td_error))
                self.q_table[state][action] = old_value + self.lr * td_error

                state, action = next_state, next_action
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
