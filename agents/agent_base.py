import random
import numpy as np
import gymnasium as gym

from typing import Any
from collections import defaultdict
from abc import abstractmethod

from agents.agent_policies import EpsilonGreedyPolicy, SoftmaxPolicy
from utils import create_env


class BaseAgent:
    policy: EpsilonGreedyPolicy | SoftmaxPolicy
    q_table: defaultdict[Any, list[float]]
    seed: Any | None
    env: gym.Env

    def set_seed(self) -> None:
        """
        Set the random seed for reproducibility.

        This sets the seed for Python's `random` module and NumPy's random generator
        to ensure consistent results across different runs.

        Returns
        -------
        None
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

    def choose_action(self, state, evaluate=False) -> int:
        """
        Choose an action based on the current state and mode.

        Parameters
        ----------
        state : object
            The current state from which to choose an action.
        evaluate : bool, optional
            If True, selects the best-known action (greedy, no exploration).
            If False, uses the exploration policy. Default is False.

        Returns
        -------
        int
            The selected action.
        """
        if evaluate:
            return int(np.argmax(self.q_table[state]))

        return self.policy.select_action(self.q_table[state], self.env.action_space)

    def evaluate(self, n_episodes=100):
        """
        Evaluate the agent over multiple episodes in a new environment.

        Parameters
        ----------
        n_episodes : int, optional
            Number of evaluation episodes (default is 100).

        Returns
        -------
        dict
            Dictionary with average reward, average steps per episode, and fall rate.

        Notes
        -----
        The agent acts using a fixed policy (no exploration) during evaluation.
        """
        env = create_env(
            size=self.env.unwrapped.nrow, seed=self.seed,
            is_slippery=self.env.unwrapped.spec.kwargs['is_slippery']
        )

        rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False

            while not done:
                action = self.choose_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            rewards.append(reward)

        env.close()

        return np.mean(rewards)

    @abstractmethod
    def train(self, episodes=1000, eval_steps=None) -> dict:
        """
        Train the agent. Need to be implemented in the child-class.

        Parameters
        ----------
        episodes : int, optional
            Number of episodes to train. Default is 1000.
        eval_steps : int, optional
            Frequency of evaluation during training (in episodes). Default is None.

        Returns
        -------
        dict
            Training results or statistics.
        """
        pass