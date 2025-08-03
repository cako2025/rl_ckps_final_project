import random
import numpy as np


class EpsilonGreedyPolicy:
    """
    Epsilon-Greedy policy for action selection in reinforcement learning.

    Attributes
    ----------
    epsilon_start : float
        Initial epsilon value (exploration rate).
    epsilon_end : float
        Minimum epsilon value after decay.
    epsilon : float
        Current epsilon value.
    decay_rate : float
        Multiplicative decay rate for epsilon per episode.
    """
    def __init__(self, epsilon_start: float, epsilon_end: float) -> None:
        """
        Initialize the EpsilonGreedyPolicy.

        Parameters
        ----------
        epsilon_start : float
            Starting value of epsilon (exploration probability).
        epsilon_end : float
            Minimum value of epsilon after decay.
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.decay_rate = 1
    
    def reset_epsilon(self, episodes: int) -> None:
        """
        Reset and compute the epsilon decay rate based on total episodes.

        Parameters
        ----------
        episodes : int
            Number of episodes over which epsilon decays from epsilon_start to epsilon_end.
        """
        self.epsilon = self.epsilon_start
        self.decay_rate = (self.epsilon_end / self.epsilon_start) ** (1 / episodes)

    def select_action(self, q_values, action_space) -> int:
        """
        Select an action using the epsilon-greedy strategy.

        Parameters
        ----------
        q_values : array-like
            Estimated action-value function for the current state.
        action_space : gym.Space
            The action space of the environment, used to sample random actions.

        Returns
        -------
        int
            Selected action index.
        """
        if random.random() < self.epsilon:
            return action_space.sample()

        return int(np.argmax(q_values))
    
    def update_epsilon(self, episode: int) -> None:
        """
        Update the epsilon value based on the current episode.

        Parameters
        ----------
        episode : int
            Current episode number.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.decay_rate ** episode))


class SoftmaxPolicy:
    """
    Softmax exploration policy for action selection in reinforcement learning.

    This policy selects actions based on a probability distribution derived from
    the softmax of Q-values, controlled by a temperature parameter.

    Parameters
    ----------
    temperature : float
        Temperature parameter controlling the randomness of action selection.
        Higher values lead to more uniform probabilities; lower values favor higher Q-values.
    """
    def __init__(self, temperature) -> None:
        """
        Initialize the SoftmaxPolicy.

        Parameters
        ----------
        temperature : float
            Temperature parameter controlling the randomness of action selection.
            Higher values lead to more uniform probabilities; lower values favor higher Q-values.
        """
        self.temperature = temperature

    def select_action(self, q_values, action_space) -> int:
        """
        Select an action using a softmax distribution over Q-values.

        Parameters
        ----------
        q_values : array-like
            Q-values for the available actions.
        action_space : gym.Space
            The action space of the environment (unused here but kept for API consistency).

        Returns
        -------
        int
            Selected action index.
        """
        q_values = np.asarray(q_values, dtype=np.float32)
        z = q_values - np.max(q_values)
        exp_q = np.exp(z / self.temperature)
        probs = exp_q / np.sum(exp_q)

        return int(np.random.choice(q_values.shape[0], p=probs))
