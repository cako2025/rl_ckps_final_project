import gzip
import pickle
import random
import gymnasium as gym
import numpy as np

from pathlib import Path
from datetime import datetime
from hydra import initialize, compose
from omegaconf import DictConfig
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def create_env(seed: int, size: int, is_slippery: bool = False, p: float = 0.9) -> gym.Env:
    """
    Create a FrozenLake-v1 Gym environment with a randomly generated map.

    Parameters
    ----------
    seed : int
        Random seed for environment reproducibility.
    size : int
        Size of the FrozenLake grid (size x size).
    is_slippery : bool, optional
        Whether the environment has slippery surfaces (default is False).
    p : float, optional
        Probability that each tile is frozen (default is 0.9).

    Returns
    -------
    env : gym.Env
        Configured FrozenLake environment.
    """
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, desc=generate_random_map(size=size, p=p, seed=size))
    set_seed(env=env, seed=seed)

    return env


def set_seed(env: gym.Env, seed: int) -> None:
    """
    Sets the random seed for reproducibility across Python's random, NumPy, and the Gym environment.

    Args:
        env (gym.Env): Gym environment whose seed will be set.
        seed (int): Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def load_config(config_name: str) -> DictConfig:
    """
    Loads a Hydra configuration from the specified config name.

    This function uses Hydra's `initialize` and `compose` functions to load 
    a configuration YAML file from the 'configs' directory.

    Args:
        config_name (str): The name of the configuration file (without the .yaml extension) 
                           to load from the 'configs' folder.

    Returns:
        DictConfig: The loaded Hydra configuration object.
    """
    with initialize(config_path="configs", version_base="1.2"):
        cfg = compose(config_name=config_name)

        return cfg


def save_results(results: list, output_path: str, filename: str) -> None:
    """
    Saves experiment results as a compressed Pickle (.pkl.gz) file.

    The function creates the output directory if it does not exist and stores
    the given results using gzip compression.

    Args:
        results (list): The list of experiment result dictionaries to save.
        output_path (str): The base directory path where the results should be stored.
        filename (str): The base filename (without extension).

    Returns:
        None
    """
    output_path = Path(output_path) / f'{filename}.pkl.gz'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, 'wb') as f:
        pickle.dump(results, f)

    log_message(f"Results saved to {output_path}")


def log_message(message: str, log_file: str = "experiment.log") -> None:
    """
    Logs a message to a specified log file with a timestamp.

    Args:
        message (str): The message to log.
        log_file (str): The path to the log file. Defaults to "experiment.log".

    Returns:
        None
    """
    output_path = Path(log_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().isoformat()}: {message}\n")
