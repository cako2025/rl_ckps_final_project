import time
import numpy as np

from datetime import datetime
from smac import HyperbandFacade, Scenario
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter

from agents.agent_qlearning import QLearningAgent
from agents.agent_sarsa import SARSAAgent
from main import run_agent
from utils import create_env, log_message


##### constants #####
ALGORITHMS = {
    'qlearning': QLearningAgent,
    'sarsa': SARSAAgent
}
RUNS = 10
MIN_EPISODES = 500
EPISODES = 5000
MAP_SIZE = 7
PARENT_FOLDER_PATH = f"hpo/frozenlake/{MAP_SIZE}x{MAP_SIZE}"
LOG_FILE = f"{PARENT_FOLDER_PATH}/hpo.log"
RESULTS_FILE = f"{PARENT_FOLDER_PATH}/results.log"
N_TRIALS = np.inf
WALLTIME_LIMIT = 1800
#####################


def target_function(cfg: dict, seed: int, budget: float) -> float:
    """
    Evaluate agent configuration by average win rate over multiple runs.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing algorithm, exploration strategy,
        and corresponding hyperparameters.
    seed : int
        Random seed (not directly used; separate seeds used per run).
    budget : float
        Number of training episodes (converted to int).

    Returns
    -------
    float
        Objective value for optimization (1 - mean win rate).

    Notes
    -----
    Trains and evaluates the agent across `RUNS` different seeds and logs statistics
    including win rates, mean, and standard deviation.
    """
    win_rates = []
    episodes = int(budget)

    for i in range(RUNS):
        env = create_env(seed=i, size=MAP_SIZE)

        agent_class = ALGORITHMS[cfg["algorithm"]]
        alpha = cfg["alpha"]
        gamma = cfg["gamma"]
        exploration = cfg["exploration"]

        common = {"alpha": alpha, "gamma": gamma}

        if exploration == "epsilon_greedy":
            explore_cfg = {
                exploration: {
                    **common,
                    "epsilon_start": cfg["epsilon_start"],
                    "epsilon_end":   cfg["epsilon_end"]
                }
            }
        else:
            explore_cfg = {
                exploration: {
                    **common,
                    "temperature": cfg["temperature"]
                }
            }

        agent, result = run_agent(
            agent_class=agent_class,
            env=env,
            episodes=episodes,
            eval_steps=None,
            exploration=exploration,
            seed=i,
            cfg=explore_cfg,
        )

        eval_result = agent.evaluate(n_episodes=MIN_EPISODES)

        avg_reward = eval_result['avg_reward']
        win_rates.append(avg_reward)

    mean = np.mean(win_rates)
    std = np.std(win_rates)

    log_str = (
        f"\n--- Evaluation Result ---\n"
        f"Algorithm: {cfg['algorithm']}\n"
        f"Exploration: {cfg['exploration']}\n"
        f"Config:\n"
        f"  alpha: {cfg['alpha']:.4f}\n"
        f"  gamma: {gamma}\n"
    )

    if cfg["exploration"] == "epsilon_greedy":
        log_str += (
            f"  epsilon_start: {cfg['epsilon_start']:.4f}\n"
            f"  epsilon_end: {cfg['epsilon_end']:.4f}\n"
        )
    elif cfg["exploration"] == "softmax":
        log_str += f"  temperature: {cfg['temperature']:.4f}\n"

    log_str += (
        f"Winrates: {', '.join(f'{w:.4f}' for w in win_rates)}\n"
        f"Mean Winrate: {mean:.4f}\n"
        f"Standard Deviation: {std:.4f}\n"
        f"-------------------------\n"
    )

    log_message(log_str, RESULTS_FILE)

    return float(1 - mean)


def main(algorithm, exploration: str, seed: int=0) -> None:
    """
    Run hyperparameter optimization for a given algorithm and exploration strategy.

    Parameters
    ----------
    algorithm : str
        Learning algorithm to optimize (e.g., 'qlearning', 'sarsa').
    exploration : str
        Exploration strategy ('epsilon_greedy' or 'softmax').
    seed : int, optional
        Random seed for reproducibility (default is 0).

    Notes
    -----
    Configures and runs SMAC hyperparameter optimization using the HyperbandFacade.
    Logs the best found configuration to the specified log file.
    """
    cs = ConfigurationSpace(seed=seed)

    cs.add([
        CategoricalHyperparameter("algorithm", [algorithm]),
        CategoricalHyperparameter("exploration", [exploration]),
        UniformFloatHyperparameter("alpha", lower=0.001, upper=0.1, log=True),
        UniformFloatHyperparameter("gamma", lower=0.85, upper=1.00),
    ])

    if exploration == 'epsilon_greedy':
        cs.add(UniformFloatHyperparameter("epsilon_start", lower=0.5, upper=1.0))
        cs.add(UniformFloatHyperparameter("epsilon_end", lower=0.01, upper=0.1))
    elif exploration == 'softmax':
        cs.add(UniformFloatHyperparameter("temperature", lower=0.01, upper=2.0, log=True))

    scenario = Scenario(
        configspace=cs,
        deterministic=True,
        n_trials=N_TRIALS,
        walltime_limit=WALLTIME_LIMIT,
        output_directory=f"{PARENT_FOLDER_PATH}/smac_output_{algorithm}_{exploration}",
        min_budget=MIN_EPISODES,
        max_budget=EPISODES,
        seed=seed
    )

    smac = HyperbandFacade(
        scenario=scenario,
        target_function=target_function,
        overwrite=True
    )

    best_cfg = smac.optimize()

    log_message(f"Best configuration for {algorithm} with the policy {exploration} is: {best_cfg}", LOG_FILE)


if __name__ == "__main__":
    total_start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(f"\n=== Experiment started at {timestamp} ===", LOG_FILE)

    for algo, policy in [
        ('qlearning', 'epsilon_greedy'),
        ('qlearning', 'softmax'),
        ('sarsa', 'epsilon_greedy'),
        ('sarsa', 'softmax')
    ]:
        log_message(f"\n=== Optimizing {algo} with {policy} ===", LOG_FILE)
        start = time.time()
        main(algo, policy)
        end = time.time()

        duration = end - start
        log_message(f"{algo} + {policy}: {duration:.2f} seconds", LOG_FILE)

    total_end = time.time()
    total_duration = total_end - total_start
    log_message(f"✅ Total time: {total_duration:.2f} seconds", LOG_FILE)
    log_message(f"=== Experiment finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n", LOG_FILE)
