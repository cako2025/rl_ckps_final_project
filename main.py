import gymnasium as gym

from omegaconf import DictConfig
from datetime import datetime, timedelta
from copy import deepcopy
from time import time

from agents.agent_qlearning import QLearningAgent
from agents.agent_sarsa import SARSAAgent
from utils import create_env, load_config, log_message, save_results


##### constants #####
ALGORITHMS = {
    'qlearning': QLearningAgent,
    'sarsa': SARSAAgent
}
####################


def run_agent(
        agent_class: type, env: gym.Env, episodes: int, exploration: str, seed: int, 
        cfg: DictConfig, eval_steps: int=None, q_table: dict | None=None,
    ):
    """
    Train an agent in the given environment, optionally using a pre-trained Q-table.

    Parameters
    ----------
    agent_class : type
        Class of the agent to be instantiated and trained.
    env : gym.Env
        The Gym environment in which the agent is trained.
    episodes : int
        Number of training episodes.
    exploration : str
        Exploration strategy to be used by the agent.
    seed : int
        Random seed for reproducibility.
    cfg : DictConfig
        Configuration dictionary for the agent.
    eval_steps : int, optional
        Number of evaluation steps per evaluation phase (default is None).
    q_table : dict, optional
        Pre-trained Q-table to initialize the agent (default is None).

    Returns
    -------
    agent :
        The trained agent instance.
    result :
        Training result returned by the agent's `train` method.
    """
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episodes)
    agent = agent_class(env, seed=seed, exploration=exploration, **cfg[exploration])

    ##### transfer learning copy q_table #####
    if q_table is not None:
        agent.q_table.update(deepcopy(q_table))
    ############################################

    result = agent.train(episodes=episodes, eval_steps=eval_steps)

    env.close()
    return agent, result


def run_experiments_with_transfer() -> None:
    """
    Run transfer learning experiments across configurations and save results.

    Notes
    -----
    Iterates over data splits and agent combinations. Performs pre-training followed by
    transfer learning using the pre-trained Q-table. Logs all steps and saves results
    for each split to a structured output path based on configuration and date.
    """
    date = datetime.now().strftime("%Y%m%d")
    base_cfg: DictConfig = load_config('base')
    log_message(f"Running experiments with base configuration: {base_cfg}")

    for split in base_cfg.splits:
        split_str = str(split).replace('.', '')
        pretrain_episodes = int(base_cfg.training_steps * (split / (1 - split)))
        tl_episodes = int(base_cfg.training_steps)
        log_message(f"Running experiments with split: {split}")
        results = []

        for pre_combo in base_cfg.combinations:
            pre_algorithm = pre_combo.algorithm
            pre_exploration = pre_combo.exploration

            for seed in range(base_cfg.runs):
                ##### pre training #####
                log_message(f"[START] pre training {pre_algorithm} {pre_exploration} | split: {split} | episodes: {pretrain_episodes} | seed: {seed}")
                env = create_env(seed=seed, size=base_cfg.env_kwargs.map_size, p=base_cfg.env_kwargs.p, is_slippery=base_cfg.env_kwargs.is_slippery)

                pre_agent, pre_result = run_agent(
                    agent_class=ALGORITHMS[pre_algorithm], env=env, episodes=pretrain_episodes,
                    exploration=pre_exploration, seed=seed, cfg=load_config(pre_algorithm)
                )

                log_message(f"[END] pre training {pre_algorithm} {pre_exploration} | split: {split} | episodes: {pretrain_episodes} | seed: {seed}")
                ########################

                ##### transfer learning: agent2 takes over Q-Table from agent1 #####
                for tl_combo in base_cfg.combinations:
                    tl_algorithm = tl_combo.algorithm
                    tl_exploration = tl_combo.exploration
                    log_message(f"[START] transfer training {tl_algorithm} {tl_exploration} | split: {split} | episodes: {tl_episodes} | seed: {seed}")

                    env = create_env(seed=seed, size=base_cfg.env_kwargs.map_size, p=base_cfg.env_kwargs.p, is_slippery=base_cfg.env_kwargs.is_slippery)

                    tl_agent, tl_result = run_agent(
                        agent_class=ALGORITHMS[tl_algorithm], env=env, episodes=tl_episodes, eval_steps=base_cfg.eval_steps,
                        exploration=tl_exploration, seed=seed, cfg=load_config(tl_algorithm), q_table=pre_agent.q_table
                    )

                    results.append({
                        'seed': seed,
                        'preagent': {
                            'algorithm': pre_algorithm,
                            'exploration': pre_exploration,
                            'episodes': pretrain_episodes
                        },
                        'agent': {
                            'algorithm': tl_algorithm,
                            'exploration': tl_exploration,
                            'episodes': tl_episodes,
                            'eval_steps': base_cfg.eval_steps
                        },
                        'results': tl_result
                        })
                    
                    log_message(f"[END] transfer training {tl_algorithm} {tl_exploration} | split: {split} | episodes: {tl_episodes} | seed: {seed}")
                #################################################

        save_results(results, '/'.join([base_cfg.output_path, base_cfg.env_name_lower, date]), split_str)
        log_message(f"Finished all runs for split: {split}")
    ####################################################################


if __name__ == '__main__':
    start_time = time()
    log_message("Starting main.py...")

    run_experiments_with_transfer()

    duration = time() - start_time
    log_message(f"Finished in: {timedelta(seconds=round(duration))}")
