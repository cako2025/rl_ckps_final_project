# RL Final Project: Exploring Value Function Transfer Between On-Policy and Off-Policy Methods in Tabular Gridworld
Repository for the final project @LUHAI<br>
Title: Exploring Value Function Transfer Between On-Policy and Off-Policy Methods in Tabular Gridworld.<br>
Semester: Summer 2025.<br>
Group: rl_ckps.<br>
Members: Can Kocak and Paul Steinbrink.<br>
Documents: [`checklist.pdf`](documents/checklist.pdf), [`poster.pdf`](documents/poster.pdf), [`proposal.pdf`](documents/proposal.pdf) and [`report.pdf`](documents/report.pdf).

## Installation
1. Clone this repository:
    * ``git clone https://github.com/cako2025/rl_ckps_final_project.git``
2. Install the uv package manager:
    * ``pip install uv``
3. Create a new environment (with -seed it will put pip in the virtual environment):
    * ``uv venv --seed``
4. Activate the new env:
    * ``source .venv/bin/activate``
5. Install this repository:
    * ``python -m pip install -r requirements.txt``

## Computational Information
- MacBook Pro (Model Name: MacBook Pro (Model Identifier: MacBookPro15,4))
- Processor: 1,4 GHz Quad-Core Intel Core i5
- Memory (RAM): 8 GB 2133 MHz LPDDR3
- GPU: Intel Iris Plus Graphics 645 1536 MB
- macOS Version: 15.6
- Python Version: 3.11.1
- Versions of Key Python Packages: listed in [`requirements.txt`](requirements.txt)

Runtime:
Total runtime per experiment:
- approximately 5.5 hours with experiment config as in base.yaml

## Configurations
### Base
[`configs/base.yaml`](configs/base.yaml) defines the core experiment settings:
- Environment: FrozenLake-v1, 7x7 grid, slippery with high transition probability (`p=0.9`).
- Training: 5,000 steps per run, 10 runs per setup, and evaluation every 5 steps.
- Experiment Splits: Uses a single split ratio of 0.50 for transfer learning.
- Agent-Exploration Combinations: Runs all combinations of Q-Learning and SARSA, each with epsilon-greedy and softmax exploration strategies.
- Outputs: All results are saved to the `results/` directory.
This file acts as the main settings hub for the reinforcement learning transfer learning experiments on the FrozenLake environment.
### Q-Learning / SARSA
[`configs/qlearning.yaml`](configs/qlearning.yaml) and [`configs/sarsa.yaml`](configs/sarsa.yaml) are defining the hyperparameters for Q-Learning and SARSA agents with different exploration strategies:
- `epsilon_greedy`:
    - `gamma`: Discount factor for future rewards.
    - `alpha`: Learning rate.
    - `epsilon_start`/`epsilon_end`: Initial and final exploration rates.
- `softmax`:
    - `gamma`: Discount factor for future rewards.
    - `alpha`: Learning rate.
    - `temperature`: Controls randomness in action selection.

## Hyperparameter Optimization 
[`hpo.py`](hpo.py) uses [`SMAC`](https://github.com/automl/SMAC3) to optimize hyperparameters. This script runs automated hyperparameter optimization for all combinations of Q-Learning and SARSA agents with both epsilon-greedy and softmax exploration strategies on the FrozenLake environment.
- It uses the SMAC library and Hyperband algorithm to efficiently search for the best learning rates, discount factors, and exploration parameters.
- Results: The best configuration for each agent-policy pairing is logged to `hpo/frozenlake/7x7/hpo.log` for easy transfer into config files.
- Usage:
    - ``python -m hpo``

## Experiment Runs
The [`main.py`](main.py) file automates the main experiment pipeline:
- Workflow: For each run, it first trains a reinforcement learning agent (Q-Learning or SARSA with either epsilon-greedy or softmax exploration) as a “pre-agent.” It then transfers the pre-trained Q-table to a new “transfer agent” with all combinations of algorithms and exploration settings.
- Purpose: Evaluates transfer learning effects between different agent and exploration settings on the FrozenLake environment, using reproducible splits, multiple random seeds, and user-defined experiment configurations from [`configs/base.yaml`](configs/base.yaml).
- Output: Saves detailed result files (for later analysis and plotting) and logs to the `results/` directory.
- Usage:
    - ``python -m main``

## Plots
[`plot.py`](plot.py) loads and visualizes results from the experiment runs. It produces comparative plots for different agents and exploration strategies, including metrics like cumulative return, episode length, and training error with confidence intervals and statistical analysis.
- Features: Automatic data grouping, moving averages, AUC calculations, and variance/t-tests for statistical significance.
- Outputs: SVG plot files for each agent/strategy combination are saved in the `results/` directory.
- Usage:
    - ``python -m plot``

## Note on Docstrings

A large portion of the docstrings in this repository were automatically generated using **ChatGPT** (by OpenAI) to follow the **NumPy/SciPy documentation style** and to improve the readability and maintainability of the code.

**Prompt used to generate the docstrings:**

> *"docstring numpyscipy style short" (along with the corresponding function or code snippet pasted into the prompt)*

This approach was intentionally used to ensure a consistent and well-documented structure across all functions in the project. The generated descriptions were manually reviewed and adjusted where necessary.