import gzip
import pickle
import numpy as np

from typing import Literal
from collections import defaultdict
from matplotlib import pyplot as plt, figure, axes
from sklearn.metrics import auc
from scipy.stats import ttest_ind, levene
from scipy.ndimage import uniform_filter1d

from utils import log_message


class plotAgentResults():
    fig: figure.Figure
    ax: axes.Axes

    def __init__(self, file_path: str, metrics: list, rolling_length: int, log_file: str="results/plot.log") -> None:
        """
        Initialize the class by loading data and setting parameters.

        Parameters
        ----------
        file_path : str
            Path to the data file to load.
        metrics : list
            List of metric names to process and plot.
        rolling_length : int
            Number of elements for rolling window in statistical tests.
        log_file : str, optional
            Path to the log file (default is "results/plot.log").
        """
        self.log_file = log_file
        self.metrics = metrics
        self.tl_grouped = []
        self.data_tl = []
        self.rolling_length = rolling_length

        self._load_data(file_path)

    def _stack_and_avg_after_ma(self, arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a uniform moving average to each array and compute the mean and standard deviation.

        This function applies a uniform filter (moving average) with window size `self.rolling_length`
        to each input array and then computes the element-wise mean and standard deviation across all
        smoothed arrays.

        Parameters
        ----------
        arrays : list of np.ndarray
            List of 1D NumPy arrays, each representing a metric over time. All arrays should be of equal length.

        Returns
        -------
        mean : np.ndarray
            1D array containing the mean values of the smoothed input arrays at each time step.

        std : np.ndarray
            1D array containing the standard deviation of the smoothed input arrays at each time step.

        Notes
        -----
        Uses `scipy.ndimage.uniform_filter1d` with default mode='reflect'. Assumes all input arrays
        have the same length. If arrays are of unequal length, behavior may be undefined.
        """
        window = self.rolling_length

        smoothed = np.array([
            uniform_filter1d(arr, size=window)
            for arr in arrays
        ])

        mean = np.mean(smoothed, axis=0)
        std = np.std(smoothed, axis=0)

        return mean, std

    def _compute_auc(self, x: np.ndarray, y: np.ndarray) -> float:
        return auc(x, y)

    def _var_tests(self, base_arr: np.ndarray, compare_arr: np.ndarray) -> None:
        """
        Perform variance and t-tests on segments of two arrays.

        Parameters
        ----------
        base_arr : np.ndarray
            Baseline data array.
        compare_arr : np.ndarray
            Comparison data array.

        Notes
        -----
        Tests the first and last `rolling_length` elements using Levene's test for equal variances,
        followed by Student's or Welch's t-test depending on variance equality.
        Logs test statistics and p-values.
        """
        alpha = 0.05
        n = self.rolling_length

        segments = [
            ('start', base_arr[:n], compare_arr[:n]),
            ('end', base_arr[-n:], compare_arr[-n:])
        ]

        for name, base_seg, compare_seg in segments:
            if len(base_seg) == 0 or len(compare_seg) == 0:
                log_message(f"[{name}] too short for tests (length base_arr={len(base_arr)}, compare_arr={len(compare_arr)})", self.log_file)
                continue

            stat_levene, p_levene = levene(base_seg, compare_seg)
            log_message(f"[{name}] Levene-Test: W={stat_levene:.3f}, p={p_levene:.3f}", self.log_file)

            if p_levene >= alpha:
                stat_t, p_t = ttest_ind(base_seg, compare_seg, equal_var=True)
                test_name = f"[{name}] Student's t-Test"
            else:
                stat_t, p_t = ttest_ind(base_seg, compare_seg, equal_var=False)
                test_name = f"[{name}] Welch's t-Test"
            log_message(f"{test_name}: t={stat_t:.3f}, p={p_t:.3f}", self.log_file)

    def create_plot(self, agent: Literal['qlearning', 'sarsa'], exploration: Literal['epsilon_greedy', 'softmax']) -> None:
        """
        Generate plots for specified agent and exploration strategy.

        Parameters
        ----------
        agent : {'qlearning', 'sarsa'}
            The learning agent algorithm to filter data.
        exploration : {'epsilon_greedy', 'softmax'}
            The exploration strategy to filter data.

        Notes
        -----
        Groups data, computes metrics, plots results with AUC and confidence intervals,
        performs statistical tests against baseline, and adds legends.
        Logs messages throughout the process.
        """
        legend_handles = []
        legend_labels = []

        self._group_data(agent=agent, exploration=exploration)

        if len(self.tl_grouped) < 1:
            log_message("No data to plot. Please check the input files.", self.log_file)
            return

        self.fig, self.ax = plt.subplots(
            ncols=len(self.metrics),
            figsize=(20, 5),
            gridspec_kw={'wspace': 0.3}
        )

        if len(self.metrics) == 1:
            self.ax = [self.ax]

        for i, metric in enumerate(self.metrics):
            auc_tl = {}
            all_labels = []
            all_curves = []

            for (tl_alg, tl_exp, pre_alg, pre_exp), runs in self.tl_grouped.items():
                if len(runs[0]['results'].get('eval_results', {})) > 0 and metric in runs[0]['results']['eval_results'][0]:
                    tl_metric_all = [
                        np.array([entry[metric] for entry in run['results']['eval_results']]).flatten()
                        for run in runs
                    ]
                elif metric in runs[0]['results']:
                    tl_metric_all = [np.array(run['results'][metric]).flatten() for run in runs]
                else:
                    log_message(f"Metric '{metric}' not found in results for {tl_alg} {tl_exp} with pre-trained {pre_alg} {pre_exp}. Skipping.", self.log_file)
                    continue
                tl_metric_ma, tl_metric_ma_std = self._stack_and_avg_after_ma(tl_metric_all)
                tl_metric_x = np.arange(len(tl_metric_ma))


                auc_area = self._compute_auc(tl_metric_x, tl_metric_ma)

                label = f"{tl_alg} {tl_exp} | pre:{pre_alg} {pre_exp}"
                line, = self.ax[i].plot(tl_metric_x, tl_metric_ma, label=f"AUC: {auc_area:.2f}")
                self.ax[i].fill_between(tl_metric_x, tl_metric_ma - tl_metric_ma_std, tl_metric_ma + tl_metric_ma_std, alpha=0.2)
                self.ax[i].legend(loc='best', fontsize='medium')

                all_curves.append(tl_metric_ma)
                all_labels.append(label)
                auc_tl[label] = auc_area

                if i == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)

            baseline_label = f"{agent} {exploration} | pre:{agent} {exploration}"

            for idx_a in range(len(all_curves)):
                if all_labels[idx_a] != baseline_label:
                    continue
                for idx_b in range(len(all_curves)):
                    if idx_a == idx_b:
                        continue
                    log_message(f"=== Statistical test for [{metric}] {all_labels[idx_a]} vs {all_labels[idx_b]} ===", self.log_file)
                    self._var_tests(all_curves[idx_a], all_curves[idx_b])


            log_message(f"=== AUC for {metric} ===", self.log_file)
            for k,v in auc_tl.items():   log_message(f"{k}: {v:.3f}", self.log_file)

        self.fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize='medium')
        plt.tight_layout(rect=[0, 0, 1, 0.90])

    def show_plot(self) -> None:
        """
        Display the current plot.

        Notes
        -----
        Blocks execution until the plot window is closed.
        """
        plt.show()

    def save_plot(self, output_path: str, transparent: bool=False, dpi: int=None) -> None:
        """
        Save the current plot to a file and clear the figure.

        Parameters
        ----------
        output_path : str
            File path to save the plot.
        transparent : bool, optional
            Whether to save the plot with a transparent background (default is False).
        dpi : int, optional
            Resolution in dots per inch (default is None, uses matplotlib default).

        Notes
        -----
        Clears and closes the plot after saving.
        """
        plt.savefig(output_path, bbox_inches='tight', transparent=transparent, dpi=dpi)
        plt.clf()
        plt.close()
        log_message(f"Plot saved to {output_path}", self.log_file)

    def set_plot_cfg(self, titles: list[str]=None, xlabels: list[str]=None, ylabels: list[str]=None, xscales: list[str]=None) -> None:
        """
        Configure plot titles, axis labels, and x-axis scales for subplots.

        Parameters
        ----------
        titles : list of str, optional
            Titles for each subplot.
        xlabels : list of str, optional
            Labels for x-axes.
        ylabels : list of str, optional
            Labels for y-axes.
        xscales : list of str, optional
            Scale types ('linear', 'log', etc.) for x-axes.

        Notes
        -----
        Applies settings up to the number of metrics/subplots available.
        """
        if titles:
            for i in range(min(len(self.metrics), len(titles))):
                self.ax[i].set_title(titles[i], fontsize=16)

        if xlabels:
            for i in range(min(len(self.metrics), len(xlabels))):
                self.ax[i].set_xlabel(xlabels[i], fontsize=13)

        if ylabels:
            for i in range(min(len(self.metrics), len(ylabels))):
                self.ax[i].set_ylabel(ylabels[i], fontsize=13)
        
        if xscales:
            for i in range(min(len(self.metrics), len(xscales))):
                self.ax[i].set_xscale(xscales[i])

                if xscales[i] == 'log':
                    self.ax[i].set_xlabel(self.ax[i].get_xlabel() + " (log)", fontsize=13)

    def _load_data(self, file_path: str) -> None:
        """
        Load and append data from a gzipped pickle file.

        Parameters
        ----------
        file_path : str
            Path to the gzipped pickle file containing the data.

        Notes
        -----
        Data is loaded as a list to `self.data_tl`.
        """
        with gzip.open(file_path, "rb") as f:
            data = pickle.load(f)

            self.data_tl = [entry for entry in data]

    def _group_data(self, agent: Literal['qlearning', 'sarsa'], exploration: Literal['epsilon_greedy', 'softmax']) -> None:
        """
        Group runs by algorithm and exploration strategy for the specified agent.

        Parameters
        ----------
        agent : {'qlearning', 'sarsa'}
            The learning agent algorithm to filter runs.
        exploration : {'epsilon_greedy', 'softmax'}
            The exploration strategy to filter runs.

        Notes
        -----
        Groups runs where both the current and pre-agent's algorithm and exploration
        match the specified criteria, storing them in `self.tl_grouped`.
        """
        self.tl_grouped = defaultdict(list)

        for run in self.data_tl:
            alg = run['agent']['algorithm']
            exp = run['agent']['exploration']
            pre_alg = run['preagent']['algorithm']
            pre_exp = run['preagent']['exploration']

            if alg == agent and exp == exploration:
                key = (alg, exp, pre_alg, pre_exp)
                self.tl_grouped[key].append(run)


if __name__ == "__main__":
    file_path = "results/<FOLDER_NAME>/05.pkl.gz"  # change file path as needed
    metrics = ['return_queue', 'length_queue', 'training_error']
    rolling_length = 500

    plot_instance = plotAgentResults(file_path=file_path, metrics=metrics, rolling_length=rolling_length)

    for agent in ['qlearning', 'sarsa']:
        for exploration in ['epsilon_greedy', 'softmax']:
            titles = ['Cumulative Return', 'Episode Length', 'Training Error']
            xlabels = ['Episodes'] * len(metrics)
            ylabels = ['Return (moving average)', 'Steps per episode (moving average)', 'Mean TD Error (moving average)']
            xscales = ['log'] * len(metrics)

            plot_instance.create_plot(agent=agent, exploration=exploration)
            plot_instance.set_plot_cfg(titles=titles, xlabels=xlabels, ylabels=ylabels, xscales=xscales)
            plot_instance.save_plot(f'results/{agent}_{exploration}.svg')
