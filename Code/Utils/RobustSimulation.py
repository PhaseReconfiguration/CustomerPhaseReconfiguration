import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class RobustSimulation:
    def __init__(self, feederbalancing, out_path, n_simulations=40) -> None:
        np.random.seed(15)
        self.feederbalancing = feederbalancing
        self.n_timesteps = feederbalancing.number_timesteps
        self.uncertainty_levels = np.array([0, 10, 20, 30, 50, 75]) / 100
        self.n_simulations = n_simulations
        self.input_path = out_path

        self.results = {u: {'before': None, 'after': []} for u in self.uncertainty_levels}
        self.P_variations = {u: [] for u in self.uncertainty_levels}
        self.fluctuations = {u: [] for u in self.uncertainty_levels}

    def run_robust_simulation(self):
        total_simulations = (len(self.uncertainty_levels) - 1) * (self.n_simulations + 1) + 2
        expected_time = 50  # seconds
        print(f"### Running {total_simulations} simulations. Estimated time: {total_simulations * expected_time / 60:.1f} mins ###")

        for u in self.uncertainty_levels:
            print(f"\nRunning BEFORE simulation for uncertainty={u*100:.0f}%")
            P = self.feederbalancing.change_P(self.feederbalancing.B_sol)  # Use optimized phase assignment
            _, results_before = self.feederbalancing.run_simulations(P, f"{self.input_path}/results_before_{u}.npy")
            self.results[u]['before'] = results_before
            self.P_variations[u].append(P)

            for s in range(self.n_simulations if u > 0 else 1):
                print(f"Running AFTER simulation for uncertainty={u*100:.0f}%, simulation={s+1}")
                P_modified = P.copy()

                for bus in self.feederbalancing.choosable_buses:
                    customer = self.feederbalancing.net.asymmetric_load.loc[self.feederbalancing.net.asymmetric_load['bus'] == bus]
                    ean = customer['ean'].values[0]
                    phases = customer['phase_load'].values[0]

                    fluctuation_factors = 1 + np.random.normal(0, u, size=(len(P), len(phases)))

                    for i, p in enumerate(phases):
                        P_modified[f'{ean}_{p}'] *= fluctuation_factors[:, i]
                        self.fluctuations[u].append(fluctuation_factors[:, i])
                    self.P_variations[u].append(P_modified)

                _, results_after = self.feederbalancing.run_simulations(P_modified, f"{self.input_path}/results_after_{u}_{s}.npy")
                self.results[u]['after'].append(results_after)

    def load_results(self, results_path):
        for u in self.uncertainty_levels:
            self.results[u]['before'] = np.load(f"{results_path}/results_before_{u}.npy", allow_pickle=True)
            n_after = self.n_simulations if u > 0 else 1
            for s in range(n_after):
                after = np.load(f"{results_path}/results_after_{u}_{s}.npy", allow_pickle=True)
                self.results[u]['after'].append(after)

    def calculate_confidence_intervals(self, metric='voltage', confidence=0.95):
        ci_data = []

        for u in self.uncertainty_levels:
            before = np.array(self.results[u]['before'])
            after = np.array(self.results[u]['after'])
            abs_deltas = []

            P_before = np.array(self.P_variations[u][0])

            for s in range(len(after)):  # for each simulation
                delta_per_timestep = []

                for t in range(self.n_timesteps):
                    timestep_deltas = []

                    for f in range(len(self.feederbalancing.feeders)):
                        b = np.array(before[f][t][metric]).flatten()
                        a = np.array(after[s][f][t][metric]).flatten()
                        deviations = np.abs(b - a)

                        timestep_deltas.append(np.mean(deviations))

                    delta_per_timestep.append(np.mean(timestep_deltas))

                abs_deltas.append(np.mean(delta_per_timestep))
                P_after = np.array(self.P_variations[u][1+s])
                # P_variation = 

            mean = np.mean(abs_deltas)
            ci = 0 if len(abs_deltas) < 2 else stats.sem(abs_deltas) * stats.t.ppf((1 + confidence) / 2., len(abs_deltas) - 1)

            ci_data.append({
                'Uncertainty (%)': u * 100,
                'Metric': metric,
                'Mean': mean,
                'CI_Lower': mean - ci,
                'CI_Upper': mean + ci
            })

        return pd.DataFrame(ci_data)

    def plot_results(self, save_path=None):
        all_metrics = ['voltage', 'losses', 'unbalance']
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(len(all_metrics), 1, figsize=(10, 6 * len(all_metrics)))

        for idx, metric in enumerate(all_metrics):
            results_df = self.calculate_confidence_intervals(metric=metric)

            ax = axes[idx] if len(all_metrics) > 1 else axes
            sns.lineplot(
                data=results_df,
                x='Uncertainty (%)',
                y='Mean',
                marker='o',
                markersize=8,
                ax=ax
            )

            ax.fill_between(
                results_df['Uncertainty (%)'],
                results_df['CI_Lower'],
                results_df['CI_Upper'],
                alpha=0.2
            )

            ax.set_title(f'Robustness Analysis: {metric.capitalize()}', fontsize=14)
            ax.set_xlabel('Uncertainty Level (%)', fontsize=12)
            ax.set_ylabel(f'{metric.capitalize()} deviation', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plots saved to {save_path}")
        else:
            plt.show()
