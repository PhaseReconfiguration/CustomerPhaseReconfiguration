import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class RobustSimulation:
    def __init__(self, feederbalancing, input_path, n_simulations=40, n_customer_to_choose=10) -> None:
        self.feederbalancing = feederbalancing
        self.n_timesteps = feederbalancing.number_timesteps
        self.uncertainty_levels = np.array([0, 10, 20, 30, 50, 75]) / 100
        self.n_simulations = n_simulations
        self.n_customer_to_choose = n_customer_to_choose
        self.input_path = input_path

        self.results = {u: {'before': None, 'after': []} for u in self.uncertainty_levels}
        self.fluctuations = {u:[] for u in self.uncertainty_levels}

    def run_robust_simulation(self, seed=42):
        total_simulations = (len(self.uncertainty_levels)+1) * self.n_simulations
        expected_time = 50 #seconds
        print(f"### Running a total of {total_simulations} simulations. Expected time: {total_simulations * expected_time} seconds ({total_simulations * expected_time / 60} mins) ###")
        np.random.seed(seed)
        for u in self.uncertainty_levels:
            print(f"\nRunning simulation BEFORE for uncertainty={u*100}%")
            P = self.feederbalancing.change_P(self.feederbalancing.B_sol)
            _, results_before = self.feederbalancing.run_simulations(P, f"{self.input_path}/Robust/results_before_{u}.npy")
            self.results[u]['before'] = results_before

            for s in range(self.n_simulations if u > 0 else 1):
                print(f"Running simulation AFTER for uncertainty={u*100}%, simulation={s}")
                P_modified = P.copy()
                selected_buses = np.random.choice(self.feederbalancing.choosable_buses, self.n_customer_to_choose, replace=False)

                for bus in selected_buses:
                    customer = self.feederbalancing.net.asymmetric_load.loc[self.feederbalancing.net.asymmetric_load['bus']==bus]

                    ean = customer['ean'].values[0]
                    phases = customer['phase_load'].values[0]

                    signs = np.where(np.random.rand(len(P)) < 0.5, 1, -1)
                    multipliers = np.array([self.feederbalancing.get_phase_splitting_values(len(phases)) for _ in range(len(P))])

                    for i, p in enumerate(phases):
                        fluctuation = signs * u * multipliers[:, i] * P[f'{ean}_{p}']
                        self.fluctuations[u].append(fluctuation)
                        P_modified[f'{ean}_{p}'] += fluctuation

                _, results_after = self.feederbalancing.run_simulations(P_modified, f"{self.input_path}/Robust/results_after_{u}_{s}.npy")
                self.results[u]['after'].append(results_after)

    def load_results(self, results_path):
        for u in self.uncertainty_levels:
            self.results[u]['before'] = np.load(f"{results_path}/results_before_{u}.npy", allow_pickle=True)
            n_after = self.n_simulations if u > 0 else 1
            for s in range(n_after):
                after = np.load(f"{results_path}/results_after_{u}_{s}.npy", allow_pickle=True)
                self.results[u]['after'].append(after)

    def calculate_confidence_intervals(self, confidence=0.95):
        ci_data = []
        metric = 'voltage' #only voltage is used here

        for u in self.uncertainty_levels:
            before = np.array(self.results[u]['before'])
            after = np.array(self.results[u]['after'])
            abs_deltas = []

            for s in range(len(after)): # for number simulations
                abs_total = []
                for f in range(len(self.feederbalancing.feeders)):
                    for t in range(self.n_timesteps):
                        b = np.array(before[f][t][metric]).flatten()
                        a = np.array(after[s][f][t][metric]).flatten()
                        deviations = np.abs(b - a)
                        # filtered_devs = deviations[deviations > 0.001]
                        # print(f"Min Deviation: {np.min(deviations)}, Max Deviation: {np.max(deviations)}, Mean Deviation: {np.mean(deviations)}, Std Deviation: {np.std(deviations)}, CI: {stats.sem(deviations)}, CI Lower: {stats.t.ppf((1 + confidence)/2., len(deviations)-1) * stats.sem(deviations)}, CI Upper: {stats.t.ppf((1 + confidence)/2., len(deviations)-1) * stats.sem(deviations)}, 95 Percentile: {np.percentile(deviations, 95)}")
                        abs_total.append(np.sum(deviations))
                abs_deltas.append(np.mean(abs_total))

            mean = np.mean(abs_deltas)
            ci = 0 if len(abs_deltas) < 2 else stats.sem(abs_deltas) * stats.t.ppf((1 + confidence)/2., len(abs_deltas)-1)

            ci_data.append({
                'Uncertainty': u * 100,
                'Metric': metric,
                'ErrorType': 'absolute',
                'Mean': mean * 100,
                'CI_Lower': mean - ci,
                'CI_Upper': mean + ci
            })

        return pd.DataFrame(ci_data)

    def plot_results(self, save_path=None):
        results_df = self.calculate_confidence_intervals()
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        ax = sns.lineplot(
            data=results_df,
            x='Uncertainty',
            y='Mean',
            hue='ErrorType',
            style='ErrorType',
            markers=True,
            dashes=False,
            markersize=10
        )

        subset = results_df[(results_df['ErrorType'] == 'absolute')]
        if not subset.empty:
            plt.fill_between(
                subset['Uncertainty'],
                subset['CI_Lower'],
                subset['CI_Upper'],
            alpha=0.2
        )

        plt.title('Robustness Analysis with Confidence Intervals', fontsize=14)
        plt.xlabel('Uncertainty Level (%)')
        plt.ylabel('Voltage Deviation')
        # plt.legend(title='Error Type')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return results_df