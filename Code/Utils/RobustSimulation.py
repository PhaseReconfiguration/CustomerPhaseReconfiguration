import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class RobustSimulation:

    def __init__(self, feederbalancing, input_path) -> None:
        self.feederbalancing = feederbalancing
        self.n_timesteps = feederbalancing.number_timesteps
        self.uncertainity_levels = np.array([0, 10, 20, 30, 50, 75]) / 100
        # self.uncertainity = self.uncertainity[:3]
        self.n_simulations = 2
        self.n_customer_to_choose = 10
        self.input_path = input_path

        # results = dictionary with "u" as key + "before" and "after" as keys with lists of results (x n_simulations). Results is [feeders, timesteps, {issue}, phases]
        self.results = {u: {'before': None, 'after': []} for u in self.uncertainity_levels}
        
    def run_robust_simulation(self):
        for u in self.uncertainity_levels:
            print(f"\nRunning simulation BEFORE for u={u}")
            P = self.feederbalancing.change_P(self.feederbalancing.B_sol)
            _, results_before = self.feederbalancing.run_simulations(P, self.input_path+f'/Robust/results_before_{u}.npy')

            self.results[u]['before'] = results_before
            for s in range(self.n_simulations):
                print(f"Running simulation AFTER for u={u}, s={s}")
                selected_buses = np.random.choice(self.feederbalancing.choosable_buses, self.n_customer_to_choose, replace=False)
                #Change P for the selected customers of u
                for bus in selected_buses:
                    customer = self.feederbalancing.net.asymmetric_load.loc[self.feederbalancing.net.asymmetric_load['bus']==bus]
                    ean = customer['ean'].values[0]
                    phases = customer['phase_load'].values[0]
                    
                    signs = np.where(np.random.random(self.n_timesteps) < 0.5, 1, -1)
                    multipliers = np.array([self.feederbalancing.get_phase_splitting_values(len(phases)) for t in range(self.n_timesteps)])
                    for i,p in enumerate(phases):
                        fluctuation = signs * u * multipliers[:, i]
                        P[f'{ean}_{p}'] += fluctuation

                _, results_after = self.feederbalancing.run_simulations(P, self.input_path+f'/Robust/results_after_{u}_{s}.npy')
                self.results[u]['after'].append(results_after)
                if(u == 0):
                    break

    def load_results(self, results_path):
        """Load simulation results from files"""
        for u in self.uncertainity_levels:
            before = np.load(f"{results_path}/results_before_{u}.npy")
            self.results[u]['before'] = before
            for s in range(self.n_simulations):
                # Load your results here
                after = np.load(f"{results_path}/results_after_{u}_{s}.npy")
                self.results[u]['after'].append(after)
                if(u == 0):
                    break

    def calculate_confidence_intervals(self, confidence=0.95):
        """Calculate mean and confidence intervals"""
        ci_data = []
        metrics = ['voltage', 'current', 'losses']
        metric_to_consider = 0
        for u in self.uncertainity_levels:
            before = np.array(self.results[u]['before'])
            after = np.array(self.results[u]['after'])

            abs_deltas = []
            rel_deltas = []
            for s in range(self.n_simulations):
                abs_delta = []
                rel_delta = []
                for t in range(len(self.feederbalancing.timesteps)):
                    bes = []
                    afs = []
                    for f in range(len(self.feederbalancing.feeders)): #Merge feeders
                        bes.extend(np.array(before[f][t][metrics[metric_to_consider]]).flatten())
                        afs.extend(np.array(after[s][f][t][metrics[metric_to_consider]]).flatten())
                    abs_delta.append(np.mean( np.abs(bes - afs) ))
                    rel_delta.append(np.mean( np.abs(bes - afs) / (np.abs(bes) + 1e-10)))
                abs_deltas.append(np.sum(abs_delta))
                rel_deltas.append(np.sum(rel_delta) * 100)
                if(u == 0):
                    break
        
            for deltas, label in zip([abs_deltas, rel_deltas], ['absolute', 'relative']):
                mean = np.mean(deltas)
                if(u == 0):
                    ci = 0
                else:
                    sem = stats.sem(deltas)
                    ci = sem * stats.t.ppf((1 + confidence)/2., len(deltas)-1)
                
                ci_data.append({
                    'Uncertainty': u*100,
                    'Metric': metrics[metric_to_consider],
                    'ErrorType': label,
                    'Mean': mean,
                    'CI_Lower': mean - ci,
                    'CI_Upper': mean + ci
                })
        
        return pd.DataFrame(ci_data)

    def plot_results(self):
        """Plot with confidence intervals"""
        results_df = self.calculate_confidence_intervals()

        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create plot
        ax = sns.lineplot(
            data=results_df,
            x='Uncertainty',
            y='Mean',
            hue='ErrorType',
            style='ErrorType',
            markers=True,
            dashes=False,
            markersize=10,
            err_style='band',
            errorbar='sd'
        )

        for error_type in results_df['ErrorType'].unique():
            subset = results_df[results_df['ErrorType'] == error_type]
            plt.fill_between(
                subset['Uncertainty'],
                subset['CI_Lower'],
                subset['CI_Upper'],
                alpha=0.2
            )
        
        # Customize plot
        plt.title('Robustness Analysis with Confidence Intervals', pad=20, fontsize=14)
        plt.xlabel('Uncertainty Level (%)', labelpad=10)
        plt.ylabel('Deviation (Absolute)', labelpad=10)
        # plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Add confidence interval shading
        for metric in results_df['Metric'].unique():
            subset = results_df[results_df['Metric'] == metric]
            plt.fill_between(
                subset['Uncertainty'],
                subset['CI_Lower'],
                subset['CI_Upper'],
                alpha=0.2
            )
        
        plt.legend(title='Metric', loc='upper left')
        plt.tight_layout()
        plt.show()