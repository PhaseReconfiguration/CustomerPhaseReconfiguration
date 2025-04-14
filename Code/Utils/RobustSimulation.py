import numpy as np

class RobustSimulation:

    def __init__(self, feederbalancing, input_path) -> None:
        self.feederbalancing = feederbalancing
        self.n_timesteps = feederbalancing.number_timesteps
        self.uncertainity = np.array([10, 20, 30, 50, 75]) / 100
        self.uncertainity = np.array([10, 20]) / 100
        self.n_simulations = 2
        self.n_customer_to_choose = 10
        self.input_path = input_path

        self.results = {u: {'before': [], 'after': []} for u in self.uncertainity}
        
    def run_robust_simulation(self):
        for u in self.uncertainity:
            for s in range(self.n_simulations):
                print(f"Running simulation for u={u}, s={s}")
                selected_buses = np.random.choice(self.feederbalancing.choosable_buses, self.n_customer_to_choose, replace=False)

                P = self.feederbalancing.change_P(self.feederbalancing.B_sol)
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

                
                _, results_before = self.feederbalancing.run_simulations(P, self.input_path+f'/Robust/results_before_{u}_{s}.npy')
                _, results_after = self.feederbalancing.run_simulations(P, self.input_path+f'/Robust/results_after_{u}_{s}.npy')
                self.results[u]['before'].append(results_before)
                self.results[u]['after'].append(results_after)

    def load_results(self, results_path):
        """Load simulation results from files"""
        for u in self.uncertainty_levels:
            for s in range(self.n_simulations):
                # Load your results here
                before = np.load(f"{results_path}/results_before_{u}_{s}.npy")
                after = np.load(f"{results_path}/results_after_{u}_{s}.npy")
                self.results[u]['before'].append(before)
                self.results[u]['after'].append(after)

    def calculate_confidence_intervals(self, confidence=0.95):
        """Calculate mean and confidence intervals"""
        ci_data = []
        for u in self.uncertainity:
            before = np.array(self.results[u]['before'])
            after = np.array(self.results[u]['after'])
            
            # Calculate improvement metrics
            improvement = before - after  # Absolute improvement
            
            # Statistics for each uncertainty level
            for metric_idx, metric_name in enumerate(['Voltage', 'Current', 'Losses']):
                metric_improvement = improvement[:, :, metric_idx].flatten()
                
                mean = np.mean(metric_improvement)
                sem = stats.sem(metric_improvement)
                ci = sem * stats.t.ppf((1 + confidence)/2., len(metric_improvement)-1)
                
                ci_data.append({
                    'Uncertainty': u*100,
                    'Metric': metric_name,
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
            hue='Metric',
            style='Metric',
            markers=True,
            dashes=False,
            markersize=10,
            err_style='band',
            ci='sd'  # Show standard deviation
        )
        
        # Customize plot
        plt.title('Robustness Analysis with Confidence Intervals', pad=20, fontsize=14)
        plt.xlabel('Uncertainty Level (%)', labelpad=10)
        plt.ylabel('Improvement (Absolute)', labelpad=10)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Add confidence interval shading
        for metric in results_df['Metric'].unique():
            subset = results_df[results_df['Metric'] == metric]
            plt.fill_between(
                subset['Uncertainty'],
                subset['CI_Lower'],
                subset['CI_Upper'],
                alpha=0.2
            )
        
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()