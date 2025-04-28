import colorsys
import matplotlib.pyplot as plt
import numpy as np

figsize = (12,6)
fontsize = 18
legend_fontsize = 14

def hex_to_rgb(hex_color):
    """Convert hex color to RGB values in the range [0, 1]."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB values in the range [0, 1] back to hex color."""
    return '#' + ''.join(f'{int(c * 255):02x}' for c in rgb)

def scale_lightness(hex_color, scale_l):
    """Scale the lightness of a color given in hex."""
    # Convert hex to RGB
    rgb = hex_to_rgb(hex_color)
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Scale lightness and convert back to RGB
    rgb_scaled = colorsys.hls_to_rgb(h, min(1, l * scale_l), s)
    # Convert RGB back to hex
    return rgb_to_hex(rgb_scaled)

def plot_P_by_feeder(B, feederbalancing, timesteps, feeder_colors, meaningful_days=None, clips=None, save_paths=None):
    Af = {}
    phase_labels = feederbalancing.avalilable_phases  # Assuming phase labels are meaningful
    feeder_labels = [f"Feeder {f+1}" for f in range(len(feederbalancing.feeders))]
    P = feederbalancing.change_P(B)
    P = P.loc[feederbalancing.timesteps]

    # Plot Phase Loads by Feeder
    plt.figure(figsize=figsize)
    for f in range(len(feederbalancing.feeders)):
        eans = feederbalancing.net.asymmetric_load.loc[
            feederbalancing.net.asymmetric_load['feeder'] == f, 'ean'
        ]
        A = np.array([sum(P.loc[timesteps, f'{ean}_{p}'] for ean in eans) for p in feederbalancing.avalilable_phases])

        # Plot each phase's load
        for p, phase in enumerate(phase_labels):
            plt.plot(A[p], label=f"{feeder_labels[f]}, Phase {phase}", linestyle='-' if f % 2 == 0 else '--')
        Af[f] = A


    label = "Timestep"
    if(meaningful_days):
        ticks = np.arange(12*4, len(meaningful_days)*24*4, 24*4)
        plt.xticks(ticks, meaningful_days, fontsize=fontsize)
        label = "Day of Year (DOY)"
    plt.xlabel(label, fontsize=fontsize)

    plt.ylabel("Load [kW]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title("Phase Load", fontsize=fontsize)
    
    if(clips):
        plt.ylim(clips[0])

    plt.legend(fontsize=legend_fontsize)
    if(save_paths):
        plt.savefig(save_paths[0]+f'/load_{save_paths[1]}.png')
    plt.show()

    # Plot Loss by Feeder
    plt.figure(figsize=figsize)
    for f in range(len(feederbalancing.feeders)):
        A = Af[f]
        KPI = np.mean(A, axis=0)
        loss = np.abs(A - KPI).sum(axis=0)

        plt.plot(loss, label=feeder_labels[f], c=feeder_colors[f])

    label = "Timestep"
    if(meaningful_days):
        ticks = np.arange(12*4, len(meaningful_days)*24*4, 24*4)
        plt.xticks(ticks, meaningful_days, fontsize=fontsize)
        label = "Day of Year (DOY)"
    plt.xlabel(label, fontsize=fontsize)
    plt.ylabel("Unbalance [kW]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title("Feeders Unbalance", fontsize=fontsize)
    
    if(clips):
        plt.ylim(clips[1])

    plt.legend(fontsize=legend_fontsize)
    if(save_paths):
        plt.savefig(save_paths[0]+f'/unbalance_{save_paths[1]}.png')
    plt.show()

    return Af, P

def plot_feeder_unbalance(feederbalancing, A_init, A_sol, feeder_colors, feeder_colors_after, meaningful_days=None, clip=None, save_path=None):
    for f in range(len(feederbalancing.feeders)):
        # Calculate unbalances
        x_init = np.abs(A_init[f] - np.mean(A_init[f], axis=0)).sum(axis=0)
        x_sol = np.abs(A_sol[f] - np.mean(A_sol[f], axis=0)).sum(axis=0)

        # Print summary statistics
        total_init = np.sum(x_init)
        total_sol = np.sum(x_sol)
        reduction_percentage = (total_init - total_sol) / total_init * 100
        print(f"Feeder {f+1}: Initial = {total_init}, Solution = {total_sol}, Reduction = {reduction_percentage:.2f}%")

        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(x_init, '-', color=feeder_colors[f], label=f'Feeder {f+1} (Initial)')
        plt.plot(x_sol, '--', color=feeder_colors_after[f], label=f'Feeder {f+1} (Solution)')

        label = "Timestep"
        if(meaningful_days):
            ticks = np.arange(12*4, len(meaningful_days)*24*4, 24*4)
            plt.xticks(ticks, meaningful_days, fontsize=fontsize)
            label = "Day of Year (DOY)"
        plt.xlabel(label, fontsize=fontsize)

        plt.ylabel('Unbalance [kW]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('Feeder Phase Unbalance Comparison', fontsize=fontsize)
        
        if(clip):
            plt.ylim(clip[f])

        plt.legend(fontsize=legend_fontsize)
        if(save_path):
            plt.savefig(save_path+f'/delta_feeder{f+1}.png')
        plt.show()

def plot_PF_results(feederbalancing, results, meaningful_days=None, clip=None, save_paths=None):
    # issues_to_consider = ['voltage', 'load_line', 'loss_line', 'load_trafo', 'loss_trafo']
    number_issues = [0,0]
    for i, issue in enumerate(['voltage', 'loss_line']):
        fig1, ax1 = plt.subplots(figsize=figsize)

        for f in range(len(feederbalancing.feeders)):
            linestyle = '-' if f % 2 == 0 else '--'  # Alternate line styles for feeders
            v = [[], [], []]  # To store results for each phase

            for t in range(len(feederbalancing.timesteps)):
                for p in range(3):  # Loop through the 3 phases
                    if issue == 'voltage':
                        node_voltages = np.mean(results[f][t][issue][p])
                        v[p].append(node_voltages)  # Voltage case
                        number_issues[0] += np.sum((node_voltages>1.05) | (node_voltages<0.95))
                    elif issue == 'loss_line':
                        line_losses = np.array(results[f][t][issue][p])
                        v[p].append(np.sum(np.abs(line_losses)))  # Line loss case
                        number_issues[1] += np.sum((line_losses>100))

            # Plot for each phase
            for p in range(3):
                ax1.plot(v[p], label=f'Feeder {f+1}, Phase {p+1}', linestyle=linestyle)  # Removed fontsize parameter

            # Print statistics
            total_sum = np.sum(v)
            total_sum_per_phase = np.sum(v, axis=1)
            print(f"Feeder {f}: Total = {total_sum*1000} (kWh), Per Phase = {total_sum_per_phase*1000} (kWh). Number of issues: {number_issues[i]} ({issue})")
            print(f"Max Values: {np.max(v[0])}, {np.max(v[1])}, {np.max(v[2])}")
            print(f"Min Values: {np.min(v[0])}, {np.min(v[1])}, {np.min(v[2])}")
            print()

        # Add legends, labels, and titles
        label = "Timestep"
        if(meaningful_days):
            ticks = np.arange(12*4, len(meaningful_days)*24*4, 24*4)
            plt.xticks(ticks, meaningful_days)
            label = "Day of Year (DOY)"
        plt.xlabel(label, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        ax1.set_title(f"{issue.capitalize()} by Phase and Feeder", fontsize=fontsize)
        ax1.legend(loc=1, fontsize=legend_fontsize)

        if(clip and issue == 'voltage'):
            ax1.set_ylabel(f"{issue.capitalize()} (p.u.)", fontsize=fontsize)
            plt.ylim(clip)
            plt.axhline(y=1.05, color='r', linestyle='--')
            plt.axhline(y=0.95, color='r', linestyle='--')
        elif(clip and issue == 'loss_line'):
            ax1.set_ylabel(f"{issue.capitalize()} (kWh)", fontsize=fontsize)

        plt.yticks(fontsize=fontsize)

        if(save_paths):
            plt.savefig(save_paths[0]+f'/{issue}_{save_paths[1]}.png')
        plt.show()
