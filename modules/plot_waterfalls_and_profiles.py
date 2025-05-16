import numpy as np
import matplotlib.pyplot as plt

def plot_waterfalls_and_profiles(data, start_phase=0.33, end_phase=0.66):
    POL_LABELS = ['I', 'Q', 'U', 'V']
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    pulse_phase = np.linspace(0, 1, data.shape[2])

    for i, label in enumerate(POL_LABELS):
        ax_waterfall = axs[0, i]
        ax_profile = axs[1, i]

        img = ax_waterfall.imshow(data[:, i, :], aspect='auto', origin='lower',
                                  extent=[0, 1, 0, data.shape[0]], cmap='viridis')
        ax_waterfall.set_title(f'{label} vs Pulse Number and Phase')
        ax_waterfall.set_xlabel('Phase')
        ax_waterfall.set_ylabel('Pulse Number')
        ax_waterfall.set_xlim(start_phase, end_phase)

        mean_profile = data[:, i, :].mean(axis=0)
        ax_profile.plot(pulse_phase, mean_profile, color='black')
        ax_profile.set_title(f'Mean {label} Profile')
        ax_profile.set_xlabel('Phase')
        ax_profile.set_ylabel('Intensity')
        ax_profile.set_xlim(start_phase, end_phase)

    plt.suptitle("Pulse Stack and Mean Profiles", fontsize=16)
    return fig
