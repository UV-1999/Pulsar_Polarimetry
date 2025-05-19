import numpy as np
import matplotlib.pyplot as plt

def plot_polarisation_parameters(data, fraction=0.1):
    num_pulses, _, num_bins = data.shape

    I = data[:, 0, :]
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]

    I_mean = np.mean(I, axis=0)

    # Define on-pulse region
    threshold = fraction * np.max(I_mean)
    on_pulse_mask = I_mean >= threshold
    on_pulse_indices = np.where(on_pulse_mask)[0]

    # Divide on-pulse region into thirds
    n = len(on_pulse_indices)
    if n < 3:
        raise ValueError("On-pulse region too small to divide into thirds.")

    left_bin = on_pulse_indices[n // 6]
    center_bin = on_pulse_indices[n // 2]
    right_bin = on_pulse_indices[5 * n // 6]

    selected_bins = {
        "Left": left_bin,
        "Center": center_bin,
        "Right": right_bin
    }

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Polarisation Distributions at Selected Pulse Longitudes")

    for i, (label, bin_idx) in enumerate(selected_bins.items()):
        I_bin = I[:, bin_idx]
        Q_bin = Q[:, bin_idx]
        U_bin = U[:, bin_idx]
        V_bin = V[:, bin_idx]

        L_bin = np.sqrt(Q_bin**2 + U_bin**2)
        p_bin = np.sqrt(Q_bin**2 + U_bin**2 + V_bin**2) / I_bin
        V_abs_bin = np.abs(V_bin)

        # Clean up division by zero or invalid values
        p_bin[~np.isfinite(p_bin)] = 0

        axs[i, 0].hist(p_bin, bins=50, color='blue', alpha=0.7)
        axs[i, 0].set_title(f"{label} - Total p")
        axs[i, 0].set_xlabel("p")
        
        axs[i, 1].hist(L_bin, bins=50, color='orange', alpha=0.7)
        axs[i, 1].set_title(f"{label} - Linear L")
        axs[i, 1].set_xlabel("L")
        
        axs[i, 2].hist(V_abs_bin, bins=50, color='green', alpha=0.7)
        axs[i, 2].set_title(f"{label} - Circular |V|")
        axs[i, 2].set_xlabel("|V|")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
