import numpy as np
import matplotlib.pyplot as plt

def plot_phase_slice_histograms_by_phase(data, phase_values, fraction=0.1, bins=200):
    """
    Plots 1D histograms of polarization quantities (P/I, L/I, |V|/I, PA, EA)
    at specified phase values (normalized between 0 and 1).

    Parameters:
    - data: ndarray of shape (num_pulses, 4, num_phase_bins)
    - phase_values: list of 3 float values in [0, 1] — normalized pulse phase
    - fraction: threshold to define on-pulse region
    - bins: number of bins for histograms
    """
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Convert normalized phase values to closest phase bin indices
    phase_bins = [np.argmin(np.abs(phase_axis - val)) for val in phase_values]

    I = data[:, 0, :]
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]

    # Estimate off-pulse standard deviation from mean profile
    mean_I = I.mean(axis=0)
    threshold = fraction * np.max(mean_I)
    off_pulse_mask = mean_I < threshold
    off_pulse_std = np.std(I[:, off_pulse_mask])

    # Derived quantities
    L = np.sqrt(Q**2 + U**2)
    L_sigma = L / off_pulse_std
    L_true = np.zeros_like(L)
    mask = L_sigma >= 1.57
    L_true[mask] = off_pulse_std * np.sqrt(L_sigma[mask]**2 - 1)

    P = np.sqrt(Q**2 + U**2 + V**2)
    p_frac = np.where(I != 0, P / I, 0)
    l_frac = np.where(I != 0, L_true / I, 0)
    absv_frac = np.where(I != 0, np.abs(V) / I, 0)
    PA = 0.5 * np.arctan2(U, Q)
    EA = 0.5 * np.arctan2(V, L_true)

    quantities = [p_frac, l_frac, absv_frac, PA, EA]
    quantity_names = [
        "Total Polarization Fraction (P/I)",
        "Bias Corrected Linear Polarization Fraction (L/I)",
        "Absolute Circular Polarization Fraction (|V|/I)",
        "Polarization Angle (PA) [rad]",
        "Ellipticity Angle (EA) [rad]"
    ]

    fig, axs = plt.subplots(len(quantities), len(phase_bins), figsize=(15, 10), constrained_layout=True)
    
    for row_idx, (quantity, name) in enumerate(zip(quantities, quantity_names)):
        for col_idx, (phase_bin, phase_val) in enumerate(zip(phase_bins, phase_values)):
            ax = axs[row_idx, col_idx]
            values = quantity[:, phase_bin]
            ax.hist(values, bins=bins, color='steelblue', alpha=0.8)
            ax.set_title(f"{name}\nPhase = {phase_val:.2f}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
    
    fig.suptitle("1D Histograms of Polarization Quantities at Selected Phases", fontsize=16)
    return fig
