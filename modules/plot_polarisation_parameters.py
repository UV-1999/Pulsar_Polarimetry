import numpy as np
import matplotlib.pyplot as plt

def plot_polarization_parameters(data, fraction=0.1, quantity_bins=200):
    """
    Plots PA, EA, p, L/I, V/I, and |V|/I vs phase (integrated over pulses)
    in a 3x2 subplot layout.
    """
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Extract Stokes parameters (averaged over pulses)
    I = data[:, 0, :].mean(axis=0)
    Q = data[:, 1, :].mean(axis=0)
    U = data[:, 2, :].mean(axis=0)
    V = data[:, 3, :].mean(axis=0)

    # Define on-pulse and off-pulse regions
    threshold = fraction * np.max(I)
    on_pulse_mask = I >= threshold
    off_pulse_mask = ~on_pulse_mask

    # Calculate off-pulse standard deviation for bias correction
    off_pulse_std = np.std(I[off_pulse_mask])

    # Derived quantities
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / off_pulse_std
    mask = L_sigma >= 1.57
    L_true[mask] = off_pulse_std * np.sqrt(L_sigma[mask]**2 - 1)

    p_frac = np.where(I != 0, np.sqrt(Q**2 + U**2 + V**2) / I, 0)
    l_frac = np.where(I != 0, L_true / I, 0)
    v_frac = np.where(I != 0, V / I, 0)
    absv_frac = np.where(I != 0, np.abs(V) / I, 0)
    PA = 0.5 * np.arctan2(U, Q)
    EA = 0.5 * np.arctan2(V, L_true)

    # Quantities and labels to plot
    quantities = [PA, EA, p_frac, l_frac, v_frac, absv_frac]
    labels = [
        "Polarization Angle (PA) [rad]",
        "Ellipticity Angle (EA) [rad]",
        "Polarized Fraction (p)",
        "Bias-Corrected Linear Polarization Fraction (L/I)",
        "Circular Polarization Fraction (V/I)",
        "Absolute Circular Polarization Fraction (|V|/I)"
    ]

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axs = axs.flatten()

    for ax, quantity, label in zip(axs, quantities, labels):
        ax.plot(phase_axis, quantity, lw=1.5)
        ax.set_ylabel(label)
        ax.grid(True)

    axs[-2].set_xlabel("Pulse Phase")
    axs[-1].set_xlabel("Pulse Phase")

    fig.suptitle("Polarization Quantities vs Pulse Phase (Integrated over Pulses)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
