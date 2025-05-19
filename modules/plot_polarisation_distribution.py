
def plot_polarization_histograms(data, fraction=0.1, quantity_bins=200):
    """
    Plots 2D histograms of PA, EA, I, P/I, L/I, and V/I vs phase.
    Each subplot shows intensity of occurrence of quantity vs phase.
    """
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Extract Stokes parameters (intergrated over all pulse, intensity as a function of phase)
    I = data[:, 0, :].mean(axis=0)
    Q = data[:, 1, :].mean(axis=0)
    U = data[:, 2, :].mean(axis=0)
    V = data[:, 3, :].mean(axis=0)
  
    # Define on-pulse as where intensity >= fraction * max
    threshold = fraction * np.max(I)
    on_pulse_mask = I >= threshold
    off_pulse_mask = ~on_pulse_mask
    on_pulse_indices = np.where(on_pulse_mask)[0]
    off_pulse_indices = np.where(off_pulse_mask)[0]

    # Calculate off-pulse standard deviation
    off_pulse_std = np.std(intensity[off_pulse_mask])

    # Derived quantities
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / sigma_off
    mask = L_sigma >= 1.57
    L_true[mask] = sigma_off * np.sqrt(L_sigma[mask]**2 - 1)
    p_frac = np.where(I != 0, np.sqrt(Q**2 + U**2 + V**2) / I, 0)
    l_frac = np.where(I != 0, L_true / I, 0)
    v_frac = np.where(I != 0, V / I, 0)
    absv_frac = np.where(I != 0, np.abs(V) / I, 0)
    PA = 0.5 * np.arctan2(U, Q)
    EA = 0.5 * np.arctan2(V, L_true)

    quantities = [PA, EA, I, p_frac, l_frac, v_frac, absv_frac]
    labels = [
        "Polarization Angle (PA) [rad]",
        "Bias Corrected Ellipticity Angle (EA) [rad]",
        "Total Intensity (I)", "polarised fraction (p)"
        "Bias Corrected Linear Polarization Fraction (L/I)",
        "Circular Polarization Fraction (V/I), Absolute Circular Polarization Fraction (|V|/I)"
    ]

    fig, axs = plt.subplots(4, 2, sharex=True, constrained_layout=True)

    for ax, quantity, label in zip(axs, quantities, labels):
        # Shape: (phase_bin, samples)
        q = quantity.T

        # Set common bin range across all phase bins for uniformity
        q_min, q_max = np.min(q), np.max(q)
        hist2d = np.zeros((quantity_bins, num_phase_bins))

        for i in range(num_phase_bins):
            hist, bin_edges = np.histogram(q[i], bins=quantity_bins, range=(q_min, q_max))
            hist2d[:, i] = hist

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        extent = [0, 1, bin_centers[0], bin_centers[-1]]

        im = ax.imshow(hist2d, aspect='auto', extent=extent, origin='lower', cmap='magma')
        ax.set_ylabel(label)
        ax.grid(False)

    axs[-1].set_xlabel("Pulse Phase (0 to 1)")
    fig.suptitle("2D Histograms of Polarization Quantities vs Phase")
