import numpy as np
import matplotlib.pyplot as plt

def plot_polarization_parameters(data, fraction=0.1, quantity_bins=200):
    """
    Plots PA, EA, I, P/I, L/I, and V/I vs phase (integrated over pulses)
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
    off_pulse_std = np.std(I[off_pulse_mask])

    # Derived quantities
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L /  off_pulse_std
    mask = L_sigma >= 1.57
    L_true[mask] =  off_pulse_std * np.sqrt(L_sigma[mask]**2 - 1)
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
        "Total Intensity (I)", "polarised fraction (p)",
        "Bias Corrected Linear Polarization Fraction (L/I)",
        "Circular Polarization Fraction (V/I), Absolute Circular Polarization Fraction (|V|/I)"
    ]

    
