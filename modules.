import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.ticker as ticker
from scipy.stats import iqr
from matplotlib import gridspec

def plot_waterfalls_and_profiles(data, start_phase, end_phase, fraction):
    POL_LABELS = ['I', 'Q', 'U', 'V']
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    pulse_phase = np.linspace(0, 1, data.shape[2])

    for i, label in enumerate(POL_LABELS):
        ax_waterfall = axs[0, i]
        ax_profile = axs[1, i]

        img = ax_waterfall.imshow(data[:, i, :], aspect='auto', origin='lower',
                                  extent=[0, 1, 0, data.shape[0]], cmap='magma')
        ax_waterfall.set_title(f'{label} vs Pulse Number and Phase')
        ax_waterfall.set_xlabel('Phase')
        ax_waterfall.set_ylabel('Pulse Number')
        ax_waterfall.set_xlim(start_phase, end_phase)

        mean_profile = data[:, i, :].mean(axis=0)
        ax_profile.plot(pulse_phase, mean_profile)
        ax_profile.set_title(f'Mean {label} Profile')
        ax_profile.set_xlabel('Phase')
        ax_profile.set_ylabel('Intensity')
        ax_profile.set_xlim(start_phase, end_phase)
        horizontal_y = fraction*np.max(mean_profile)
        ax_profile.axhline(y=horizontal_y, color='red', linestyle='--', linewidth=1)
    return fig

def plot_polarisation_parameters(data, start_phase, end_phase, fraction=0.1):
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
    absv_frac = np.where(I != 0, np.abs(V / I), 0)
    PA = 0.5 * np.arctan2(U, Q) * 180 / np.pi
    EA = 0.5 * np.arctan2(V, L_true) * 180 / np.pi

    # Quantities and labels to plot
    quantities = [PA, EA, p_frac, l_frac, v_frac, absv_frac]
    labels = [
        "PA [deg]",
        "EA [deg]",
        "P/I",
        "L/I",
        "V/I",
        "|V/I|"
    ]

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()

    for ax, quantity, label in zip(axs, quantities, labels):
        ax.plot(phase_axis, quantity, lw=1.5)
        ax.set_ylabel(label)
        ax.set_xlim(start_phase, end_phase)
        ax.grid(True)

    axs[-2].set_xlabel("Pulse Phase")
    axs[-1].set_xlabel("Pulse Phase")
    return fig

def plot_polarisation_histograms(data, start_phase, end_phase, fraction=0.1, base_quantity_bins=200):
    """
    Plots 2D histograms of PA, EA, I, P/I, L/I, V/I, and |V/I| vs phase over a user-defined phase range.
    """
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Extract Stokes parameters
    I = data[:, 0, :]
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]

    # On-pulse mask
    threshold = fraction * np.max(I.mean(axis=0))
    on_pulse_mask = I.mean(axis=0) >= threshold
    off_pulse_mask = ~on_pulse_mask
    off_pulse_std = np.std(I.mean(axis=0)[off_pulse_mask])

    # Derived quantities
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / off_pulse_std
    mask = L_sigma >= 1.57
    L_true[mask] = off_pulse_std * np.sqrt(L_sigma[mask]**2 - 1)

    p_frac = np.where(I != 0, np.sqrt(Q**2 + U**2 + V**2) / I, 0)
    l_frac = np.where(I != 0, L_true / I, 0)
    v_frac = np.where(I != 0, V / I, 0)
    absv_frac = np.where(I != 0, np.abs(V / I), 0)
    PA = 0.5 * np.arctan2(U, Q) * 180 / np.pi
    EA = 0.5 * np.arctan2(V, L_true) * 180 / np.pi

    quantities = [PA, EA, I, p_frac, l_frac, v_frac, absv_frac]
    labels = ["PA [deg]", "EA [deg]", "I", "P/I", "L/I", "V/I", "|V/I|"]

    # Determine selected phase range
    start_idx = np.searchsorted(phase_axis, start_phase)
    end_idx = np.searchsorted(phase_axis, end_phase)
    selected_phase_axis = phase_axis[start_idx:end_idx]
    selected_phase_bins = end_idx - start_idx

    # Dynamically adjust quantity bins
    quantity_bins = max(50, min(base_quantity_bins, selected_phase_bins))

    # Prepare subplot layout
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.2])
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(6)]
    axs.append(fig.add_subplot(gs[3, 0:2]))  # Full-width subplot for 7th plot

    for ax, quantity, label in zip(axs, quantities, labels):
        q = quantity.T[start_idx:end_idx]  # Shape: (selected_phase_bins, pulses)
        q_min, q_max = np.min(q), np.max(q)
        hist2d = np.zeros((quantity_bins, selected_phase_bins))

        for i in range(selected_phase_bins):
            hist, bin_edges = np.histogram(q[i], bins=quantity_bins, range=(q_min, q_max))
            hist2d[:, i] = hist

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        extent = [selected_phase_axis[0], selected_phase_axis[-1], bin_centers[0], bin_centers[-1]]

        im = ax.imshow(hist2d, aspect='auto', extent=extent, origin='lower', cmap='magma')
        ax.set_ylabel(label, fontsize=10)
        #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else '0'))
        ax.grid(False)

    axs[-2].set_xlabel("Pulse Phase")
    axs[-1].set_xlabel("Pulse Phase")
    return fig

def plot_phase_slice_histograms_by_phase(data, phase_values, fraction=0.1, default_bins=200):
    """
    Plots 1D histograms of polarization quantities at specified phase values,
    with dynamic bin count based on data spread.
    """
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Convert normalized phase values to closest phase bin indices
    phase_bins = [np.argmin(np.abs(phase_axis - val)) for val in phase_values]

    I = data[:, 0, :]
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]

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
    v_frac = np.where(I != 0, V / I, 0)
    absv_frac = np.where(I != 0, np.abs(V / I), 0)
    PA = 0.5 * np.arctan2(U, Q) * 180 / np.pi
    EA = 0.5 * np.arctan2(V, L_true) * 180 / np.pi

    quantities = [p_frac, l_frac, v_frac, absv_frac, PA, EA]
    quantity_names = ["P/I", "L/I", "V/I", "|V/I|", "PA [deg]", "EA [deg]"]

    fig, axs = plt.subplots(len(quantities), len(phase_bins), figsize=(15, 10), constrained_layout=True)

    for row_idx, (quantity, name) in enumerate(zip(quantities, quantity_names)):
        for col_idx, (phase_bin, phase_val) in enumerate(zip(phase_bins, phase_values)):
            ax = axs[row_idx, col_idx]
            values = quantity[:, phase_bin]

            # Dynamically compute bin count using Freedman–Diaconis rule
            val_iqr = iqr(values)
            if val_iqr > 0:
                bin_width = 2 * val_iqr / (len(values) ** (1/3))
                range_ = np.ptp(values)  # Peak-to-peak (max - min)
                bins = int(np.clip(range_ / bin_width, 20, 300))
            else:
                bins = default_bins  # Fallback if IQR is zero

            ax.hist(values, bins=bins, color='steelblue', alpha=0.8)
            ax.set_title(f"{name}\nPhase = {phase_val:.2f}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

    return fig


def plot_poincare_aitoff_from_data(data, segments_phase_ranges, fraction=0.1):
    """
    Plot polarization trajectories on the Poincaré sphere using Aitoff projection,
    colored by phase within the segment.
    """
    num_pulses, _, num_bins = data.shape
    phase_axis = np.linspace(0, 1, num_bins)

    I = data[:, 0, :].mean(axis=0)
    Q = data[:, 1, :].mean(axis=0)
    U = data[:, 2, :].mean(axis=0)
    V = data[:, 3, :].mean(axis=0)

    # Define on-pulse region
    threshold = fraction * np.max(I)
    on_pulse_mask = I >= threshold
    off_pulse_mask = ~on_pulse_mask
    sigma_off = np.std(I[off_pulse_mask])

    # Compute derived quantities
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / sigma_off
    mask = L_sigma >= 1.57
    L_true[mask] = sigma_off * np.sqrt(L_sigma[mask]**2 - 1)

    # Compute PA and EA (in radians)
    PA = 0.5 * np.arctan2(U, Q) 
    EA = 0.5 * np.arctan2(V, L_true)

    # Convert phase ranges to bin indices
    start_idx = np.searchsorted(phase_axis, segments_phase_ranges[0])
    end_idx = np.searchsorted(phase_axis, segments_phase_ranges[1])

    # Prepare lon, lat and colors
    lon = 2 * PA[start_idx:end_idx]
    lat = 2 * EA[start_idx:end_idx]
    lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi  # Wrap to [-π, π]
    colors = np.linspace(segments_phase_ranges[0], segments_phase_ranges[1], end_idx-start_idx)  # Phase gradient

    # Plot in Aitoff projection
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='aitoff')
    sc = ax.scatter(lon, lat, c=colors, cmap='magma', alpha=0.8, s=20)

    ax.grid(True)
    fig.colorbar(sc, ax=ax, orientation='horizontal', label='Pulse Phase')
    plt.tight_layout()
    return fig
    
def plot_interactive_poincare_sphere(data, segments_phase_ranges, fraction=0.1):
    """
    Create an interactive 3D Poincaré sphere plot using Plotly.

    Parameters:
    - data: numpy array of shape (num_pulses, 4, num_phase_bins)
    - segments_phase_ranges: tuple of (start_phase, end_phase) in [0, 1]
    - fraction: threshold to define on-pulse region from average I
    """
    num_pulses, _, num_bins = data.shape
    phase_axis = np.linspace(0, 1, num_bins)

    I = data[:, 0, :].mean(axis=0)
    Q = data[:, 1, :].mean(axis=0)
    U = data[:, 2, :].mean(axis=0)
    V = data[:, 3, :].mean(axis=0)

    threshold = fraction * np.max(I)
    off_pulse_mask = I < threshold
    sigma_off = np.std(I[off_pulse_mask])

    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / sigma_off
    mask = L_sigma >= 1.57
    L_true[mask] = sigma_off * np.sqrt(L_sigma[mask]**2 - 1)

    PA = 0.5 * np.arctan2(U, Q)
    EA = 0.5 * np.arctan2(V, L_true)

    lon = 2 * PA
    lat = 2 * EA
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    # Filter segment
    start_phase, end_phase = segments_phase_ranges
    start_idx = np.searchsorted(phase_axis, start_phase)
    end_idx = np.searchsorted(phase_axis, end_phase)

    x = x[start_idx:end_idx]
    y = y[start_idx:end_idx]
    z = z[start_idx:end_idx]

    # Create plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+lines',
        marker=dict(size=4, color=np.linspace(0, 1, len(x)), colorscale='magma'),
        line=dict(color='rgba(0,0,0,0.3)', width=2),
        name="Polarization Path"
    ))

    # Optional: add faint reference sphere
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.1, showscale=False, colorscale='Greys'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Q', range=[-1, 1]),
            yaxis=dict(title='U', range=[-1, 1]),
            zaxis=dict(title='V', range=[-1, 1]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
