import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

today = datetime.today().strftime("%B %d, %Y")

# --- Importing plot functions ---
from modules.plots import *

# --- Page configuration ---
st.set_page_config(page_title="Pulsar Polarimeter", layout="wide")

# --- Title ---
st.title("Pulsar Polarimeter")

# --- Introduction with links ---
st.markdown("""
<h4>
A data-analysis tool by for visualizing and exploring single-pulse polarimetry data from the  
<a href="https://psrweb.jb.man.ac.uk/meertime/singlepulse/" target="_blank">MeerTime Single Pulse Database</a>.
</h4>
""", unsafe_allow_html=True)


# --- About section ---
with st.expander("About this App", expanded=True):
    st.markdown(f"""
    This tool lets you upload pulsar single-pulse data in `.npy` or `.npz` format (as available from the MeerTime database),
    and generates a series of interactive visualizations to explore the polarization state of the pulsar signal.
    
    **Features included as on {today}:**
    - Waterfall plots and integrated profiles of each Stokes parameter.
    - Polarization parameter vs pulse phase intergrated over all pulses.
    - 2D Histograms of polarization parameters with 1D Histograms for specific phases.
    - Trajectories of polarization state on the Poincaré sphere (Aitoff projection and 3D) in specific phase region.
    - Linear polarisation parameter is corrected for bias assuming on-pulse window to be full width at tenth (default) of maximum of the integrated profile.
    - On-pulse window can be changed after plotting the intergrated profile.

    **Abbreviations and symbols used:**
    - I, Q, U, V are the four Stokes parameters.
    - PA is polarisation angle, EA is ellipticity angle.
    - L is linear polarisation parameter and P is total polarisation parameter.
    - lowercase l and p are fractional polarisation parameters for linear and total respectively.
    
    **Need help or have any suggestions?** Reach out to <a href="https://uv-1999.github.io/" target="_blank">Piyush Marmat</a>.
    """, unsafe_allow_html=True)

# --- File Upload ---
st.header("Upload Data File")
st.markdown("Upload a `.npy` or `.npz` file containing single-pulse polarimetric data in shape `(num_pulses, 4, num_phase_bins)`")
st.warning("Reloading the page will clear your uploaded file. Be sure to download your results if needed.")
uploaded_file = st.file_uploader("Choose a file", type=["npy", "npz"])

@st.cache_data(show_spinner=False)
def load_data(file):
    if file.name.endswith('.npz'):
        npzfile = np.load(file)
        key = list(npzfile.keys())[0]
        return npzfile[key]
    else:
        return np.load(file)

theme_choice = st.radio("Select theme for plots:", ["Light", "Dark"], horizontal=True)

# Store in session_state to persist across reruns
st.session_state["theme"] = theme_choice.lower()

def apply_streamlit_theme_to_matplotlib():
    import matplotlib.pyplot as plt
    theme = st.session_state.get("theme", "light")
    plt.rcdefaults()
    if theme == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

# --- Load and check data ---
if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        st.success(f"Data loaded: shape = {data.shape}")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    if len(data.shape) != 3 or data.shape[1] != 4:
        st.error("Invalid data shape. Expected shape: (num_pulses, 4, num_phase_bins)")
        st.stop()

    # ---------------- Waterfall & Profiles ----------------
    st.header("Waterfall and Integrated Profiles")
    st.markdown("Visualize how each Stokes parameter evolves with pulse number and rotational phase. You can zoom in on a selected phase range.")

    default_start = 0.33
    default_end = 0.66

    col1, col2, col3 = st.columns(3)
    with col1:
        start_phase = st.number_input("Start phase", min_value=0.0, max_value=1.0, value=default_start, step=0.01)
    with col2:
        end_phase = st.number_input("End phase", min_value=0.0, max_value=1.0, value=default_end, step=0.01)
    with col3:
    	fraction = st.number_input("Fraction of maximum to define off-pulse cut-off", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    apply_streamlit_theme_to_matplotlib()
    fig = plot_waterfalls_and_profiles(data, start_phase, end_phase, fraction)
    st.pyplot(fig)
    
    # ---------------- Polarisation Parameters ----------------
    st.header("Polarisation Parameters vs Phase")
    st.markdown("""
    Plot of key polarization parameters as a function of rotational phase. You can zoom in on a selected phase range.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Initial phase", min_value=0.0, max_value=1.0, value=default_start, step=0.01)
    with col2:
        end_phase = st.number_input("Final phase", min_value=0.0, max_value=1.0, value=default_end, step=0.01)
        
    fig = plot_polarisation_parameters(data, start_phase, end_phase)
    st.pyplot(fig)
    
    
    # ---------------- Distribution Heatmaps ----------------
    st.header("2D Phase-Resolved Parameter Histograms")
    st.markdown("""
    These heatmaps show how each parameter is distributed across both phase and multiple pulses, providing insight into the variability of polarization states.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("First phase", min_value=0.0, max_value=1.0, value=default_start, step=0.01)
    with col2:
        end_phase = st.number_input("Last phase", min_value=0.0, max_value=1.0, value=default_end, step=0.01)
    
    fig = plot_polarisation_histograms(data, start_phase, end_phase, fraction)
    st.pyplot(fig)

    
    # ---------------- Phase Slice Histograms ----------------
    st.header("1D Parameter Histograms at Selected Phases")
    st.markdown("""
    Explore how polarization parameters are distributed at individual phase slices across all pulses.
    """)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        left_phase = st.number_input("left phase", min_value=0.0, max_value=1.0, value=default_start, step=0.01)
    with col4:
        mid_phase = st.number_input("mid phase", min_value=0.0, max_value=1.0, value=default_end, step=0.01)
    with col5:
        right_phase = st.number_input("right phase", min_value=0.0, max_value=1.0, value=default_end, step=0.01)
        
    fig = plot_phase_slice_histograms_by_phase(data, [left_phase, mid_phase, right_phase], fraction)
    st.pyplot(fig)

    # ---------------- Poincaré Sphere ----------------
    st.header("Polarization State on the Poincaré Sphere")
    st.markdown("""
    Hammer-Aitoff Projection of the Poincaré Sphere
    """)
    
    col6, col7 = st.columns(2)
    with col6:
        start_phase = st.number_input("Starting phase", min_value=0.0, max_value=1.0, value=default_start, step=0.01)
    with col7:
        end_phase = st.number_input("Ending phase", min_value=0.0, max_value=1.0, value=default_end, step=0.01)

    fig = plot_poincare_aitoff_from_data(data, [start_phase, end_phase], fraction)
    st.pyplot(fig)
    
    st.markdown("""
    Interactive 3D visualisation of the Poincaré Sphere
    """)

    fig = plot_interactive_poincare_sphere(data, [start_phase, end_phase], fraction)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a valid `.npy` or `.npz` file to begin.")
