import streamlit as st
import numpy as np

# Section where we import new functions for plotting:
from modules.plot_waterfalls_and_profiles import *
# #

# --- Page configuration ---
st.set_page_config(page_title="Pulsar Polarimeter", layout="wide")

# --- Title and subtitle with clickable links ---
st.title("Pulsar Polarimeter")

st.markdown(
    """
    <h4>
    A data-analysis app by 
    <a href="https://uv-1999.github.io/" target="_blank">Piyush Marmat</a> 
    for pulsar polarimetry using the 
    <a href="https://psrweb.jb.man.ac.uk/meertime/singlepulse/" target="_blank">MeerTime Single Pulse Database</a>.
    </h4>
    """, 
    unsafe_allow_html=True
)

# --- Description ---
st.markdown("""
This Streamlit app takes in data in numpy array format (.npy) available at the MeerTime Database for download.  
This app helps researchers to visualize patterns in data by providing the following plots and more to come in the future:

- Waterfalls and Integrated profiles for each Stokes channel (I, Q, U, V)
- Distributions of polarization degree (total p, linear L, and circular V)
- Polarization Angle (PA) vs Pulse Phase, for Integrated profile  
- Poincaré Sphere views segmented by pulse phase  
- Histograms and greyscale distribution plots at selected pulse phases

For help, queries and suggestions, please contact **Piyush Marmat**.
""")

st.header("Upload your data file (.npz or .npy)")
uploaded_file = st.file_uploader("Upload your .npz or .npy data file", type=['npz', 'npy'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.npz'):
        npzfile = np.load(uploaded_file)
        # Assuming the first array inside .npz is your data
        data_key = list(npzfile.keys())[0]
        data = npzfile[data_key]
    else:  # .npy file
        data = np.load(uploaded_file)
    
    st.write(f"Loaded data shape: {data.shape}")
    if len(data.shape) != 3 or data.shape[1] != 4:
        st.error("Unexpected data shape. Expected shape: (num_pulses, 4, num_phase_bins)")
    else:
        ##########################################################################################################################
        # This is where we call functions for specific plots
        st.header("Waterfalls and Integrated profiles for each Stokes channel (I, Q, U, V)")
        st.markdown("Select Pulse Phase to Zoom")
        start_phase = st.number_input("Start phase (default 0.33)", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
        end_phase = st.number_input("End phase (default 0.66)", min_value=0.0, max_value=1.0, value=0.66, step=0.01)
        if st.button("Plot with selected phases"):
        # Call your function here, pass the start_phase and end_phase as arguments
            fig = plot_waterfalls_and_profiles(data, start_phase, end_phase)
            st.pyplot(fig)
        
        ##########################################################################################################################
else:
    st.info("Please upload a valid NumPy .npy or .npz file containing your pulsar data.")
