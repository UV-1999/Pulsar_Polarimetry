import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth
import io
import re

today = datetime.today().strftime("%B %d, %Y")
from modules.plots import *
st.set_page_config(page_title="Pulsar Polarimeter", layout="wide")
st.title("Pulsar Polarimeter")
st.markdown("""
<h4>
A data-analysis tool by for visualizing and exploring single-pulse polarimetry data from the  
<a href="https://psrweb.jb.man.ac.uk/meertime/singlepulse/" target="_blank">MeerTime Single Pulse Database</a>.
</h4>
""", unsafe_allow_html=True)
with st.expander("About this App", expanded=False):
    st.markdown(f"""
    This tool lets you upload pulsar single-pulse data in .npy or .npz format (as available from the MeerTime database),
    and generates a series of interactive visualizations to explore the polarization state of the pulsar signal.
    
    **Features included as on {today}:**
    - Waterfall plots and integrated profiles of each Stokes parameter.
    - Individual pulse profiles for selected pulse indices.
    - Polarization parameter vs pulse phase intergrated over all pulses.
    - 2D Histograms of polarization parameters with 1D Histograms for specific phases.
    - Trajectories of polarization state on the Poincaré sphere (Aitoff projection and 3D) in specific phase region.
    - Linear polarisation parameter is corrected for bias.
    - On-pulse window can be changed after plotting the intergrated profile.
    - Plot radius of curvature (via circle fitting) of the polarization trajectory on the Poincaré sphere as a function of pulse phase

    **Abbreviations and symbols used:**
    - I, Q, U, V are the four Stokes parameters.
    - PA is polarisation angle, EA is ellipticity angle.
    - L is linear polarisation parameter and P is total polarisation parameter.
    - lowercase l and p are fractional polarisation parameters for linear and total respectively.
    
    **Need help or have any suggestions?** Reach out to <a href="https://uv-1999.github.io/" target="_blank">Piyush Marmat</a>.
    """, unsafe_allow_html=True)

def load_data(uploaded_file):
     if uploaded_file.name.endswith(".npz"):
         with np.load(uploaded_file) as npzfile:
             key = list(npzfile.keys())[0]
             data = npzfile[key]
     elif uploaded_file.name.endswith(".npy"):
         data = np.load(uploaded_file)
     return data

@st.cache_data
def generate_plot1(data, start_phase, end_phase, fraction):
    return plot_waterfalls_and_profiles(data, start_phase, end_phase, fraction)
@st.fragment
def H1(data):
    st.header("Waterfall and Integrated Stokes Parameters")
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h11")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h12")
    fig = generate_plot1(data, start_phase, end_phase, fraction)
    st.pyplot(fig)
   
@st.cache_data
def generate_plot2(data, start_phase, end_phase, fraction, pulse_index):
    return plot_single_pulse_stokes(data, start_phase, end_phase, fraction, pulse_index)
@st.fragment
def H2(data):
    st.header("Polarisation Parameters for an Individual Pulse Profile")
    top_indices = get_top_pulse_indices(data, 10)
    st.write(f"Top 10 pulses by energy:")
    st.write(", ".join(str(i) for i in top_indices))
    mindex = data.shape[0] - 1
    col1, col2, col3 = st.columns(3)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h21")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h22")
    with col3:
        pulse_index = st.number_input("Pulse Index", min_value=0, max_value=mindex, value=0, step=1)
    fig = generate_plot2(data, start_phase, end_phase, fraction, pulse_index)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot3(data, start_phase, end_phase, fraction):
    return plot_polarisation_parameters(data, start_phase, end_phase, fraction)
@st.fragment
def H3(data):
    st.header("Polarisation Parameters for Integrated Profile")
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h31")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h32")
    fig = generate_plot3(data, start_phase, end_phase, fraction)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot4(data, start_phase, end_phase, fraction):
    return plot_polarisation_histograms(data, start_phase, end_phase, fraction)    
@st.fragment
def H4(data):
    st.header("2D Phase-Resolved Parameter Histograms (Log-Color)")
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h41")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h42")
    fig = generate_plot4(data, start_phase, end_phase, fraction)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot5(data, left_phase, mid_phase, right_phase, fraction):
    return plot_phase_slice_histograms_by_phase(data, left_phase, mid_phase, right_phase, fraction)
@st.fragment
def H5(data):
    st.header("1D Parameter Histograms at Selected Phases")
    col1, col2, col3 = st.columns(3)
    with col1:
        left_phase = st.number_input("Left Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h51")
    with col2:
        mid_phase = st.number_input("Mid Phase", min_value=0.0, max_value=1.0, value=default_mid, step=0.001, format="%.3f", key="h52")
    with col3:
        right_phase = st.number_input("Right Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h53")
    fig = generate_plot5(data, left_phase, mid_phase, right_phase, fraction)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot6(data, start_phase, end_phase, fraction):
    return plot_poincare_aitoff_from_data(data, start_phase, end_phase, fraction)    
@st.fragment
def H6(data):
    st.header("Polarization State on the Poincaré Sphere")
    st.subheader("""
    Hammer-Aitoff Projection of the Poincaré Sphere
    """)
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h61")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h62")
    fig = generate_plot6(data, start_phase, end_phase, fraction)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot7(data, start_phase, end_phase, fraction):
    return plot_interactive_poincare_sphere(data, start_phase, end_phase, fraction)
@st.fragment
def H7(data):
    st.subheader("""
    Interactive 3D visualisation of the Poincaré Sphere
    """)
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h71")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h72")
    fig = generate_plot7(data, start_phase, end_phase, fraction)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def generate_plot8(data, start_phase, end_phase, fraction):
    return plot_radius_of_curvature_from_data(data, start_phase, end_phase, fraction)
@st.fragment
def H8(data):
    st.subheader("""
    Radius of curvature (via circle fitting) of the polarization trajectory on the Poincaré sphere as a function of pulse phase
    """)
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=default_start, step=0.001, format="%.3f", key="h81")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=default_end, step=0.001, format="%.3f", key="h82")
    fig = generate_plot8(data, start_phase, end_phase, fraction)
    st.pyplot(fig)

st.subheader("Upload Data (`.npz` or `.npy`) or download from the MeerTime database")
data = None
obs_id = None
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.obs_id = None

def extract_obs_id(url: str) -> str:
    match = re.search(r"singlepulse/([^/]+/[^/]+)/", url)
    if match:
        return match.group(1).replace("/", "_")
    return "Unknown"

uploaded_file = st.file_uploader("A", type=["npz", "npy"], label_visibility="collapsed")
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".npz"):
            with np.load(uploaded_file) as npzfile:
                key = list(npzfile.keys())[0]
                st.session_state.data = npzfile[key]
        elif uploaded_file.name.endswith(".npy"):
            st.session_state.data = np.load(uploaded_file)
        st.session_state.obs_id = uploaded_file.name
        st.success(f"File uploaded successfully: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        st.stop()

default_url = "https://psrweb.jb.man.ac.uk/meertime/singlepulse/J0304+1932/2021-01-25-18:54:21/1284/plots/2021-01-25-18:54:21.npz"
url = st.text_input("Paste the full .npz file URL here:", value=default_url)
with st.expander("Authentication"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

if st.button("Load from URL"):
    if not url or not username or not password:
        st.warning("Please fill in the URL and credentials.")
    else:
        try:
            st.info("Downloading file...")
            response = requests.get(url, auth=HTTPBasicAuth(username, password))
            response.raise_for_status()

            with io.BytesIO(response.content) as f:
                with np.load(f) as npzfile:
                    key = list(npzfile.keys())[0]
                    st.session_state.data = npzfile[key]
            st.session_state.obs_id = extract_obs_id(url)
            st.success(f"File loaded successfully: {st.session_state.obs_id}")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} – {e.response.reason}")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

data = st.session_state.data
obs_id = st.session_state.obs_id
default_start = 0.40
default_mid   = 0.50
default_end   = 0.60

if data is not None:
    if len(data.shape) != 3 or data.shape[1] != 4:
        st.error("Invalid data shape. Expected shape: (num_pulses, 4, num_phase_bins)")
        st.stop()
    
    st.success(f"Data shape: {data.shape}")
    st.warning("Reloading the page will clear your uploaded file. Be sure to download your results if needed.")    
    st.warning("Visually identify the on-pulse region: ")
    fraction = st.number_input("Fraction of maximum to define off-pulse cut-off", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")

    with st.expander("View/Hide - Waterfall and Integrated Stokes Parameters", expanded=True):
        H1(data)
    with st.expander("View/Hide - Polarisation Parameters for Integrated Profile", expanded=False):
        H3(data)
    with st.expander("View/Hide - 2D Phase-Resolved Parameter Histograms (Log-Color)", expanded=False):
        H4(data)
    with st.expander("View/Hide - Polarisation Parameters for an Individual Pulse Profile", expanded=False):
        H2(data)
    with st.expander("View/Hide - 1D Parameter Histograms at Selected Phases", expanded=False):
        H5(data)
    with st.expander("View/Hide - Hammer-Aitoff Projection of the Poincaré Sphere", expanded=False):
        H6(data)
    with st.expander("View/Hide - Interactive 3D visualisation of the Poincaré Sphere", expanded=False):
        H7(data)
    with st.expander("View/Hide - Radius of curvature (via circle fitting) of the polarization trajectory on the Poincaré sphere as a function of pulse phase", expanded=False):
        H8(data)
else:
    st.info("Please upload a valid file OR provide a valid link with credentials.")
