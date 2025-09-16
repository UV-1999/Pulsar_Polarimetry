import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth
import io
import re
from streamlit_js_eval import streamlit_js_eval

today = datetime.today().strftime("%B %d, %Y")
from modules.plots import *

st.set_page_config(page_title="Pulsar Polarimeter", layout="wide")

st.title("Pulsar Polarimeter")
st.markdown("""
<h4>
An open-source data-analysis tool for visualizing and exploring single-pulse polarimetry data from the  
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
    - 2|EA| v/s P/I plot as described in <a href="https://doi.org/10.1093/mnras/stad2271" target="_blank">Oswald et al. (2023)</a>. This plots helps in applying the partial coherence model.
    - 2D Histograms of polarization parameters with 1D Histograms for specific phases.
    - Trajectories of polarization state on the Poincaré sphere (Aitoff projection and 3D) in specific phase region.
    - Polarization states on the Poincaré sphere at a fixed phase for all pulses (change phase (normalised pulse longitude) to see polarisation modes (O and X) as clusters)
    - Linear polarisation parameter is corrected for bias.
    - Radius of curvature (via circle fitting) of the polarization trajectory on the Poincaré sphere as a function of pulse phase.
    - For an uploaded Numpy file, the on pulse window is inferred from the noise floor which is a fraction (user-input) of maximum intensity of the integrated profile.
    - For data directly uploaded from Meertime URL, the on pulse is automatically inferred.

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
def generate_plot1(data, start_phase, end_phase, obs_id):
    return plot_waterfalls_and_profiles(data, start_phase, end_phase, obs_id)
@st.fragment
def H1(data):
    st.header("Waterfall and Integrated Stokes Parameters")
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h11")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h12")
    fig = generate_plot1(data, start_phase, end_phase, obs_id)
    st.pyplot(fig)
   
@st.cache_data
def generate_plot2(data, start_phase, end_phase, on_pulse, pulse_index, obs_id):
    return plot_single_pulse_stokes(data, start_phase, end_phase, on_pulse, pulse_index, obs_id)
@st.fragment
def H2(data):
    st.header("Polarisation Parameters for an Individual Pulse Profile")
    top_indices = get_top_pulse_indices(data, 10)
    st.write(f"Top 10 pulses by energy:")
    st.write(", ".join(str(i) for i in top_indices))
    mindex = data.shape[0] - 1
    col1, col2, col3 = st.columns(3)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h21")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h22")
    with col3:
        pulse_index = st.number_input("Pulse Index", min_value=0, max_value=mindex, value=top_indices[0], step=1)
    fig = generate_plot2(data, start_phase, end_phase, on_pulse, pulse_index, obs_id)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot3(data, start_phase, end_phase, on_pulse, obs_id):
    return plot_polarisation_parameters(data, start_phase, end_phase, on_pulse, obs_id)
@st.fragment
def H3(data):
    st.header("Polarisation Parameters for Integrated Profile")
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h31")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h32")
    fig = generate_plot3(data, start_phase, end_phase, on_pulse, obs_id)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot4(data, start_phase, end_phase, on_pulse, obs_id):
    return plot_polarisation_histograms(data, start_phase, end_phase, on_pulse, obs_id)    
@st.fragment
def H4(data):
    st.header("2D Phase-Resolved Parameter Histograms (Log-Color)")
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h41")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h42")
    fig = generate_plot4(data, start_phase, end_phase, on_pulse, obs_id)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot5(data, left_phase, mid_phase, right_phase, on_pulse, obs_id):
    return plot_phase_slice_histograms_by_phase(data, left_phase, mid_phase, right_phase, on_pulse, obs_id)
@st.fragment
def H5(data):
    st.header("1D Parameter Histograms at Selected Phases")
    col1, col2, col3 = st.columns(3)
    with col1:
        left_phase = st.number_input("Left Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h51")
    with col2:
        mid_phase = st.number_input("Mid Phase", min_value=0.0, max_value=1.0, value=def_mid, step=0.001, format="%.3f", key="h52")
    with col3:
        right_phase = st.number_input("Right Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h53")
    fig = generate_plot5(data, left_phase, mid_phase, right_phase, on_pulse, obs_id)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot6(data, start_phase, end_phase, on_pulse, obs_id):
    return plot_poincare_aitoff_from_data(data, start_phase, end_phase, on_pulse, obs_id)    
@st.fragment
def H6(data):
    st.header("Polarization State on the Poincaré Sphere")
    st.subheader("""
    Hammer-Aitoff Projection of the Poincaré Sphere
    """)
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h61")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h62")
    fig = generate_plot6(data, start_phase, end_phase, on_pulse, obs_id)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot7(data, start_phase, end_phase, on_pulse, obs_id):
    return plot_interactive_poincare_sphere(data, start_phase, end_phase, on_pulse, obs_id)
@st.fragment
def H7(data):
    st.subheader("""
    Interactive 3D visualisation of the Poincaré Sphere
    """)
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h71")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h72")
    fig = generate_plot7(data, start_phase, end_phase, on_pulse, obs_id)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def generate_plot8(data, start_phase, end_phase, on_pulse, obs_id):
    return plot_radius_of_curvature_from_data(data, start_phase, end_phase, on_pulse, obs_id)
@st.fragment
def H8(data):
    st.subheader("""
    Radius of curvature (via circle fitting) of the polarization trajectory on the Poincaré sphere as a function of pulse phase
    """)
    col1, col2 = st.columns(2)
    with col1:
        start_phase = st.number_input("Start Phase", min_value=0.0, max_value=1.0, value=def_start, step=0.001, format="%.3f", key="h81")
    with col2:
        end_phase = st.number_input("End Phase", min_value=0.0, max_value=1.0, value=def_end, step=0.001, format="%.3f", key="h82")
    fig = generate_plot8(data, start_phase, end_phase, on_pulse, obs_id)
    st.pyplot(fig)
    
@st.cache_data
def generate_plot9(data, on_pulse, cphase, obs_id):
    return plot_poincare_aitoff_at_phase(data, on_pulse, cphase, obs_id)
@st.fragment
def H9(data):
    st.subheader("""
    Plot Aitoff projection of polarization states on the Poincaré sphere at a fixed phase for all pulses.
    """)
    col = st.columns(1)[0]
    with col:
        cphase = st.number_input("Phase", min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f", key="h91")
    fig = generate_plot9(data, on_pulse, cphase, obs_id)
    st.pyplot(fig)

st.subheader("Upload Data (`.npz` or `.npy`) or download from the MeerTime database")
data = None
obs_id = None
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.obs_id = None
    st.session_state.default_start = None
    st.session_state.default_end = None

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
        
        I = st.session_state.data[:, 0, :].mean(axis=0)
        fraction = st.number_input("Fraction", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f", key="f")
        threshold = fraction * np.max(I)
        on_pulse_mask = I >= threshold
        on_indices = np.where(on_pulse_mask)[0]
        start_idx = on_indices[0]
        end_idx = on_indices[-1]
        phase_axis = np.linspace(0, 1, st.session_state.data.shape[2])
        default_start = phase_axis[start_idx]
        default_end = phase_axis[end_idx]
        st.session_state.default_start = default_start
        st.session_state.default_end = default_end

    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        st.stop()

def extract_obs_id(url: str, info: dict) -> str:
    match = re.search(r"singlepulse/([^/]+)/([^/]+)/", url)
    if match:
        pulsar = match.group(1)
        datetime_str = match.group(2)
        date_parts = datetime_str.split("-")
        if len(date_parts) >= 3:
            date = "-".join(date_parts[:3])
            time = "-".join(date_parts[3:])
        else:
            date = datetime_str
            time = "Unknown"
        freq = round(float(info.get("input_data", {}).get("header", {}).get("FREQ", "0.0")), 2)
        freq_str = f"{freq:.2f}"
        return f"Pulsar-{pulsar}_Date-{date}_Time-{time}_Obs_Freq-{freq_str}_MHz"
    return "Unknown"
    
DEFAULT_URL = "https://psrweb.jb.man.ac.uk/meertime/singlepulse/J0304+1932/2021-01-25-18:54:21/1284/plots/2021-01-25-18:54:21.npz"
js_result = streamlit_js_eval(js_expressions="sessionStorage.getItem('last_url')", key="get_url")

if js_result is None:
    js_result = DEFAULT_URL

initial_url = js_result or DEFAULT_URL
url = st.text_input("Paste the full .npz file URL here and hit enter to cache it:", value=initial_url)
if url and url != js_result:
    streamlit_js_eval(js_expressions=f"sessionStorage.setItem('last_url', `{url}`)", key="set_url")


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
            base_url = url.rsplit('/', 1)[0]
            json_url = base_url + "/pipeline_info.json"
            json_response = requests.get(json_url, auth=HTTPBasicAuth(username, password))
            json_response.raise_for_status()
            info = json_response.json()
            st.session_state.obs_id = extract_obs_id(url, info)
            windows_on = info.get("windows", {}).get("on", [[0.0, 1.0]])[0]
            default_start, default_end = windows_on
            st.session_state.default_start = default_start
            st.session_state.default_end = default_end
            st.success(f"File loaded successfully")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} – {e.response.reason}")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

data = st.session_state.data
obs_id = st.session_state.obs_id
def_start = st.session_state.default_start
def_end = st.session_state.default_end
on_pulse = [def_start, def_end]

if data is not None:
    if len(data.shape) != 3 or data.shape[1] != 4:
        st.error("Invalid data shape. Expected shape: (num_pulses, 4, num_phase_bins)")
        st.stop()
    st.success(f"Data shape: {data.shape}")
    st.warning("The app can sleep if left inactive for long time and requires reloading. Reloading will clear the data and plots. Be sure to download your results if needed.")    
    def_mid   = (def_start + def_end)/2.0
    with st.expander("View/Hide - Waterfall and Integrated Stokes Parameters", expanded=True):
        H1(data)
    with st.expander("View/Hide - Polarisation Parameters for Integrated Profile", expanded=True):
        H3(data)
    with st.expander("View/Hide - 2D Phase-Resolved Parameter Histograms (Log-Color)", expanded=True):
        H4(data)
    with st.expander("View/Hide - Polarisation Parameters for an Individual Pulse Profile", expanded=True):
        H2(data)
    with st.expander("View/Hide - 1D Parameter Histograms at Selected Phases", expanded=True):
        H5(data)
    with st.expander("View/Hide - Hammer-Aitoff Projection of the Poincaré Sphere", expanded=True):
        H6(data)
    with st.expander("View/Hide - Interactive 3D visualisation of the Poincaré Sphere", expanded=True):
        H7(data)
    with st.expander("View/Hide - Radius of curvature (via circle fitting) of the polarization trajectory on the Poincaré sphere as a function of pulse phase", expanded=True):
        H8(data)
    with st.expander("View/Hide - Plot Aitoff projection of polarization states on the Poincaré sphere at a fixed phase for all pulses", expanded=True):
        H9(data)
else:
    st.info("Please upload a valid file OR provide a valid link with credentials.")
