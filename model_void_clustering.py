# Limit number of cores
import os
os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Create logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

import pyccl as ccl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import galaxy alignment predictor
# Add root directory to path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)
    

rootpath = Path('/disks/shear16/herle/code/GNAT')
if str(rootpath) not in os.sys.path:
    os.sys.path.append(str(rootpath))

from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum

import pandas as pd
import h5py
import re

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
    n_s=0.967
)

# Define redshifts at which to calculate estimators
redshift_list = np.array([0])
z=0

kmin, kmax = 1e-5, 1e2

rp_values2v = np.logspace(-1, np.log10(900), 300)

# Define scales at which to calculate estimators
rp_array = rp_values2v
r_array = rp_array

# Pi_max for projection integrals
pi_max = 100.0  # in Mpc
h = 0.681

# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T

powerSpectrumHandler = powerSpectrum()

# ============================================================
#  Read FLAMINGO P(k)
# ============================================================

def read_flamingo_power_spectrum(file_path):
    with h5py.File(file_path, 'r') as f:
        z_keys = sorted(
            [key for key in f.keys() if re.match(r"z=\d+\.\d+", key)],
            key=lambda k: float(k.split('=')[1])
        )

        a_list, pk_list = [], []

        for zkey in z_keys:
            z = float(zkey.split('=')[1])
            a = 1.0 / (1.0 + z)
            k = f[f"{zkey}/k"][:]
            p = f[f"{zkey}/P(k)"][:]

            pk_list.append(p)
            a_list.append(a)

    a_array = np.array(a_list)
    P_array = np.array(pk_list).T

    sort_idx = np.argsort(a_array)
    a_array = a_array[sort_idx]
    P_array = P_array[:, sort_idx]

    return P_array, k, a_array


flamingo_file_path = '/disks/shear16/herle/power_spectra/L2p8_m9.hdf5'
P_flamingo, k_flamingo, a_flamingo = read_flamingo_power_spectrum(flamingo_file_path)

k_switch = k_flamingo.min() 

pk_ccl = powerSpectrumHandler.ccl_power_spectrum(
    cosmo=cosmo_ccl,
    k_input=k_array_single,
    redshift=z,
    pk_type='nonlinear_matter'
)

pk = np.zeros_like(k_array_single)

a_target = 1/(1+0.0)
idx = np.argmin(np.abs(a_flamingo - a_target))

print(a_flamingo[idx], 1/a_flamingo[idx]-1)

mask_low = k_array_single < k_switch
mask_high = ~mask_low

pk[mask_low] = pk_ccl[mask_low]

interp = interp1d(
    np.log(k_flamingo),
    np.log(P_flamingo[:, idx]),
    kind="linear",
    bounds_error=True
)

pk[mask_high] = np.exp(interp(np.log(k_array_single[mask_high])))


from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Smooth stitched power spectrum on uniform log-k grid
# ============================================================

logk = np.log(k_array_single)
logpk = np.log(pk)

# Create uniform log-k grid
N_log = 20000
logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

# Interpolate log P(k) onto uniform log-k grid
interp_func = interp1d(
    logk,
    logpk,
    kind="linear"
)

logpk_uniform = interp_func(logk_uniform)

# Savitzky-Golay smoothing
window_length = 1001
polyorder = 3

if window_length % 2 == 0:
    window_length += 1

logpk_uniform_smooth = savgol_filter(
    logpk_uniform,
    window_length,
    polyorder
)

# Interpolate smoothed log P(k) back to original k grid
interp_back = interp1d(
    logk_uniform,
    logpk_uniform_smooth,
    kind="linear"
)

logpk_smoothed = interp_back(logk)

# Convert back
pk_smoothed = np.exp(logpk_smoothed)
# reshape to predictor format: (n_k, n_pk, n_redshifts)
pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))


delta_v = np.load('/disks/shear16/herle/code/zobov_FIDUCIAL_void_delta_k_bin0_raw.npy')

# Reconstruct the k-grid delta_v was actually saved on
# (must match the void script exactly: log10kmin=-5, log10kmax=2, N=1000001, then *h)
k_native = np.logspace(-5, 2, 1000001)   # h/Mpc
k_void = k_native * h                     # Mpc^-1, matches delta_v's saved values

assert delta_v.shape[0] == k_void.shape[0], \
    f"delta_v has {delta_v.shape[0]} points but k_void has {k_void.shape[0]}"

# Interpolate delta_v(k) from its native grid onto k_array_single
logk_void = np.log10(k_void)
logk_target = np.log10(k_array_single)

interp_to_target = interp1d(
    logk_void, delta_v, kind="linear",
    bounds_error=False, fill_value=0.0
)
delta_v_on_target = interp_to_target(logk_target)   # shape (1000,), matches k_array_single

# --- now smooth on k_array_single's own log grid ---
logk = logk_target

N_log = 20000
logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

interp_func = interp1d(logk, delta_v_on_target, kind="linear")
delta_uniform = interp_func(logk_uniform)

window_length = 3001
polyorder = 3
if window_length % 2 == 0:
    window_length += 1

delta_uniform_smooth = savgol_filter(delta_uniform, window_length, polyorder)

interp_back = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")
delta_v_smoothed = interp_back(logk)   # shape (1000,), exactly matches k_array_single


for idx_z, z in enumerate(redshift_list):
    pk_input[:, 0, idx_z] = -np.pi**2*pk_smoothed * (delta_v_smoothed**2)

predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)

predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input,
    probetype_order = ['gg'],  # order 2 for g+, order 0 for gg
)

# The rest of the procedure is the same as above
predictor.calculate_estimators(
    rp_support=rp_array,
    beta_shape=0, # No RSD, for example in real space
    beta_density=0, # No RSD, for example in real space
    pi_max=pi_max,
)


# Retrieve projection estimators as arrays
w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
    rp_support=rp_array,
    r_support=r_array,
)


w_vv =  w_arr[:, 0, 0]

model_rp = rp_array * h

def model_wVV(p):

    b_v = p[0]

    wVV_model = w_vv.copy()
    
    wVV_model *= b_v**2

    return model_rp, wVV_model


b = 1.0

p = [b]

rp_model_VV, wVV_model = model_wVV(p) 

scaling = 1e-11
wVV_model *=scaling

run = 'L1000N1800'
redshift = '0.0'
Rmin, Rmax = 10, 20
#/disks/shear16/herle/correlations1/correlations_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'.h5
meas_file = (
    f'/disks/shear16/herle/correlations1/'
    f'correlations_joint_revolver_{Rmin}_{Rmax}hMpc_' + run +
    f'_HYDRO_FIDUCIAL_' + redshift + '.h5'
)

df = pd.read_hdf(meas_file, key='a')

rp_meas = df['wgg_rp'].values
wvv_meas = df['wgg_xip'].values 
err_meas = df['wgg_std_from_cov'].values 




plt.figure(figsize=(10, 10))


# --------------------
# Model
# --------------------
model_pos = wVV_model >= 0
model_neg = ~model_pos

plt.plot(
    rp_model_VV[model_pos],
    (rp_model_VV * wVV_model)[model_pos],
    '-',
)

plt.plot(
    rp_model_VV[model_neg],
    np.abs((rp_model_VV * wVV_model)[model_neg]),
    '--',  # optional: dashed for negative
)

# --------------------
# Measurements
# --------------------
data_pos = wvv_meas >= 0
data_neg = ~data_pos

# Filled markers
plt.errorbar(
    rp_meas[data_pos],
    (rp_meas * wvv_meas)[data_pos],
    yerr=(rp_meas * err_meas)[data_pos],
    fmt='o',
    mfc='C0',
    mec='C0',
    capsize=3,
    label='Zobov'
)

# Open markers
plt.errorbar(
    rp_meas[data_neg],
    np.abs((rp_meas * wvv_meas)[data_neg]),
    yerr=(rp_meas * err_meas)[data_neg],
    fmt='o',
    mfc='none',
    mec='C0',
    capsize=3,
)



plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r_p$ [Mpc/h]')
plt.ylabel(r'$r_p w_{\mathrm{VV}} [\mathrm{Mpc}^2/h^2]$')


plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/w_vv_model_by_hand.png')
