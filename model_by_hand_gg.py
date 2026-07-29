import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T

# # Fill pk_input with g+ and gg power spectra
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

for idx_z, z in enumerate(redshift_list):
    pk_input[:, 0, idx_z] = -np.pi**2*pk_smoothed

# Then, we initialise the power spectra in the predictor
predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)

# 2 void term
predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input,
    probetype_order = ['g+'],  # order 2 for g+, order 0 for gg
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


w_2v =  w_arr[:, 0, 0]

# 1 void term

h = 0.681

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

# --- reshape to match pk_input convention: (n_k, n_pk, n_redshifts) ---
pk_input_1v = np.zeros((len(k_array_single), 1, len(redshift_list)))
for idx_z, z in enumerate(redshift_list):
    pk_input_1v[:, 0, idx_z] = -np.pi**2*delta_v_smoothed

predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input_1v,
    probetype_order=['g+'],
)

predictor.calculate_estimators(
    rp_support=rp_array,
    beta_shape=0,
    beta_density=0,
    pi_max=pi_max,
)

w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
    rp_support=rp_array,
    r_support=r_array,
)

w_1v = w_arr[:, 0, 0]

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo_ccl['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo_ccl, scale_factor)

wVII = w_2v.copy()
wVII *= scaling
wVII *=h

wVI = w_1v.copy()
wVI *= scaling
wVI *= h

model_rp = rp_array * h

# Pi_max for projection integrals
pi_max = 100.0  # in Mpc/h

# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(1e-5, 5e2, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 2, 1)).T
pk_input = np.zeros((len(k_array_single), 2, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

# Fill pk_input with g+ and gg power spectra
powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )

    # Initialise power spectra
    pkgp = powerSpectrumHandler.convert_pk_matter_to_pk_gplus(
        pk_matter=pkmm_ccl,
        cosmo=cosmo_ccl,
        redshift=z
    )
    pkgg = powerSpectrumHandler.convert_pk_matter_to_pk_gg(
        pk_matter=pkmm_ccl,
        cosmo=cosmo_ccl,
        redshift=z,
        )

    pk_input[:, 0, idx_z] = pkgp
    pk_input[:, 1, idx_z] = pkgg

# Then, we initialise the power spectra in the predictor
predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)
predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input,
    probetype_order = ['g+', 'gg'],
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

# Retrieve splines for projection estimators
# projection_splines is a list of lists of interp1d objects
# Outer list runs over redshifts, inner list over power spectrum types (g+, gg)
projection_splines = predictor.return_projection_splines()

for idx_z, z in enumerate(redshift_list):
    w_gg_spline = projection_splines[idx_z][1]  # gg projection spline
    w_gg = w_gg_spline(rp_array)

h = 0.681

Rv = 15.0

lower_rp_VI = 6.0
upper_rp_VI = 2.5 * Rv

lower_rp_VII = 4.0 * Rv
upper_rp_VII = 500.0

model_mask_VI = (model_rp >= lower_rp_VI) & (model_rp <= upper_rp_VI)
model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

def model_wVI(p):

    A2_v = p[1]
    b_g = p[3]

    wVI_model = wVI[model_mask_VI].copy()
    wVI_model *= A2_v * b_g

    return model_rp[model_mask_VI], wVI_model

def model_wVII(p):

    A1_v = p[0]
    b_v = p[2]

    wVII_model = wVII[model_mask_VII].copy()
    wVII_model *= A1_v*b_v

    return model_rp[model_mask_VII], wVII_model

def model_wgg(p):

    b_g = p[3]

    wgg_model = w_gg.copy()
    wgg_model *=h
    wgg_model *= b_g**2

    return model_rp, wgg_model



A_1v, A_2v, b_v, b_g = 5.0, 5.0, 1.3, 1.6

p = [A_1v, A_2v, b_v, b_g]

rp_model_VI, wVI_model   = model_wVI(p)      # returns only the VI part
rp_model_VII, wVII_model = model_wVII(p)     # returns only the VII part

rp_model_wgg, wgg_model = model_wgg(p)



run = 'L1000N1800'
redshift = '0.0'

meas_file = (
    f'/disks/shear16/herle/correlations1/'
    f'correlations_joint_revolver_10_20hMpc_{run}_HYDRO_FIDUCIAL_{redshift}.h5'
)

df = pd.read_hdf(meas_file)

# --------------------------
# Measurements
# --------------------------

rp_wvv = df['wvv_rp'].values
wvv = df['wvv_xip'].values
wvv_err = df['wvv_std_from_cov'].values
xi_vv = df['xi_vv'].values
xi_jk = np.load('/disks/shear16/herle/correlations1/xi_vv_jk_voids_joint_revolver_10_20hMpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.npy')
X = xi_jk[:, 0, :]   # because npi = 1
Njk = X.shape[0]
mean = X.mean(axis=0)
diff = X - mean
cov = (Njk - 1) / Njk * diff.T @ diff
xi_vv_err = np.sqrt(np.diag(cov))

rp_wvp = df['wvp_rp'].values
wvp = df['wvp_xip'].values
wvp_err = df['wvp_std_from_cov'].values

rp_wgg = df['wgg_rp'].values
wgg = df['wgg_xip'].values
wgg_err = df['wgg_std_from_cov'].values

xi_gg = df['xi_gg'].values
xi_jk = np.load('/disks/shear16/herle/correlations1/xi_gg_jk_voids_joint_revolver_10_20hMpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.npy')
X = xi_jk[:, 0, :]   # because npi = 1
Njk = X.shape[0]
mean = X.mean(axis=0)
diff = X - mean
cov = (Njk - 1) / Njk * diff.T @ diff
xi_gg_err = np.sqrt(np.diag(cov))

def model_xi_vv(p):

    b_g = p[3]
    b_v = p[2]


    xi_vv_model = ((b_g/b_v)**2)*xi_gg

    return model_rp, xi_vv_model

rp_model_xi_vv, xi_vv_model = model_xi_vv(p)

# --------------------------
# Figure
# --------------------------

fig, axs = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharex=True
)

##########################################################
# wvp
##########################################################

ax = axs[0]

ax.errorbar(
    rp_wvp,
    rp_wvp * wvp,
    yerr=rp_wvp * wvp_err,
    fmt='o',
    color='k',
    capsize=3,
    label='Measurement'
)

ax.plot(rp_model_VII, rp_model_VII*wVII_model, ls='--')
ax.plot(rp_model_VI,  rp_model_VI*wVI_model, ls='--')

ax.axvspan(lower_rp_VI, upper_rp_VI, alpha=0.08)
ax.axvspan(lower_rp_VII, upper_rp_VII, alpha=0.08)

ax.axhline(0, color='k', ls=':')

ax.set_xscale('log')
ax.set_xlabel(r'$r_p$ [Mpc/h]')
ax.set_ylabel(r'$r_p w_{\rm V+}$')

ax.legend(frameon=False)

##########################################################
# wgg
##########################################################

ax = axs[1]

ax.errorbar(
    rp_wgg,
    rp_wgg * wgg,
    yerr=rp_wgg * wgg_err,
    fmt='o',
    color='k',
    capsize=3
)


ax.plot(rp_model_wgg, rp_model_wgg*wgg_model, ls='--')

ax.axhline(0, color='k', ls=':')

ax.set_xscale('log')
ax.set_xlabel(r'$r_p$ [Mpc/h]')
ax.set_ylabel(r'$r_p w_{\rm gg}$')

##########################################################
# wvv
##########################################################

ax = axs[2]



ax.errorbar(
    rp_wvv,
    (xi_vv),
    yerr=(xi_vv_err),
    fmt='o',
    mfc='k',
    mec='k',
    ecolor='k',
    capsize=3
)

ax.plot(rp_model_wvv, xi_vv_model, ls='--')

ax.axhline(0, color='k', ls=':')

ax.set_xscale('log')


ax.set_xlabel(r'$r_p$ [Mpc/h]')
ax.set_ylabel(r'$xi_{\rm vv}$')


plt.tight_layout()

plt.savefig(
    '/disks/shear16/herle/code/plots/clustering_joint_measurements.png',
    dpi=300
)


