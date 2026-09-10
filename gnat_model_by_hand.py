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

rp_values2v = np.logspace(-1, np.log10(900), 300) # Mpc/h

# Define scales at which to calculate estimators
rp_array = rp_values2v
r_array = rp_array

h = 0.681
# Pi_max for projection integrals
pi_max = 100.0  # in Mpc


# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
# pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

# # linear or non-linear ccl power spectrum

# # Fill pk_input with g+ and gg power spectra
powerSpectrumHandler = powerSpectrum()
# for idx_z, z in enumerate(redshift_list):
#     pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
#         cosmo=cosmo_ccl,
#         k_input=k_array_single,
#         redshift=z,
#         pk_type='nonlinear_matter'
#     )

#     # pk_input[:, 0, idx_z] = -np.pi**2*pkmm_ccl
#     pk_input[:, 0, idx_z] = pkmm_ccl

# print("Power spectrum: ", pk_input[:, 0, 0])
# print("Shape: ", np.shape(pk_input))

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

# print("FLAMINGO ps: ", P_flamingo)
# print("FLAMINGO ps shape: ", np.shape(P_flamingo))

# print("FLAMINGO + ccl ps: ", pk)
# print("FLAMINGO + ccl shape: ", np.shape(pk))


import matplotlib.pyplot as plt

# plt.rcdefaults()
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['font.size'] = 22

# plt.figure(figsize=(8,6))

# plt.loglog(k_array_single, pk_input[:,0,0], label='CCL nonlinear')
# plt.loglog(k_array_single, pk_ccl, '--', label='CCL linear')
# plt.loglog(k_flamingo, P_flamingo[:, idx], label='FLAMINGO')
# plt.loglog(k_array_single, pk, lw=3, label='Hybrid')

# plt.axvline(k_switch, color='red', ls=':', label=f'$k_{{switch}}={k_switch:.3g}$')

# plt.axvline(x=1e-5, color='k', linestyle='--')
# plt.axvline(x=1e2, color='k', linestyle='--')

# plt.xlabel(r"$k$")
# plt.ylabel(r"$P(k)$")
# plt.legend()
# plt.tight_layout()
# plt.savefig('/disks/shear16/herle/code/plots/comp_ps.png')

# plt.figure(figsize=(8,5))

# ccl_interp = interp1d(
#     np.log(k_array_single),
#     np.log(pk_ccl),
#     bounds_error=False,
#     fill_value="extrapolate"
# )

# ratio = P_flamingo[:, idx] / np.exp(ccl_interp(np.log(k_flamingo)))

# plt.semilogx(k_flamingo, ratio)
# plt.axhline(1, color='k', ls='--')
# plt.xlabel(r"$k$")
# plt.ylabel(r"$P_{\rm FLAMINGO}/P_{\rm non-linear}$")

# plt.ylim(-1, 5)
# plt.tight_layout()

# plt.savefig('/disks/shear16/herle/code/plots/comp_ps_ratio_zoom.png')

# print("k_switch =", k_switch)
# print("FLAMINGO k range =", k_flamingo.min(), k_flamingo.max())
# print("CCL k range =", k_array_single.min(), k_array_single.max())

# # Interpolate FLAMINGO onto the full k grid
# flam_interp = interp1d(
#     np.log(k_flamingo),
#     np.log(P_flamingo[:, idx]),
#     bounds_error=False,
#     fill_value="extrapolate"
# )

# pk_flam_full = np.exp(flam_interp(np.log(k_array_single)))

# # Ratio of stitched spectrum to CCL linear spectrum
# ratio = pk / pk_ccl

# plt.figure(figsize=(8,5))

# plt.semilogx(k_array_single, ratio)

# plt.axhline(1, color='k', ls='--')
# plt.axvline(k_switch, color='r', ls=':', 
#             label=f'$k_{{switch}}={k_switch:.2e}$')

# plt.xlim(1e-5,1e0)
# plt.ylim(0.8,1.2)

# plt.xlabel(r"$k$")
# plt.ylabel(r"$P_{\rm stitched}/P_{\rm CCL}$")
# plt.legend()
# plt.tight_layout()

# plt.savefig('/disks/shear16/herle/code/plots/comp_ps_ratio_zoom_check_transition.png')

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
# print("Smoothed spectrum shape:", pk_smoothed.shape)

# plt.figure(figsize=(8,6))

# plt.loglog(
#     k_array_single,
#     pk,
#     label="stitched",
#     lw=2
# )

# plt.loglog(
#     k_array_single,
#     pk_smoothed,
#     label="stitched + SavGol",
#     lw=2
# )

# plt.axvline(
#     k_switch,
#     color="r",
#     ls=":",
#     label=r"$k_{\rm switch}$"
# )

# plt.xlabel(r"$k$")
# plt.ylabel(r"$P(k)$")
# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     "/disks/shear16/herle/code/plots/stitched_smoothed_ps.png",
#     dpi=200
# )
# plt.figure(figsize=(8,5))

# ratio_smoothing = pk_smoothed / pk

# plt.semilogx(
#     k_array_single,
#     ratio_smoothing
# )

# plt.axhline(
#     1,
#     color="k",
#     ls="--"
# )

# plt.axvline(
#     k_switch,
#     color="r",
#     ls=":",
#     label=r"$k_{\rm switch}$"
# )

# plt.xlim(1e-4, 1e-1)
# plt.ylim(0.98,1.02)

# plt.xlabel(r"$k$")
# plt.ylabel(r"$P_{\rm smooth}/P_{\rm stitched}$")

# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     "/disks/shear16/herle/code/plots/smoothing_effect.png",
#     dpi=200
# )
# plt.figure(figsize=(8,5))

# ratio_before = pk / pk_ccl
# ratio_after  = pk_smoothed / pk_ccl

# plt.semilogx(
#     k_array_single,
#     ratio_before,
#     label="stitched"
# )

# plt.semilogx(
#     k_array_single,
#     ratio_after,
#     label="stitched + smoothing"
# )

# plt.axhline(
#     1,
#     color="k",
#     ls="--"
# )

# plt.axvline(
#     k_switch,
#     color="r",
#     ls=":",
#     label=r"$k_{\rm switch}$"
# )

# plt.xlim(1e-4,1e-1)
# plt.ylim(0.8,1.2)

# plt.xlabel(r"$k$")
# plt.ylabel(r"$P/P_{\rm CCL,nl}$")

# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     "/disks/shear16/herle/code/plots/switch_smoothing_check.png",
#     dpi=200
# )

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

delta_v = np.load('/disks/shear16/herle/code/temp_data/zobov_FIDUCIAL_void_delta_k_bin1_raw.npy')


# Reconstruct the k-grid delta_v was actually saved on
# (must match the void script exactly: log10kmin=-5, log10kmax=2, N=1000001, then *h)
k_native = np.logspace(-5, 2, 1000001)   # h/Mpc
k_void = k_native                
# k_void = k_native                  

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
#wVII *=h

wVI = w_1v.copy()
wVI *= scaling
#wVI *= h

# model_rp = rp_array * h
model_rp = rp_array

# model_rp = model_rp /h  # in Mpc
# print("Model rp range: ", model_rp.min(), model_rp.max())

Rv = 15.0
# masks
lower_rp_VI = 6.0
upper_rp_VI = 2.5*Rv

lower_rp_VII = 4.0*Rv
upper_rp_VII = 500.0

model_mask_VI = (model_rp >= lower_rp_VI) & (model_rp <= upper_rp_VI)
model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)



run = 'L1000N1800'
redshift = '0.0'

meas_file = (
    f'/disks/shear16/herle/correlations1/'
    f'correlations_revolver_10_20Mpc_' + run +
    f'_HYDRO_FIDUCIAL_' + redshift + '_JK.h5'
)

df = pd.read_hdf(meas_file, key='a')

rp_meas = df['wgp_rp'].values
wvp_meas = df['wgp_xip'].values 
err_meas = df['wgp_std_from_cov'].values 

# rp_meas = df['wgp_rp'].values /h
# wvp_meas = df['wgp_xip'].values /h 
# err_meas = df['wgp_std_from_cov'].values /h


mask_VI  = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
mask_VII = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)



def model_wVI(p):

    A1_v = p[0]

    wVI_model = wVI[model_mask_VI].copy()
    #wVI_model *= -1.0*A1_v
    wVI_model *= A1_v

    return model_rp[model_mask_VI], wVI_model

def model_wVII(p):

    A1_v = p[0]
    b_v = p[1]

    wVII_model = wVII[model_mask_VII].copy()
    wVII_model *= A1_v*b_v

    return model_rp[model_mask_VII], wVII_model

A, b = 2.0, 0.8

p = [A, b]

rp_model_VI, wVI_model   = model_wVI(p)      # returns only the VI part
rp_model_VII, wVII_model = model_wVII(p)     # returns only the VII part

print("-----------------------------------")

print("rp_model_VI range: ", rp_model_VI.min(), rp_model_VI.max())
print("rp_model_VII range: ", rp_model_VII.min(), rp_model_VII.max())

print("Measurement rp range: ", rp_meas.min(), rp_meas.max())

print("-----------------------------------")
print("A_v: ", A)
print("b_v: ", b)

# print("model wv1 :" , wVI_model)
# print("model wv2 :" , wVII_model)

# concatenate model pieces into one vector (matching data vector order)
rp_model_total = np.concatenate([rp_model_VI, rp_model_VII])
corr_model_total = np.concatenate([wVI_model, wVII_model])



plt.figure(figsize=(10, 10))


plt.plot(rp_model_VII, rp_model_VII*wVII_model, alpha=0.2, ls='--')
plt.plot(rp_model_VI,  rp_model_VI*wVI_model,  alpha=0.2, ls='--')

plt.errorbar(
    rp_meas,
    rp_meas*wvp_meas,
    yerr=rp_meas*err_meas,
    fmt='o',
    capsize=3,
    label='Zobov'
)

plt.ylim(-7, 7)

plt.xscale('log')
plt.xlabel(r'$r_p$ [Mpc/h]')
plt.ylabel(r'$r_p w_{\mathrm{V}+} [\mathrm{Mpc}^2/h^2]$')

ax = plt.gca()
# correct_tick_marks(ax)

mask_model_VI  = (model_rp >= lower_rp_VI)  & (model_rp <= upper_rp_VI)
mask_model_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

line_VII, = plt.plot(
    rp_model_VII,
    rp_model_VII*wVII_model,
    label='2-void'
)

line_VI, = plt.plot(
    rp_model_VI,
    rp_model_VI*wVI_model,
    label='1-void'
)

plt.axvspan(lower_rp_VII, upper_rp_VII, color=line_VII.get_color(), alpha=0.08, zorder=0)
plt.axvspan(lower_rp_VI, upper_rp_VI,  color=line_VI.get_color(),  alpha=0.08, zorder=0)



plt.hlines(0, xmin=1, xmax=600, colors='k', linestyles='dotted')

plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/zobov_fiducial_fitby_hand_fiducial_flamingo_ps.png')
