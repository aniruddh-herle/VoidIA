import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pyccl as ccl
from scipy import integrate
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import savgol_filter
from hankel import HankelTransform


import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt
import glob
import pandas as pd
from nautilus import Sampler, Prior  # <<< moved Prior import here
from multiprocessing import Pool


import argparse

import os
os.environ["OMP_NUM_THREADS"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('-box_size', type=float, default=1000, help='Size of the box.')
parser.add_argument('-n_cores', type=int, default=192, help='Number of CPU cores to use for multiprocessing.')
args = parser.parse_args()

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

output_dir = "/disks/shear16/herle/code/temp_data/"

om = 0.306
h = 0.681

A1 = 1
C1 = 5e-14

rp_values = np.logspace(-1, np.log10(600), 50)

kperp_min, kperp_max = 1e-5, 1e2
kz_min, kz_max = 1e-5, 1e2

PIMAX = 100

delta_v = np.load("/disks/shear16/herle/code/temp_data/zobov_fiducial_void_delta_k_bin0_raw.npy")

N = delta_v.shape[0]
k_native_hMpc = np.logspace(-5, 2, N)  # h/Mpc, matches delta_v point by point

assert delta_v.shape[0] == k_native_hMpc.shape[0], (
    f"delta_v has {delta_v.shape[0]} points but k_native_hMpc has "
    f"{k_native_hMpc.shape[0]}"
)

logk = np.log10(k_native_hMpc)

# Smooth on a uniform log-k grid, then interpolate back

N_log = 20000
logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

delta_uniform = interp1d(logk, delta_v, kind="linear")(logk_uniform)

window_length = 1001
if window_length % 2 == 0:
    window_length += 1
polyorder = 3

delta_uniform_smooth = savgol_filter(delta_uniform, window_length, polyorder)

interp_back = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")
dk_smooth = interp_back(logk)

_delta_v_spline = PchipInterpolator(logk, dk_smooth)

def Delta_v(k_hMpc):
    return _delta_v_spline(np.log10(k_hMpc))


cosmo_ccl = ccl.Cosmology(
    Omega_c=om - 0.0486,
    Omega_b=0.0486,
    Omega_k=0.0,
    h=h,
    A_s=2.099e-9,
    n_s=0.967,
    w0=-1.0,
    wa=0.0,
)

def mean_density(z, omega_m=om):
    cosmo = FlatLambdaCDM(H0=100, Om0=omega_m)
    rho = cosmo.critical_density(z) * cosmo.Om(z)
    return rho.to(u.Msun / u.Mpc**3).value

prefactor = A1 * C1 * mean_density(0) / np.pi**2

def kz_integral(kperp):
    def integrand(kz):
        k = np.sqrt(kz**2 + kperp**2)
        return kperp**3 / (k**2 * kz) * Delta_v(k)

    value, _ = integrate.quad(
        integrand, kz_min, kz_max,
        weight="sin", wvar=PIMAX,
        limit=200,
    )
    return value

def compute_w(rp_values, N=400, hstep=1e-3):
    kperp_grid = np.logspace(np.log10(kperp_min), np.log10(kperp_max), 300)
    F_vals = np.array([kz_integral(kp) for kp in kperp_grid])
    F_spline = PchipInterpolator(kperp_grid, F_vals, extrapolate=False)

    def g(kperp):
        kperp = np.atleast_1d(kperp)
        out = np.zeros_like(kperp)
        mask = (kperp >= kperp_min) & (kperp <= kperp_max)
        out[mask] = F_spline(kperp[mask]) / kperp[mask]
        return out

    ht = HankelTransform(nu=2, N=N, h=hstep)
    w_values = ht.transform(g, rp_values, ret_err=False)

    return prefactor * w_values

print("Computing the 1-void term...")
w_values = compute_w(rp_values)
np.savetxt(output_dir + "th_flams_th.txt", w_values)


## gnat implementation for 1 void term

redshift_list = np.array([0])
kmin, kmax = 1e-5, 1e2
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )

    pk_input[:, 0, idx_z] = -np.pi**2*pkmm_ccl

# Then, we initialise the power spectra in the predictor
predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)

#delta_v = np.load(f'/disks/shear16/herle/code/temp_data/zobov_fiducial_void_delta_k_bin0_raw.npy')
k_native = np.logspace(-5, 2, 1000001)   # h/Mpc

h = 0.681
k_void = k_native                      # Mpc^-1, matches delta_v's saved values

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

window_length = 1001
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
    rp_support=rp_values,
    beta_shape=0,
    beta_density=0,
    pi_max=PIMAX,
)

w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
    rp_support=rp_values,
    r_support=rp_values,
)

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo_ccl['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo_ccl, scale_factor)


w_1v = scaling * w_arr[:, 0, 0]


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

# Top panel
ax1.plot(rp_values, w_values, color="blue", label="Pauline")
ax1.plot(rp_values, w_1v, color="crimson", label="GNAT")
ax1.axhline(0.0, color="gray", linestyle="--")
ax1.set_xscale("log")
ax1.set_ylabel(r"$w_{v+}(r_p)$")

ax1.legend()

# Bottom panel: ratio
ratio = w_1v / w_values

ax2.plot(rp_values, ratio, color="black")
ax2.axhline(1.0, color="gray", linestyle="--")
ax2.set_ylim(-2,2)
ax2.set_xscale("log")
ax2.set_xlabel(r"$r_p$ [Mpc/h]")
ax2.set_ylabel("Ratio")

plt.savefig("/disks/shear16/herle/code/plots/pauline_1void_comp.png", dpi=300, bbox_inches="tight")
