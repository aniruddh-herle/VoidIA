# ==========================================================
# IMPORTS
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy import constants, integrate, special
from joblib import Parallel, delayed

import astropy.units as u
import pyccl as ccl

from astropy.cosmology import FlatLambdaCDM

# ==========================================================
# CONFIGURATION
# ==========================================================

output_dir = "/disks/shear16/herle/code/"


onevoids = True
twovoids = True
compute_th = True

# ----------------------------------------------------------
# Cosmology
# ----------------------------------------------------------

om = 0.306
h = .681

cosmo_astropy = FlatLambdaCDM(H0=100 * h, Om0=om)

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
    n_s=0.967
)

# ----------------------------------------------------------
# IA / normalization
# ----------------------------------------------------------

A1 = 1
C1 = 5e-14


# ----------------------------------------------------------
# rp grids
# ----------------------------------------------------------

rp_values2v = np.logspace(-1, np.log10(600), 20)


# ----------------------------------------------------------
# k integration limits
# ----------------------------------------------------------
kmin = 1e-5
kmax = 1e2

kperp_min = kmin
kperp_max = kmax

kz_min = kmin
kz_max = kmax


# ----------------------------------------------------------
# LOS integration
# ----------------------------------------------------------

PIMAX = 100


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

outfilename = "flams_th.txt"

# ==========================================================
# USEFUL FUNCTIONS
# ==========================================================

def pgz(z, sigma=0.1):

    norm = 0.5

    return (
        (1 / norm)
        * (1 / (sigma * np.sqrt(2*np.pi)))
        * np.exp(-0.5 * (z / sigma)**2)
        * (z >= 0)
    )


def pvz(z, sigma=0.1):

    norm = 0.5

    return (
        (1 / norm)
        * (1 / (sigma * np.sqrt(2*np.pi)))
        * np.exp(-0.5 * (z / sigma)**2)
        * (z >= 0)
    )


def Plin(kz, kperp, z):

    k = np.sqrt(kz**2 + kperp**2)

    return cosmo_ccl.linear_power(k, 1 / (1 + z))


def Pnl(kz, kperp, z):

    k = np.sqrt(kz**2 + kperp**2)

    return cosmo_ccl.nonlin_power(k, 1 / (1 + z))


def dxdz(z):

    hh = 1e-5

    return (
        ccl.comoving_radial_distance(cosmo_ccl, 1/(1+z+hh))
        - ccl.comoving_radial_distance(cosmo_ccl, 1/(1+z))
    ) / hh


def int2(z):

    chi = ccl.comoving_radial_distance(cosmo_ccl, 1/(1+z))

    return pvz(z) * pgz(z) / (chi**2 * dxdz(z))

def W(z):

    return 1


def D(z):

    return ccl.growth_factor(cosmo_ccl, 1/(1+z))


def mean_density_h2Mpc(z, omega_m=om, h=h):

    cosmo = FlatLambdaCDM(H0=100, Om0=omega_m)

    rho_matter = cosmo.critical_density(z) * cosmo.Om(z)

    return rho_matter.to(u.Msun / u.Mpc**3).value


# ==========================================================
# 2-VOID TERM
# ==========================================================

def integrand_k_2v(kz, kperp, rp):

    k = np.sqrt(kz**2 + kperp**2)

    if k == 0 or kz == 0:
        return 0

    term1 = kperp**3 / (k**2 * kz)

    term2 = Plin(kz, kperp, 0)

    term3 = np.sin(kz * PIMAX)

    term4 = special.jv(2, kperp * rp)

    return term1 * term2 * term3 * term4


def integrate_k_2v(rp):

    kperp_limits_local = (kperp_min, kperp_max)
    kz_limits_local = (kz_min, kz_max)

    def integrand_2v(kperp, kz):

        return integrand_k_2v(kz, kperp, rp)

    integral_k, _ = integrate.nquad(
        integrand_2v,
        [kperp_limits_local, kz_limits_local],
        opts={
            'limit': 20,
            'epsabs': 1e-4,
            'epsrel': 1e-4
        }
    )

    return integral_k


def w_v_plus_2v(rp):

    prefactor = (
        A1
        * C1
        * mean_density_h2Mpc(0)
        / (np.pi**2)
    )

    #integral_z = prefactor * integrate_k_2v(rp)
    integral_z = integrate_k_2v(rp)

    return integral_z


prefactor = (
        A1
        * C1
        * mean_density_h2Mpc(0)
        / (np.pi**2)
    )


# ==========================================================
# THEORY COMPUTATION
# ==========================================================

if compute_th:


    if twovoids:

        print("COMPUTE THE 2 VOID TERM")

        w_values_2v = Parallel(n_jobs=60)(
            delayed(w_v_plus_2v)(rp)
            for rp in rp_values2v
        )

        np.savetxt(
            output_dir + 'th_2voids_' + outfilename,
            w_values_2v
        )

# ==========================================================
# PLOT THEORY
# ==========================================================

# plt.figure(figsize=(10, 6))


# if twovoids:

#     w_values_2v = np.loadtxt(
#         output_dir + 'th_2voids_' + outfilename
#     )

#     plt.plot(
#         rp_values2v,
#         w_values_2v,
#         color='navy',
#         label='2-void'
#     )

# plt.xlabel(r'$r_p$')
# plt.ylabel(r'$w_{v+}(r_p)$')


# plt.xscale('log')

# plt.legend()

# plt.savefig(
#     '/disks/shear16/herle/code/plots/w_v_plus_rp_plot.png',
#     bbox_inches='tight'
# )


base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0_trunckrange3/'

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = (C1rhocrit*cosmo_ccl['Omega_m'])/((np.pi**2)*ccl.growth_factor(cosmo_ccl, scale_factor))


# wVII = scaling*np.load(base_folder+'/wvII.npy')
wVII = np.load(base_folder+'/wvII.npy')
wVII/=h


model_rp = np.loadtxt(base_folder+'/w_gp_z0.0.txt')[:, 0] 
model_rp/=h
print("pauline rp: ", rp_values2v)
print("model rp: ", model_rp)


# Limit number of cores
import os
os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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

# Define redshifts at which to calculate estimators
redshift_list = np.array([0])

# Define scales at which to calculate estimators
rp_array = rp_values2v
r_array = rp_array

# Pi_max for projection integrals
pi_max = PIMAX  # in Mpc


# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

# Fill pk_input with g+ and gg power spectra
powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='linear_matter'
    )

    pk_input[:, 0, idx_z] = -np.pi**2*pkmm_ccl

# Then, we initialise the power spectra in the predictor
predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)
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


w_fftlog =  w_arr[:, 0, 0]



fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 10),
    sharex='col',
    gridspec_kw={'height_ratios': [2, 1]}  
)

# Top row = main plot, bottom row = residuals
ax_main = axes[0]
ax_ratio  = axes[1]

ax_main.plot(rp_values2v, w_values_2v, color='navy', label='Pauline')
ax_main.plot(model_rp, wVII, color='red', label='Aniruddh')
ax_main.plot(rp_array, w_fftlog, color='green', label='GNAT')
ax_main.set_xlabel(r'$r_p [Mpc/h]$')
ax_main.set_ylabel(r'$w_{v+}(r_p)$')
ax_main.set_title(
    f'Theory $w_{{v+}}(r_p)$, k-range {kmin} to {kmax}'
)
ax_main.set_xscale('log')
ax_main.set_yscale('log')
ax_main.legend(frameon=False)

ratio = w_fftlog / wVII

ax_ratio.plot(rp_values2v, ratio, color='red')

ratio = w_fftlog / w_fftlog

ax_ratio.plot(rp_values2v, ratio, color='green')

ratio = w_fftlog / w_values_2v

ax_ratio.plot(rp_values2v, ratio, color='navy')

ax_ratio.set_ylim(0.95, 1.05)
ax_ratio.set_xlabel(r'$r_p [Mpc/h]$')
ax_ratio.set_ylabel(r'Ratio')
ax_ratio.set_xscale('log')


plt.savefig('/disks/shear16/herle/code/plots/2void_term_comp.png')
