import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt
import glob
import pandas as pd
from nautilus import Sampler, Prior
from multiprocessing import Pool
from mcfit import P2xi

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

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486,
    Omega_b=0.0486,
    h=0.681,
    sigma8=0.807,
    n_s=0.967
)

# Define redshifts at which to calculate estimators
redshift_list = np.array([0])

kmin, kmax = 1e-5, 1e2

rp_values2v = np.logspace(-1, np.log10(900), 200)

# Define scales at which to calculate estimators
rp_array = rp_values2v
r_array = rp_array

# Pi_max for projection integrals
pi_max = 100.0  # in Mpc/h

k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))

powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )
    pk_input[:, 0, idx_z] = -np.pi**2 * pkmm_ccl

predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)

# 2 void term
predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input,
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

w_2v = w_arr[:, 0, 0]

h = 0.681

modes = ['FIDUCIAL']

k_native = np.logspace(-5, 2, 1000001)   # h/Mpc

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0 / (1.0 + z_i)
scaling = C1rhocrit * cosmo_ccl['Omega_m'] / (np.pi**2) / ccl.growth_factor(cosmo_ccl, scale_factor)

Rv = 8.0
lower_rp_VI  = 6.0
upper_rp_VI  = 2.5 * Rv
lower_rp_VII = 4.0 * Rv
upper_rp_VII = 500.0

from getdist import plots, MCSamples

def compute_chi2(logl, samples, corr):
    logL_max = np.max(logl)
    chi2_eff = -2 * logL_max
    Ndata = len(corr)
    Nparams_flat = samples.shape[1]
    Ndof = Ndata - Nparams_flat
    chi2_red = chi2_eff / Ndof
    return chi2_eff, chi2_red

def read_nautilus_samples(npz_filename):
    data = np.load(npz_filename)
    samples = data['samples']
    weights = data['weights']
    logl    = data['logl']
    return samples, weights, logl

names  = ['A1_v', 'A2_v', 'b_v']
labels = ['A1_v', 'A2_v', 'b_v']

# -------------------------------------------------
# Precompute xi_mm once
# -------------------------------------------------
z_target = 0.0
a_target = 1.0 / (1.0 + z_target)

k_hmpc   = np.geomspace(1e-5, 1e3, 8192)
pk_ccl_mm = ccl.nonlin_matter_power(cosmo_ccl, k_hmpc * h, a_target)
pk_hmpc  = pk_ccl_mm * h**3

r_mm, xi_mm = P2xi(k_hmpc, lowring=True)(pk_hmpc, extrap=True)

keep   = (r_mm >= 0.1) & (r_mm <= 300.0)
r_mm   = r_mm[keep]
xi_mm  = xi_mm[keep]

xi_mm_interp = interp1d(r_mm, xi_mm, kind='cubic',
                         bounds_error=False, fill_value=0.0)

# =================================================================
# Main loop over modes
# =================================================================
for i in range(len(modes)):

    print("-----------------------------")
    print(modes[i])
    print("-----------------------------")

    # -------------------------------------------------
    # 1-void term: load and smooth delta_v
    # -------------------------------------------------
    delta_v = np.load(f'/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin0_raw_particles.npy')

    k_void = k_native   # h/Mpc — same grid delta_v was saved on

    assert delta_v.shape[0] == k_void.shape[0], \
        f"delta_v has {delta_v.shape[0]} points but k_void has {k_void.shape[0]}"

    logk_void   = np.log10(k_void)
    logk_target = np.log10(k_array_single)

    interp_to_target = interp1d(
        logk_void, delta_v, kind="linear",
        bounds_error=False, fill_value=0.0
    )
    delta_v_on_target = interp_to_target(logk_target)

    logk = logk_target
    N_log = 20000
    logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

    interp_func  = interp1d(logk, delta_v_on_target, kind="linear")
    delta_uniform = interp_func(logk_uniform)

    window_length = 3001
    polyorder = 3
    if window_length % 2 == 0:
        window_length += 1

    delta_uniform_smooth = savgol_filter(delta_uniform, window_length, polyorder)

    interp_back   = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")
    delta_v_smoothed = interp_back(logk)

    pk_input_1v = np.zeros((len(k_array_single), 1, len(redshift_list)))
    for idx_z, z in enumerate(redshift_list):
        pk_input_1v[:, 0, idx_z] = -np.pi**2 * delta_v_smoothed

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

    wVII = w_2v.copy() * scaling
    wVI  = w_1v.copy() * scaling

    model_rp = rp_array

    model_mask_VI  = (model_rp >= lower_rp_VI)  & (model_rp <= upper_rp_VI)
    model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

    run      = 'L1000N1800'
    redshift = '0.0'

    # -------------------------------------------------
    # Load measurements and JK realizations
    # -------------------------------------------------
    meas_file = (
        f'/disks/shear16/herle/correlations1/'
        f'correlations_joint_revolver_20_30hMpc_{run}_HYDRO_{modes[i]}_{redshift}_env_dep_analytical.h5'
    )

    wvp_JK_measured = np.load(
        f'/disks/shear16/herle/correlations1/'
        f'wvp_xip_jk_voids_joint_revolver_0_10hMpc_{run}_HYDRO_{modes[i]}_{redshift}_env_dep_analytical.npy'
    )

    xi_vv_JK_measured = np.load(
        f'/disks/shear16/herle/correlations1/'
        f'xi_vv_jk_voids_joint_revolver_0_10hMpc_{run}_HYDRO_{modes[i]}_{redshift}_env_dep_analytical.npy'
    )

    df = pd.read_hdf(meas_file, key='a')

    rp_meas  = df['wvp_rp'].values
    wvp_meas = df['wvp_xip'].values

    rp_xi_vv  = df['xi_vv_r'].values
    xi_vv     = df['xi_vv'].values
    xi_vv_err = df['xi_vv_err'].values

    # -------------------------------------------------
    # Build masks
    # -------------------------------------------------
    mask_VI    = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
    mask_VII   = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)
    mask_total = mask_VI | mask_VII

    rp_cut = rp_meas[mask_total]
    corr   = wvp_meas[mask_total]

    xi_rmin = 50.0
    xi_rmax = 200

    mask_xi       = (rp_xi_vv >= xi_rmin) & (rp_xi_vv <= xi_rmax)
    rp_xi_cut     = rp_xi_vv[mask_xi]
    xi_vv_cut     = xi_vv[mask_xi]
    xi_vv_err_cut = xi_vv_err[mask_xi]

    wvp_JK_cut   = wvp_JK_measured[:, mask_total]
    xi_vv_JK_cut = xi_vv_JK_measured[:, mask_xi]

    # -------------------------------------------------
    # Joint covariance
    # -------------------------------------------------
    wvp_mean   = np.mean(wvp_JK_cut,   axis=0)
    xi_vv_mean = np.mean(xi_vv_JK_cut, axis=0)

    wvp_diff   = pd.DataFrame(wvp_JK_cut   - wvp_mean,   columns=rp_cut)
    xi_vv_diff = pd.DataFrame(xi_vv_JK_cut - xi_vv_mean, columns=rp_xi_cut)

    corr_diff  = pd.concat([wvp_diff, xi_vv_diff], axis=1)
    all_scales = np.concatenate([rp_cut, rp_xi_cut])

    NPatches = wvp_JK_cut.shape[0]

    cov_mat = ((NPatches - 1) / NPatches) * np.sum(
        np.einsum('ij,ik->ijk', corr_diff.values, corr_diff.values),
        axis=0
    )

    cov_mat = pd.DataFrame(data=cov_mat, index=all_scales, columns=all_scales)
    cov_arr = cov_mat.values
    inv_cov = np.linalg.pinv(cov_arr)

    corr_joint = np.concatenate([corr, xi_vv_cut])

    assert wvp_JK_cut.shape[0] == xi_vv_JK_cut.shape[0], \
        "wvp and xi_vv JK samples have different numbers of patches"
    assert wvp_JK_cut.shape[0] == NPatches

    # -------------------------------------------------
    # Model functions
    # -------------------------------------------------
    def model_wVI(p):
        A1_v = p[0]
        wVI_model = wVI[model_mask_VI].copy()
        wVI_model *= A1_v
        return model_rp[model_mask_VI], wVI_model

    def model_wVII(p):
        A2_v = p[1]
        b_v  = p[2]
        wVII_model = wVII[model_mask_VII].copy()
        wVII_model *= A2_v * b_v
        return model_rp[model_mask_VII], wVII_model

    def model_xi_vv(p):
        b_v = p[2]
        return rp_xi_cut, b_v**2 * xi_mm_interp(rp_xi_cut)

    def log_likelihood(p):
        rp_model_VI,  wVI_model  = model_wVI(p)
        rp_model_VII, wVII_model = model_wVII(p)

        rp_model_total   = np.concatenate([rp_model_VI,  rp_model_VII])
        corr_model_total = np.concatenate([wVI_model,    wVII_model])

        wvp_model_interpol = np.interp(rp_cut, rp_model_total, corr_model_total)

        _, xi_vv_model = model_xi_vv(p)

        model_joint = np.concatenate([wvp_model_interpol, xi_vv_model])

        delta = corr_joint - model_joint
        chisq = delta.dot(inv_cov.dot(delta))

        ll = -0.5 * chisq
        if np.isnan(ll):
            return -np.inf, chisq
        return ll, chisq

    # -------------------------------------------------
    # Prior and sampler
    # -------------------------------------------------
    prior_obj = Prior()

    param_names = ['A1_v', 'A2_v', 'b_v']
    prior_obj.add_parameter('A1_v', dist=(-10.0, 10.0))
    prior_obj.add_parameter('A2_v', dist=(-10.0, 10.0))
    prior_obj.add_parameter('b_v',  dist=(-10.0, 0.0))

    def loglikelihood_fn(p_dict):
        p = np.array([p_dict[key] for key in param_names])
        logl, _ = log_likelihood(p)
        return logl

    box_size = args.box_size
    n_cores  = args.n_cores

    def run_sampler():
        n_live = 3000
        output_path = '/disks/shear16/herle/models/models/'
        filename = output_path + f"nautilus_chain_revolver_{modes[i]}_voids_{box_size}_Rv20_30Mpc_env_dep_analytical.h5"

        sampler = Sampler(
            prior_obj,
            loglikelihood_fn,
            n_live=n_live,
            pool=n_cores if n_cores > 1 else None,
            filepath=filename,
            resume=False,
            n_networks=16,
        )

        sampler.run(verbose=True, discard_exploration=True)
        points, log_w, log_l = sampler.posterior()
        weights = np.exp(log_w)
        logz    = sampler.log_z
        output_file = (output_path +
                       f"nautilus_chain_revolver_{modes[i]}_voids_{box_size}_Rv20_30Mpc_env_dep_analytical.npz")

        np.savez(output_file,
                 samples=points,
                 logl=log_l,
                 weights=weights,
                 logz=logz)

        print(f"Saved results to {output_file}")

    # # ============================================================
    # # DIAGNOSTIC PLOTS / CHECKS FOR JOINT DATA + COVARIANCE
    # # ============================================================

    # import matplotlib.pyplot as plt
    # import numpy as np

    # print("\n" + "="*70)
    # print("JOINT DATA / COVARIANCE DIAGNOSTICS")
    # print("="*70)

    # # ------------------------------------------------------------
    # # Basic dimensions
    # # ------------------------------------------------------------

    # N_wvp = len(rp_cut)
    # N_xi  = len(rp_xi_cut)
    # N_tot = len(corr_joint)

    # print(f"Number of wvp points       : {N_wvp}")
    # print(f"Number of xi_vv points     : {N_xi}")
    # print(f"Total joint data points    : {N_tot}")
    # print(f"Covariance shape            : {cov_arr.shape}")
    # print(f"Inverse covariance shape    : {inv_cov.shape}")
    # print(f"Number of JK patches        : {NPatches}")

    # assert corr_joint.shape[0] == cov_arr.shape[0]
    # assert cov_arr.shape[0] == cov_arr.shape[1]

    # # ------------------------------------------------------------
    # # Basic value ranges
    # # ------------------------------------------------------------

    # print("\nData ranges:")
    # print(f"wvp     : {np.nanmin(corr[:N_wvp]): .5e} "
    #     f"to {np.nanmax(corr[:N_wvp]): .5e}")
    # print(f"xi_vv   : {np.nanmin(xi_vv_cut): .5e} "
    #     f"to {np.nanmax(xi_vv_cut): .5e}")

    # print("\nCovariance:")
    # print(f"min     : {np.nanmin(cov_arr): .5e}")
    # print(f"max     : {np.nanmax(cov_arr): .5e}")
    # print(f"diag min: {np.nanmin(np.diag(cov_arr)): .5e}")
    # print(f"diag max: {np.nanmax(np.diag(cov_arr)): .5e}")

    # # ------------------------------------------------------------
    # # Symmetry check
    # # ------------------------------------------------------------

    # symmetry_error = np.max(np.abs(cov_arr - cov_arr.T))

    # print("\nCovariance symmetry:")
    # print(f"max |C - C.T| = {symmetry_error:.5e}")

    # # ------------------------------------------------------------
    # # Positive-definiteness / eigenvalues
    # # ------------------------------------------------------------

    # eigvals = np.linalg.eigvalsh(cov_arr)

    # print("\nCovariance eigenvalues:")
    # print(f"minimum eigenvalue = {eigvals.min():.5e}")
    # print(f"maximum eigenvalue = {eigvals.max():.5e}")
    # print(f"number <= 0        = {np.sum(eigvals <= 0)}")

    # if eigvals.min() > 0:
    #     print("Covariance is positive definite.")
    # else:
    #     print("WARNING: covariance has non-positive eigenvalues.")

    # # ------------------------------------------------------------
    # # Condition number
    # # ------------------------------------------------------------

    # cond_number = np.linalg.cond(cov_arr)

    # print(f"\nCovariance condition number = {cond_number:.5e}")

    # # ------------------------------------------------------------
    # # Standard deviations from covariance
    # # ------------------------------------------------------------

    joint_err = np.sqrt(np.diag(cov_arr))

    # print("\nTypical fractional errors:")
    # print(f"wvp   median |sigma / data| = "
    #     f"{np.median(np.abs(joint_err[:N_wvp] / corr[:N_wvp])):.3f}")
    # print(f"xi_vv median |sigma / data| = "
    #     f"{np.median(np.abs(joint_err[N_wvp:] / xi_vv_cut)):.3f}")


    # # ============================================================
    # # PLOT 1: JOINT DATA VECTOR
    # # ============================================================

    # fig, ax = plt.subplots(figsize=(9, 5))

    # ax.errorbar(
    #     rp_cut,
    #     corr[:N_wvp],
    #     yerr=joint_err[:N_wvp],
    #     fmt='o',
    #     ms=4,
    #     label=r'$w_{\rm VP}$'
    # )

    # ax.errorbar(
    #     rp_xi_cut,
    #     xi_vv_cut,
    #     yerr=joint_err[N_wvp:],
    #     fmt='s',
    #     ms=4,
    #     label=r'$\xi_{\rm vv}$'
    # )

    # # Mark the transition between the two data-vector blocks
    # ax.axvline(
    #     rp_cut.max(),
    #     linestyle='--',
    #     alpha=0.5
    # )

    # ax.set_xscale('log')
    # ax.set_yscale('symlog', linthresh=1e-4)

    # ax.set_xlabel(r'$r_p\ [{\rm Mpc}/h]$')
    # ax.set_ylabel('Correlation')
    # ax.set_title('Joint data vector')
    # ax.legend()

    # plt.tight_layout()
    # plt.savefig('/disks/shear16/herle/code/plots/joint_data_vector.png')


    # # ============================================================
    # # PLOT 2: COVARIANCE MATRIX
    # # ============================================================

    # fig, ax = plt.subplots(figsize=(7, 6))

    # # Symmetric logarithmic normalization makes both small
    # # off-diagonal terms and large diagonal terms visible.
    # from matplotlib.colors import SymLogNorm

    # max_abs_cov = np.max(np.abs(cov_arr))

    # im = ax.imshow(
    #     cov_arr,
    #     origin='lower',
    #     aspect='auto',
    #     cmap='coolwarm',
    #     norm=SymLogNorm(
    #         linthresh=max_abs_cov * 1e-4,
    #         vmin=-max_abs_cov,
    #         vmax=max_abs_cov
    #     )
    # )

    # # Boundary between wvp and xi_vv
    # ax.axhline(N_wvp - 0.5, color='black', linestyle='--', alpha=0.7)
    # ax.axvline(N_wvp - 0.5, color='black', linestyle='--', alpha=0.7)

    # ax.set_xlabel('Data-vector index')
    # ax.set_ylabel('Data-vector index')
    # ax.set_title('Joint covariance matrix')

    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label(r'Covariance')

    # plt.tight_layout()
    # plt.savefig('/disks/shear16/herle/code/plots/joint_covariance.png')


    # # ============================================================
    # # PLOT 3: CORRELATION MATRIX
    # # ============================================================

    # sigma = np.sqrt(np.diag(cov_arr))

    # corr_mat = cov_arr / np.outer(sigma, sigma)

    # fig, ax = plt.subplots(figsize=(7, 6))

    # im = ax.imshow(
    #     corr_mat,
    #     origin='lower',
    #     aspect='auto',
    #     cmap='coolwarm',
    #     vmin=-1,
    #     vmax=1
    # )

    # # Boundary between wvp and xi_vv
    # ax.axhline(N_wvp - 0.5, color='black', linestyle='--', alpha=0.7)
    # ax.axvline(N_wvp - 0.5, color='black', linestyle='--', alpha=0.7)

    # ax.set_xlabel('Data-vector index')
    # ax.set_ylabel('Data-vector index')
    # ax.set_title('Joint correlation matrix')

    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Correlation coefficient')

    # plt.tight_layout()
    # plt.savefig('/disks/shear16/herle/code/plots/correlation_matrix.png')


    # # ============================================================
    # # CROSS-COVARIANCE BLOCK DIAGNOSTIC
    # # ============================================================

    # cross_cov = cov_arr[:N_wvp, N_wvp:]

    # cross_corr = corr_mat[:N_wvp, N_wvp:]

    # print("\nCross-covariance block:")
    # print(f"shape                 = {cross_cov.shape}")
    # print(f"min covariance        = {cross_cov.min():.5e}")
    # print(f"max covariance        = {cross_cov.max():.5e}")
    # print(f"max |correlation|     = {np.max(np.abs(cross_corr)):.5f}")
    # print(f"median correlation    = {np.median(cross_corr):.5f}")

    # print("="*70 + "\n")

    # ============================================================
    # PLOT 5: xi_vv AGAINST MATTER TEMPLATE + RATIO
    # ============================================================

    # # Matter correlation over the full measured xi_vv range
    # xi_mm_full = xi_mm_interp(rp_xi_vv)

    # # Matter correlation over the fitting range
    # xi_mm_cut = xi_mm_interp(rp_xi_cut)

    # fig, (ax1, ax2) = plt.subplots(
    #     2, 1,
    #     figsize=(8, 8),
    #     sharex=True,
    #     gridspec_kw={'height_ratios': [2, 1]}
    # )

    # # ============================================================
    # # TOP: xi_vv and xi_mm
    # # ============================================================

    # # Full xi_vv data
    # ax1.errorbar(
    #     rp_xi_vv,
    #     (rp_xi_vv**2)*xi_vv,
    #     yerr=(rp_xi_vv**2)*xi_vv_err,
    #     fmt='o',
    #     ms=4,
    #     alpha=0.7,
    #     label=r'Measured $\xi_{\rm vv}$'
    # )

    # # Full matter correlation
    # ax1.plot(
    #     rp_xi_vv,
    #     (rp_xi_vv**2)*xi_mm_full,
    #     '--',
    #     label=r'$\xi_{\rm mm}$'
    # )

    # # Highlight the fitting range
    # ax1.axvspan(
    #     xi_rmin,
    #     xi_rmax,
    #     alpha=0.15,
    #     label=f'Fit range'
    # )

    # ax1.axhline(
    #     0,
    #     color='black',
    #     linestyle=':',
    #     alpha=0.5
    # )

    # #ax1.set_yscale('log')
    # ax1.set_xscale('log')
    # ax1.set_ylabel(r'$r^2 \times \xi(r)$')
    # ax1.set_title(r'Void correlation vs. matter correlation')
    # ax1.legend()


    # # ============================================================
    # # BOTTOM: ratio xi_vv / xi_mm
    # # ============================================================

    # # ============================================================
    # # BOTTOM: ratio xi_vv / xi_mm
    # # ============================================================

    # ratio = xi_vv / xi_mm_full

    # # Propagated uncertainty.
    # # xi_mm is a theoretical template, so only xi_vv contributes.
    # ratio_err = xi_vv_err / np.abs(xi_mm_full)

    # ax2.errorbar(
    #     rp_xi_vv,
    #     ratio,
    #     yerr=ratio_err,
    #     fmt='o-',
    #     ms=4,
    #     capsize=3
    # )

    # # Fitting range
    # ax2.axvspan(
    #     xi_rmin,
    #     xi_rmax,
    #     alpha=0.15
    # )

    # # Reference line
    # ax2.axhline(
    #     1,
    #     color='black',
    #     linestyle='--',
    #     alpha=0.6
    # )

    # ax2.set_xscale('log')
    # ax2.set_xlabel(r'$r\ [{\rm Mpc}/h]$')
    # ax2.set_ylabel(r'$\xi_{\rm vv}/\xi_{\rm mm}$')

    # plt.tight_layout()

    # plt.savefig(
    #     '/disks/shear16/herle/code/plots/xi_vv_xi_mm.png',
    #     dpi=200,
    #     bbox_inches='tight')

    # print("\nxi_vv / xi_mm diagnostics:")
    # print(f"Full xi_vv range: {rp_xi_vv.min():.2f} - {rp_xi_vv.max():.2f} Mpc/h")
    # print(f"Fit xi_vv range : {rp_xi_cut.min():.2f} - {rp_xi_cut.max():.2f} Mpc/h")

    # print("\nRatio in fitting range:")
    # ratio_cut = xi_vv_cut / xi_mm_cut

    # for r, rat in zip(rp_xi_cut, ratio_cut):
    #     print(f"r = {r:7.2f}  xi_vv/xi_mm = {rat: .4f}")

    # print(f"\nMedian ratio in fit range = {np.median(ratio_cut):.4f}")

    if __name__ == '__main__':
        run_sampler()

    # -------------------------------------------------
    # Load posterior
    # -------------------------------------------------
    filename = f'/disks/shear16/herle/models/models/nautilus_chain_revolver_{modes[i]}_voids_{box_size}_Rv20_30Mpc_env_dep_analytical.npz'
    samples, weights, logl = read_nautilus_samples(filename)

    getdist = MCSamples(samples=samples,
                        weights=weights,
                        names=names,
                        labels=labels,
                        ignore_rows=0)

    chi2_eff, chi2_red = compute_chi2(logl, samples, corr_joint)
    print(f"[{modes[i]}] Reduced Chi2: ", chi2_red)

    weights_norm    = weights / np.sum(weights)
    inds            = np.random.choice(len(samples), size=500, p=weights_norm)
    posterior_draws = samples[inds]

    # -------------------------------------------------
    # Read best-fit parameters
    # -------------------------------------------------
    d = getdist.get1DDensity('A1_v')
    xyind = np.where(d.P == np.amax(d.P))
    A1v = d.x[xyind[0]][0]
    errors_density = d.getLimits(0.68)
    A1v_error = (errors_density[1] - errors_density[0]) / 2.0

    d = getdist.get1DDensity('A2_v')
    xyind = np.where(d.P == np.amax(d.P))
    A2v = d.x[xyind[0]][0]
    errors_density = d.getLimits(0.68)
    A2v_error = (errors_density[1] - errors_density[0]) / 2.0

    d = getdist.get1DDensity('b_v')
    xyind = np.where(d.P == np.amax(d.P))
    bv = d.x[xyind[0]][0]
    errors_density = d.getLimits(0.68)
    bv_error = (errors_density[1] - errors_density[0]) / 2.0

    A1_best = A1v
    A2_best = A2v
    bv_best = bv

    # -------------------------------------------------
    # Best-fit model vectors
    # -------------------------------------------------
    wVI_bestfit  = wVI  * A1_best
    wVII_bestfit = wVII * A2_best * bv_best

    # Posterior draws
    A1_draws = posterior_draws[:, 0]
    A2_draws = posterior_draws[:, 1]
    bv_draws = posterior_draws[:, 2]

    wVI_ens, wVII_ens = [], []
    for A1, A2, bv_d in zip(A1_draws, A2_draws, bv_draws):
        wVI_ens.append(wVI   * A1)
        wVII_ens.append(wVII * A2 * bv_d)

    wVI_ens  = np.array(wVI_ens)
    wVII_ens = np.array(wVII_ens)

    p16_VI,  p50_VI,  p84_VI  = np.percentile(wVI_ens,  [16, 50, 84], axis=0)
    p16_VII, p50_VII, p84_VII = np.percentile(wVII_ens, [16, 50, 84], axis=0)

    # -------------------------------------------------
    # Errors from joint covariance diagonal
    # -------------------------------------------------
    n_wvp   = len(rp_cut)
    err_cut = np.sqrt(np.diag(cov_arr))[:n_wvp]   # wvp part only
    err_xi  = np.sqrt(np.diag(cov_arr))[n_wvp:]   # xi_vv part

    mask_data_VI  = mask_VI[mask_total]
    mask_data_VII = mask_VII[mask_total]

    # -------------------------------------------------
    # Best-fit and posterior band for xi_vv
    # -------------------------------------------------
    _, xi_vv_bestfit = model_xi_vv([A1_best, A2_best, bv_best])

    xi_vv_ens = []
    for A1, A2, bv_d in zip(A1_draws, A2_draws, bv_draws):
        _, xi_vv_draw = model_xi_vv([A1, A2, bv_d])
        xi_vv_ens.append(xi_vv_draw)
    xi_vv_ens = np.array(xi_vv_ens)
    p16_xi, p50_xi, p84_xi = np.percentile(xi_vv_ens, [16, 50, 84], axis=0)

    color = 'k'
    label = modes[i]

    # -------------------------------------------------
    # Figure: 3 rows — wvp main, wvp residual, xi_vv
    # -------------------------------------------------
    fig, (ax_wvp, ax_wvp_res, ax_xi) = plt.subplots(
        3, 1,
        figsize=(10, 14),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 2], "hspace": 0.35}
    )

    # ── wvp main panel ──────────────────────────────
    ax_wvp.plot(
        model_rp[model_mask_VII],
        model_rp[model_mask_VII] * wVII_bestfit[model_mask_VII],
        color=color, label='2-void'
    )
    ax_wvp.plot(
        model_rp[model_mask_VI],
        model_rp[model_mask_VI] * wVI_bestfit[model_mask_VI],
        color=color, ls='--', label='1-void'
    )
    ax_wvp.errorbar(
        rp_cut, rp_cut * corr, yerr=rp_cut * err_cut,
        fmt='o', capsize=3, color=color, label=label
    )
    ax_wvp.fill_between(
        model_rp[model_mask_VI],
        model_rp[model_mask_VI] * p16_VI[model_mask_VI],
        model_rp[model_mask_VI] * p84_VI[model_mask_VI],
        alpha=0.25, color=color
    )
    ax_wvp.fill_between(
        model_rp[model_mask_VII],
        model_rp[model_mask_VII] * p16_VII[model_mask_VII],
        model_rp[model_mask_VII] * p84_VII[model_mask_VII],
        alpha=0.25, color=color
    )
    ax_wvp.axvspan(lower_rp_VI,  upper_rp_VI,  color='grey', alpha=0.08, zorder=0)
    ax_wvp.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)
    ax_wvp.axhline(0, color='k', ls=':')
    ax_wvp.set_xscale('log')
    ax_wvp.set_xlim(1, 600)
    ax_wvp.set_ylabel(r'$r_p w_{\mathrm{V}+}\,[\mathrm{Mpc}^2/h^2]$')
    ax_wvp.legend(frameon=False)

    # ── wvp residuals ────────────────────────────────
    model_at_data = np.full_like(rp_cut, np.nan)
    model_at_data[mask_data_VI] = np.interp(
        rp_cut[mask_data_VI],
        model_rp[model_mask_VI],
        wVI_bestfit[model_mask_VI]
    )
    model_at_data[mask_data_VII] = np.interp(
        rp_cut[mask_data_VII],
        model_rp[model_mask_VII],
        wVII_bestfit[model_mask_VII]
    )
    residual = 100.0 * (corr - model_at_data) / corr
    frac_err = 100.0 * err_cut / np.abs(corr)

    ax_wvp_res.errorbar(
        rp_cut, residual, yerr=frac_err,
        fmt='o', capsize=3, color=color
    )
    ax_wvp_res.axvspan(lower_rp_VI,  upper_rp_VI,  color='grey', alpha=0.08, zorder=0)
    ax_wvp_res.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)
    ax_wvp_res.axhline(0, color='k', ls=':')
    ax_wvp_res.set_xscale('log')
    ax_wvp_res.set_xlim(1, 600)
    ax_wvp_res.set_ylim(-200, 200)
    ax_wvp_res.set_xlabel(r'$r_p\,[\mathrm{Mpc}/h]$')
    ax_wvp_res.set_ylabel(r'$\Delta/\mathrm{Data}\,[\%]$')

    # ── xi_vv panel ──────────────────────────────────
    ax_xi.plot(rp_xi_cut, (rp_xi_cut**2)*xi_vv_bestfit, color=color, label=r'$b_v^2\,\xi_{mm} \times r^2$')
    ax_xi.fill_between(rp_xi_cut, p16_xi, p84_xi, alpha=0.25, color=color)
    ax_xi.errorbar(
        rp_xi_cut, xi_vv_cut, yerr=(rp_xi_cut**2)*err_xi,
        fmt='o', capsize=3, color=color, label=r'$\xi_{vv}$ data'
    )
    ax_xi.axhline(0, color='k', ls=':')
    ax_xi.set_xscale('log')
    ax_xi.set_ylabel(r'$\xi_{vv}(r)\times r^2$')
    ax_xi.legend(frameon=False)

    # xi_vv residuals as inset
    ax_xi_res = ax_xi.inset_axes([0.0, -0.55, 1.0, 0.4])
    residual_xi = 100.0 * (xi_vv_cut - xi_vv_bestfit) / xi_vv_cut
    frac_err_xi = 100.0 * err_xi / np.abs(xi_vv_cut)
    ax_xi_res.errorbar(
        rp_xi_cut, residual_xi, yerr=frac_err_xi,
        fmt='o', capsize=3, color=color
    )
    ax_xi_res.axhline(0, color='k', ls=':')
    ax_xi_res.set_xscale('log')
    ax_xi_res.set_xlabel(r'$r\,[\mathrm{Mpc}/h]$')
    ax_xi_res.set_ylabel(r'$\Delta/\mathrm{Data}\,[\%]$')
    ax_xi_res.set_ylim(-200, 200)

    plt.savefig('/disks/shear16/herle/code/plots/check_fits_env_dep_analytical_20_30.png',
                bbox_inches='tight')
    plt.close()

    # -------------------------------------------------
    # Print parameter summaries
    # -------------------------------------------------
    for j, name in enumerate(names):
        d = getdist.get1DDensity(name)
        xyind = np.where(d.P == np.amax(d.P))
        max_density = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)
        lower_error = errors_density[0]
        upper_error = errors_density[1]
        print(f"{name}_map", max_density)
        print(f"{name}_err_plus",  upper_error - max_density)
        print(f"{name}_err_minus", max_density - lower_error)

    # -------------------------------------------------
    # Triangle plot
    # -------------------------------------------------
    g = plots.get_subplot_plotter()
    g.settings.fig_width_inch  = 12
    g.settings.legend_fontsize = 25
    g.settings.fontsize        = 25
    g.settings.axes_labelsize  = 30
    g.settings.axes_fontsize   = 25

    g.triangle_plot(
        [getdist],
        ['A1_v', 'A2_v', 'b_v'],
        filled=[True],
        marker_args={'lw': 1.5},
        contour_colors=['k'],
        legend_kwargs={'frameon': False, 'ncol': 2}
    )

    plt.savefig('/disks/shear16/herle/code/plots/check_contours_env_dep_analytical_20_30.png')
    plt.close()