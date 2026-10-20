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
parser.add_argument('-box_size', type=float, default=1000)
parser.add_argument('-n_cores',  type=int,   default=192)
args = parser.parse_args()

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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
root_dir   = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

rootpath = Path('/disks/shear16/herle/code/GNAT')
if str(rootpath) not in os.sys.path:
    os.sys.path.append(str(rootpath))

from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
import pandas as pd

# ============================================================
# HANKEL TRANSFORM
# ============================================================

def j0(x):
    out = np.ones_like(x)
    mask = x != 0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def hankel_transform(r, delta_r, k_vals):
    dr = np.gradient(r)
    kr = np.outer(k_vals, r)
    kernel = j0(kr)
    integrand = kernel * (r**2 * dr * delta_r)
    return 4*np.pi * integrand.sum(axis=1)

# ============================================================
# REBINNING
# ============================================================

def rebin_measurement(rp, wvp, cov, edges):
    n_out = len(edges) - 1
    A = np.zeros((n_out, len(rp)))
    for j in range(n_out):
        mask = (rp >= edges[j]) & (rp < edges[j+1])
        if mask.sum() > 0:
            A[j, mask] = 1.0 / mask.sum()
    rp_rebinned  = A @ rp
    wvp_rebinned = A @ wvp
    cov_rebinned = A @ cov @ A.T
    nonempty = np.array([
        ((rp >= edges[j]) & (rp < edges[j+1])).any()
        for j in range(n_out)
    ])
    return (
        rp_rebinned[nonempty],
        wvp_rebinned[nonempty],
        cov_rebinned[np.ix_(nonempty, nonempty)]
    )

# ============================================================
# COSMOLOGY
# ============================================================

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486,
    Omega_b=0.0486,
    h=0.681,
    sigma8=0.807,
    n_s=0.967
)

redshift_list = np.array([0])
kmin, kmax    = 1e-5, 1e2

rp_array = np.logspace(-1, np.log10(900), 200)
r_array  = rp_array
pi_max   = 100.0

k_array_single = np.geomspace(kmin, kmax, 1000)
k_array        = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
pk_input       = np.zeros((len(k_array_single), 1, len(redshift_list)))

powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl, k_input=k_array_single,
        redshift=z, pk_type='nonlinear_matter'
    )
    pk_input[:, 0, idx_z] = -np.pi**2 * pkmm_ccl

predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl, redshift_list=redshift_list, logger=logger,
)
predictor.initialise_power_spectra(
    k_input=k_array, pk_input=pk_input, probetype_order=['g+'],
)
predictor.calculate_estimators(
    rp_support=rp_array, beta_shape=0, beta_density=0, pi_max=pi_max,
)
w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
    rp_support=rp_array, r_support=r_array,
)
w_2v = w_arr[:, 0, 0]

# ============================================================
# BASIC SETTINGS
# ============================================================

h        = 0.681
modes    = ['FIDUCIAL']
k_native = np.logspace(-5, 2, 1000001)   # h/Mpc

C1rhocrit  = 0.0134
z_i        = 0.0
scale_factor = 1.0 / (1.0 + z_i)
scaling = (
    C1rhocrit * cosmo_ccl['Omega_m']
    / (np.pi**2)
    / ccl.growth_factor(cosmo_ccl, scale_factor)
)

rmax_trunc = 400.0   # Mpc/h truncation for xi_vm Hankel transform

# coarser rp grid for the wvp fit
fit_rp_edges = np.logspace(np.log10(6.0), np.log10(500.0), 12)

# ============================================================
# VOID RADIUS BINS
# ============================================================

# void_bins = [(0, 10), (10, 20), (20, 30)]
# Rv_median = {'FIDUCIAL': [8.46, 14.72, 23.17]}
# bin_labels = ['0_10', '10_20', '20_30']
# delta_v_bin_labels = ['bin0', 'bin1', 'bin2']

void_bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
Rv_median = {
    'FIDUCIAL':    [8.23, 12.49, 17.17, 21.94, 26.76, 31.92]
}

bin_labels = ['0_10', '10_15', '15_20', '20_25', '25_30', '30_50']
delta_v_bin_labels = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5']


# ============================================================
# PRECOMPUTE xi_mm ONCE
# ============================================================

z_target = 0.0
a_target = 1.0 / (1.0 + z_target)
k_hmpc   = np.geomspace(1e-5, 1e3, 8192)

pk_ccl_mm = ccl.nonlin_matter_power(cosmo_ccl, k_hmpc * h, a_target)
pk_hmpc   = pk_ccl_mm * h**3

r_mm, xi_mm = P2xi(k_hmpc, lowring=True)(pk_hmpc, extrap=True)

keep   = (r_mm >= 0.1) & (r_mm <= 300.0)
r_mm   = r_mm[keep]
xi_mm  = xi_mm[keep]

xi_mm_interp = interp1d(
    r_mm, xi_mm, kind='cubic', bounds_error=False, fill_value=0.0
)

# Right after building xi_mm_interp, add:
print(f"\n--- xi_mm diagnostics ---")
print(f"r_mm range: {r_mm.min():.2f} to {r_mm.max():.2f} Mpc/h")
print(f"xi_mm at r=50:  {xi_mm_interp(50.0):.6e}")
print(f"xi_mm at r=100: {xi_mm_interp(100.0):.6e}")
print(f"xi_mm at r=200: {xi_mm_interp(200.0):.6e}")
print(f"xi_mm min/max: {xi_mm.min():.6e} / {xi_mm.max():.6e}")

# ============================================================
# GETDIST / HELPER FUNCTIONS
# ============================================================

from getdist import plots, MCSamples

def compute_chi2(logl, samples, corr):
    logL_max     = np.max(logl)
    chi2_eff     = -2 * logL_max
    Ndof         = len(corr) - samples.shape[1]
    chi2_red     = chi2_eff / Ndof
    return chi2_eff, chi2_red

def read_nautilus_samples(npz_filename):
    data    = np.load(npz_filename)
    return data['samples'], data['weights'], data['logl']

names  = ['A1_v', 'A2_v', 'b_v']
labels = ['A1_v', 'A2_v', 'b_v']

# ============================================================
# MAIN LOOP
# ============================================================

for i in range(len(modes)):

    print()
    print("-----------------------------")
    print(modes[i])
    print("-----------------------------")

    for bin_idx, (void_bin, bin_label, delta_bin_label) in enumerate(
        zip(void_bins, bin_labels, delta_v_bin_labels)
    ):
        Rv = Rv_median[modes[i]][bin_idx]

        print(f"\nProcessing {modes[i]} Rv bin {void_bin[0]}-{void_bin[1]} Mpc/h")
        print(f"Median Rv = {Rv:.2f} Mpc/h")

        lower_rp_VI  = 6.0
        upper_rp_VI  = 2.5 * Rv
        lower_rp_VII = 4.0 * Rv
        upper_rp_VII = 500.0

        print(f"VI  fitting range = {lower_rp_VI:.2f} - {upper_rp_VI:.2f} Mpc/h")
        print(f"VII fitting range = {lower_rp_VII:.2f} - {upper_rp_VII:.2f} Mpc/h")

        # ====================================================
        # 1-VOID TERM: xi_vm -> Hankel -> delta_v
        # ====================================================

        run      = 'L2800N5040'
        redshift = '0.0'

        df_corr = pd.read_hdf(
            f'/disks/shear16/herle/correlations1/'
            f'correlations_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical_morerp_fullt.h5',
            key='a'
        )

        r_xi  = df_corr['xi_vm_r'].values
        xi_vm_for_hankel = df_corr['xi_vm'].values

        # interpolate onto fine r grid
        r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 1000)
        xi_fine = interp1d(r_xi, xi_vm_for_hankel, kind='cubic')(r_fine)

        # truncate
        mask_r   = r_fine <= rmax_trunc
        r_trunc  = r_fine[mask_r]
        xi_trunc = xi_fine[mask_r]

        # Hankel transform
        delta_v  = hankel_transform(r_trunc, xi_trunc, k_native) / h**3

        # interpolate onto k_array_single
        interp_to_target = interp1d(
            np.log10(k_native), delta_v, kind="linear",
            bounds_error=False, fill_value=0.0
        )
        delta_v_on_target = interp_to_target(np.log10(k_array_single))

        # smooth on uniform log-k grid
        logk         = np.log10(k_array_single)
        logk_uniform = np.linspace(logk.min(), logk.max(), 20000)

        delta_uniform = interp1d(logk, delta_v_on_target, kind="linear")(logk_uniform)

        window_length = 1501
        polyorder     = 3
        delta_uniform_smooth = savgol_filter(delta_uniform, window_length, polyorder)

        delta_v_smoothed = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")(logk)

        # feed into GNAT
        pk_input_1v = np.zeros((len(k_array_single), 1, len(redshift_list)))
        for idx_z, z in enumerate(redshift_list):
            pk_input_1v[:, 0, idx_z] = -np.pi**2 * delta_v_smoothed

        predictor.initialise_power_spectra(
            k_input=k_array, pk_input=pk_input_1v, probetype_order=['g+'],
        )
        predictor.calculate_estimators(
            rp_support=rp_array, beta_shape=0, beta_density=0, pi_max=pi_max,
        )
        w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
            rp_support=rp_array, r_support=r_array,
        )
        w_1v = w_arr[:, 0, 0]

        # ====================================================
        # MODEL TEMPLATES
        # ====================================================

        wVII = w_2v.copy() * scaling
        wVI  = w_1v.copy() * scaling

        model_rp      = rp_array
        model_mask_VI  = (model_rp >= lower_rp_VI)  & (model_rp <= upper_rp_VI)
        model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

        # ====================================================
        # LOAD wvp MEASUREMENT + BUILD COVARIANCE + REBIN
        # ====================================================

        wvp_JK = np.load(
            f'/disks/shear16/herle/correlations1/'
            f'wvp_xip_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical_morerp_fullt.npy'
        )

        rp_meas  = df_corr['wvp_rp'].values
        wvp_meas = df_corr['wvp_xip'].values

        mask_total = (
            ((rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)) |
            ((rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII))
        )

        rp_cut     = rp_meas[mask_total]
        corr       = wvp_meas[mask_total]
        wvp_JK_cut = wvp_JK[:, mask_total]

        NPatches = wvp_JK_cut.shape[0]
        print(f'NPatches: {NPatches}')

        diff_wvp    = wvp_JK_cut - wvp_JK_cut.mean(axis=0)
        cov_wvp_arr = ((NPatches - 1) / NPatches) * (diff_wvp.T @ diff_wvp)

        # rebin wvp onto coarser rp grid
        rp_cut, corr, cov_wvp_arr = rebin_measurement(
            rp_cut, corr, cov_wvp_arr, fit_rp_edges
        )

        # masks on rebinned rp
        mask_data_VI  = (rp_cut >= lower_rp_VI)  & (rp_cut <= upper_rp_VI)
        mask_data_VII = (rp_cut >= lower_rp_VII) & (rp_cut <= upper_rp_VII)

        # ====================================================
        # LOAD xi_vm MEASUREMENT + BUILD COVARIANCE
        # ====================================================

        xi_vm_JK = np.load(
            f'/disks/shear16/herle/correlations1/'
            f'xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical_morerp_fullt.npy'
        )

        rp_xi_vm    = df_corr['xi_vm_r'].values
        xi_vm_meas  = df_corr['xi_vm'].values

        xi_rmin = max(3.0 * Rv, 150.0)   # well beyond compensation wall
        xi_rmax = 300.0
        mask_xi = (rp_xi_vm >= xi_rmin) & (rp_xi_vm <= xi_rmax)

        rp_xi_cut    = rp_xi_vm[mask_xi]
        xi_vm_cut    = xi_vm_meas[mask_xi]
        xi_vm_JK_cut = xi_vm_JK[:, mask_xi]

        print(f"\n--- xi_vm data diagnostics ---")
        print(f"rp_xi_cut range: {rp_xi_cut.min():.2f} to {rp_xi_cut.max():.2f} Mpc/h")
        print(f"xi_vm_cut: {xi_vm_cut}")
        print(f"xi_vm_cut min/max: {xi_vm_cut.min():.6e} / {xi_vm_cut.max():.6e}")

        diff_xi    = xi_vm_JK_cut - xi_vm_JK_cut.mean(axis=0)
        cov_xi_arr = ((NPatches - 1) / NPatches) * (diff_xi.T @ diff_xi)

        err_xi = np.sqrt(np.diag(cov_xi_arr))

        # ====================================================
        # JOINT COVARIANCE
        # ====================================================

        # cross-covariance block between wvp and xi_vm
        # (wvp_JK_cut here is already rebinned — recompute
        #  from the rebinned rp_cut against the full JK array)
        # We need JK realisations on the rebinned wvp grid.
        # Rebin each JK realisation the same way.
        A_rebin = np.zeros((len(rp_cut), wvp_JK[:, mask_total].shape[1]))
        rp_fine = rp_meas[mask_total]
        for j in range(len(fit_rp_edges) - 1):
            m = (rp_fine >= fit_rp_edges[j]) & (rp_fine < fit_rp_edges[j+1])
            if m.sum() > 0:
                # find which output bin this maps to
                pass  # handled implicitly below

        # Cleaner: just rebin each JK realisation directly
        wvp_JK_rebinned = np.array([
            rebin_measurement(
                rp_meas[mask_total],
                wvp_JK[p, mask_total],
                np.eye(mask_total.sum()),   # dummy cov — we only need rp/wvp
                fit_rp_edges
            )[1]
            for p in range(NPatches)
        ])

        diff_wvp_rb = wvp_JK_rebinned - wvp_JK_rebinned.mean(axis=0)
        cross_block = ((NPatches - 1) / NPatches) * (diff_wvp_rb.T @ diff_xi)

        n_wvp = len(rp_cut)
        n_xi  = len(rp_xi_cut)

        cov_joint = np.zeros((n_wvp + n_xi, n_wvp + n_xi))
        cov_joint[:n_wvp, :n_wvp] = cov_wvp_arr
        cov_joint[n_wvp:, n_wvp:] = cov_xi_arr
        cov_joint[:n_wvp, n_wvp:] = cross_block
        cov_joint[n_wvp:, :n_wvp] = cross_block.T

        inv_cov    = np.linalg.pinv(cov_joint)
        corr_joint = np.concatenate([corr, xi_vm_cut])

        err_cut = np.sqrt(np.diag(cov_wvp_arr))

        # ====================================================
        # MODEL FUNCTIONS
        # ====================================================

        def model_wVI(p):
            A1_v = p[0]
            wVI_model = wVI[model_mask_VI].copy() * A1_v
            return model_rp[model_mask_VI], wVI_model

        def model_wVII(p):
            A2_v = p[1]
            b_v  = p[2]
            wVII_model = wVII[model_mask_VII].copy() * A2_v * b_v
            return model_rp[model_mask_VII], wVII_model

        def model_xi_vm(p):
            b_v = p[2]
            return rp_xi_cut, b_v * xi_mm_interp(rp_xi_cut)



        # ====================================================
        # JOINT LOG-LIKELIHOOD
        # ====================================================

        def log_likelihood(p):
            rp_VI,  wVI_model  = model_wVI(p)
            rp_VII, wVII_model = model_wVII(p)

            rp_total   = np.concatenate([rp_VI, rp_VII])
            wvp_total  = np.concatenate([wVI_model, wVII_model])

            wvp_interp = np.interp(rp_cut, rp_total, wvp_total)

            _, xi_model = model_xi_vm(p)

            model_joint = np.concatenate([wvp_interp, xi_model])

            delta = corr_joint - model_joint
            chisq = delta.dot(inv_cov.dot(delta))
            ll    = -0.5 * chisq

            if np.isnan(ll):
                return -np.inf, chisq
            return ll, chisq

        # ====================================================
        # PRIORS
        # ====================================================

        prior_obj   = Prior()
        param_names = ['A1_v', 'A2_v', 'b_v']

        prior_obj.add_parameter('A1_v', dist=(-10.0, 10.0))
        prior_obj.add_parameter('A2_v', dist=(-50.0, 50.0))
        prior_obj.add_parameter('b_v',  dist=(-10.0,  10.0))

        def loglikelihood_fn(p_dict):
            p = np.array([p_dict[key] for key in param_names])
            logl, _ = log_likelihood(p)
            return logl

        box_size = args.box_size
        n_cores  = args.n_cores

        # ====================================================
        # NAUTILUS SAMPLER
        # ====================================================

        def run_sampler():
            filename = (
                f'/disks/shear16/herle/models/models/'
                f'nautilus_chain_revolver_{modes[i]}_voids_{box_size}'
                f'_Rv{bin_label}Mpc_env_dep_analytical_xivmhankel_morerp_fullt_wl1501.h5'
            )
            sampler = Sampler(
                prior_obj, loglikelihood_fn,
                n_live=5000,
                pool=n_cores if n_cores > 1 else None,
                filepath=filename,
                resume=False,
                n_networks=32,
            )
            sampler.run(verbose=True, discard_exploration=True)
            points, log_w, log_l = sampler.posterior()
            weights = np.exp(log_w)
            output_file = (
                f'/disks/shear16/herle/models/models/'
                f'nautilus_chain_revolver_{modes[i]}_voids_{box_size}'
                f'_Rv{bin_label}Mpc_env_dep_analytical_xivmhankel_morerp_fullt_wl1501.npz'
            )
            np.savez(output_file,
                     samples=points, logl=log_l,
                     weights=weights, logz=sampler.log_z)
            print(f"Saved results to {output_file}")

        if __name__ == '__main__':
            run_sampler()

        # ====================================================
        # LOAD POSTERIOR
        # ====================================================

        filename = (
            f'/disks/shear16/herle/models/models/'
            f'nautilus_chain_revolver_{modes[i]}_voids_{box_size}'
            f'_Rv{bin_label}Mpc_env_dep_analytical_xivmhankel_morerp_fullt_wl1501.npz'
        )
        samples, weights, logl = read_nautilus_samples(filename)

        getdist = MCSamples(
            samples=samples, weights=weights,
            names=names, labels=labels, ignore_rows=0
        )

        chi2_eff, chi2_red = compute_chi2(logl, samples, corr_joint)
        print(f"[{modes[i]} | {bin_label}] Reduced Chi2: {chi2_red:.3f}")

        map_idx  = np.argmax(logl)
        A1_best  = samples[map_idx, 0]
        A2_best  = samples[map_idx, 1]
        bv_best  = samples[map_idx, 2]

        print(f"Joint MAP: A1_v={A1_best:.4f}  A2_v={A2_best:.4f}  b_v={bv_best:.4f}")

        # marginalised summary
        for j, name in enumerate(names):
            d = getdist.get1DDensity(name)
            xyind = np.where(d.P == np.amax(d.P))
            mode_val = d.x[xyind[0]][0]
            lims = d.getLimits(0.68)
            print(f"{name}: mode={mode_val:.4f}  "
                  f"+{lims[1]-mode_val:.4f} / -{mode_val-lims[0]:.4f}  "
                  f"MAP={samples[map_idx, j]:.4f}")

        # ====================================================
        # BEST-FIT MODEL VECTORS
        # ====================================================

        wVI_bestfit  = wVI  * A1_best
        wVII_bestfit = wVII * A2_best * bv_best

        weights_norm    = weights / np.sum(weights)
        inds            = np.random.choice(len(samples), size=500, p=weights_norm)
        posterior_draws = samples[inds]

        A1_draws = posterior_draws[:, 0]
        A2_draws = posterior_draws[:, 1]
        bv_draws = posterior_draws[:, 2]

        wVI_ens  = np.array([wVI  * A1 for A1 in A1_draws])
        wVII_ens = np.array([wVII * A2 * bv for A2, bv in zip(A2_draws, bv_draws)])

        p16_VI,  p50_VI,  p84_VI  = np.percentile(wVI_ens,  [16, 50, 84], axis=0)
        p16_VII, p50_VII, p84_VII = np.percentile(wVII_ens, [16, 50, 84], axis=0)

        _, xi_vm_bestfit = model_xi_vm([A1_best, A2_best, bv_best])
        xi_vm_ens = np.array([
            model_xi_vm([A1, A2, bv])[1]
            for A1, A2, bv in zip(A1_draws, A2_draws, bv_draws)
        ])
        p16_xi, p50_xi, p84_xi = np.percentile(xi_vm_ens, [16, 50, 84], axis=0)

        print(f"\n--- xi_vm model diagnostics ---")
        print(f"b_v MAP value: {bv_best:.6f}")
        print(f"xi_mm at rp_xi_cut: {xi_mm_interp(rp_xi_cut)}")
        print(f"xi_vm_bestfit: {xi_vm_bestfit}")
        print(f"xi_vm_bestfit min/max: {xi_vm_bestfit.min():.6e} / {xi_vm_bestfit.max():.6e}")
        print(f"ratio xi_vm_cut / xi_mm: {xi_vm_cut / xi_mm_interp(rp_xi_cut)}")

        # ====================================================
        # RESIDUALS
        # ====================================================

        model_at_data = np.full_like(rp_cut, np.nan)
        model_at_data[mask_data_VI] = np.interp(
            rp_cut[mask_data_VI], model_rp[model_mask_VI], wVI_bestfit[model_mask_VI]
        )
        model_at_data[mask_data_VII] = np.interp(
            rp_cut[mask_data_VII], model_rp[model_mask_VII], wVII_bestfit[model_mask_VII]
        )

        residual  = 100.0 * (corr - model_at_data) / corr
        frac_err  = 100.0 * err_cut / np.abs(corr)

        residual_xi  = 100.0 * (xi_vm_cut - xi_vm_bestfit) / xi_vm_cut
        frac_err_xi  = 100.0 * err_xi / np.abs(xi_vm_cut)

        color = 'k'
        label = f'{modes[i]} $R_v={void_bin[0]}$--${void_bin[1]}$'

        # ====================================================
        # FIGURE
        # ====================================================

        fig, (ax_wvp, ax_wvp_res, ax_xi) = plt.subplots(
            3, 1, figsize=(10, 14), sharex=False,
            gridspec_kw={"height_ratios": [3, 1, 2], "hspace": 0.35}
        )

        # wvp main
        ax_wvp.plot(model_rp[model_mask_VII],
                    model_rp[model_mask_VII] * wVII_bestfit[model_mask_VII],
                    color=color, label='2-void')
        ax_wvp.plot(model_rp[model_mask_VI],
                    model_rp[model_mask_VI] * wVI_bestfit[model_mask_VI],
                    color=color, ls='--', label='1-void')
        ax_wvp.errorbar(rp_cut, rp_cut * corr, yerr=rp_cut * err_cut,
                        fmt='o', capsize=3, color=color, label=label)
        ax_wvp.fill_between(model_rp[model_mask_VI],
                            model_rp[model_mask_VI] * p16_VI[model_mask_VI],
                            model_rp[model_mask_VI] * p84_VI[model_mask_VI],
                            alpha=0.25, color=color)
        ax_wvp.fill_between(model_rp[model_mask_VII],
                            model_rp[model_mask_VII] * p16_VII[model_mask_VII],
                            model_rp[model_mask_VII] * p84_VII[model_mask_VII],
                            alpha=0.25, color=color)
        ax_wvp.axvspan(lower_rp_VI,  upper_rp_VI,  color='grey', alpha=0.08, zorder=0)
        ax_wvp.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)
        ax_wvp.axhline(0, color='k', ls=':')
        ax_wvp.set_xscale('log')
        ax_wvp.set_xlim(1, 600)
        ax_wvp.set_ylabel(r'$r_p w_{\mathrm{V}+}\,[\mathrm{Mpc}^2/h^2]$')
        ax_wvp.set_title(
            f'{modes[i]}  |  $R_v={void_bin[0]}$–${void_bin[1]}$ Mpc/$h$  '
            f'|  $\\langle R_v\\rangle={Rv:.2f}$ Mpc/$h$  '
            f'|  $\\chi^2_{{\\rm red}}={chi2_red:.2f}$',
            fontsize=11
        )
        ax_wvp.legend(frameon=False)

        # wvp residuals
        ax_wvp_res.errorbar(rp_cut, residual, yerr=frac_err,
                            fmt='o', capsize=3, color=color)
        ax_wvp_res.axvspan(lower_rp_VI,  upper_rp_VI,  color='grey', alpha=0.08, zorder=0)
        ax_wvp_res.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)
        ax_wvp_res.axhline(0, color='k', ls=':')
        ax_wvp_res.set_xscale('log')
        ax_wvp_res.set_xlim(1, 600)
        ax_wvp_res.set_ylim(-200, 200)
        ax_wvp_res.set_xlabel(r'$r_p\,[\mathrm{Mpc}/h]$')
        ax_wvp_res.set_ylabel(r'$\Delta/\mathrm{Data}\,[\%]$')

        # xi_vm panel
        ax_xi.plot(rp_xi_cut, (rp_xi_cut**2) * xi_vm_bestfit,
                   color=color, label=r'$b_v\,\xi_{mm}\times r^2$')
        ax_xi.fill_between(rp_xi_cut,
                           (rp_xi_cut**2) * p16_xi,
                           (rp_xi_cut**2) * p84_xi,
                           alpha=0.25, color=color)
        ax_xi.errorbar(rp_xi_cut, (rp_xi_cut**2) * xi_vm_cut,
                       yerr=(rp_xi_cut**2) * err_xi,
                       fmt='o', capsize=3, color=color, label=r'$\xi_{vm}$ data')
        ax_xi.axhline(0, color='k', ls=':')
        ax_xi.set_xscale('log')
        ax_xi.set_ylabel(r'$\xi_{vm}(r)\times r^2$')
        ax_xi.legend(frameon=False)

        # xi_vm residuals inset
        ax_xi_res = ax_xi.inset_axes([0.0, -0.55, 1.0, 0.4])
        ax_xi_res.errorbar(rp_xi_cut, residual_xi, yerr=frac_err_xi,
                           fmt='o', capsize=3, color=color)
        ax_xi_res.axhline(0, color='k', ls=':')
        ax_xi_res.set_xscale('log')
        ax_xi_res.set_xlabel(r'$r\,[\mathrm{Mpc}/h]$')
        ax_xi_res.set_ylabel(r'$\Delta/\mathrm{Data}\,[\%]$')
        ax_xi_res.set_ylim(-200, 200)

        plt.savefig(
            f'/disks/shear16/herle/code/plots/'
            f'check_fits_env_dep_analytical_{bin_label}_xivmhankel_morerp_fullt_wl1501.png',
            bbox_inches='tight'
        )
        plt.close()

        # triangle plot
        g = plots.get_subplot_plotter()
        g.settings.fig_width_inch  = 12
        g.settings.legend_fontsize = 25
        g.settings.fontsize        = 25
        g.settings.axes_labelsize  = 30
        g.settings.axes_fontsize   = 25
        g.triangle_plot(
            [getdist], ['A1_v', 'A2_v', 'b_v'],
            filled=[True], marker_args={'lw': 1.5},
            contour_colors=['k'],
            legend_kwargs={'frameon': False, 'ncol': 2}
        )
        plt.savefig(
            f'/disks/shear16/herle/code/plots/'
            f'check_contours_env_dep_analytical_{bin_label}_xivmhankel_morerp_fullt_wl1501.png',
            bbox_inches='tight'
        )
        plt.close()

        print(f"\nFinished {modes[i]} Rv bin {void_bin[0]}-{void_bin[1]}")
        print("--------------------------------------------------")