import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt
import glob
import pandas as pd
from nautilus import Sampler, Prior
from multiprocessing import Pool

import argparse

import os
os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-box_size', type=float, default=1000, help='Size of the box.')
parser.add_argument('-n_cores', type=int, default=128, help='Number of CPU cores to use for multiprocessing.')
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
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

rootpath = Path('/disks/shear16/herle/code/GNAT')
if str(rootpath) not in os.sys.path:
    os.sys.path.append(str(rootpath))

from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum

import pandas as pd

# =====================================================
# Hankel transform
# =====================================================
def j0(x):
    out = np.ones_like(x)
    mask = x != 0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def hankel_transform(r, delta_r, k_vals):
    dr = np.gradient(r)
    kr = np.outer(k_vals, r)      # shape (Nk, Nr)
    kernel = j0(kr)
    integrand = kernel * (r**2 * dr * delta_r)
    return 4*np.pi * integrand.sum(axis=1)

# =====================================================

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486,
    Omega_b=0.0486,
    h=0.681,
    sigma8=0.807,
    n_s=0.967
)

redshift_list = np.array([0])

kmin, kmax = 1e-5, 1e2

rp_values2v = np.logspace(-1, np.log10(900), 200)

rp_array = rp_values2v
r_array  = rp_array

pi_max = 100.0

k_array_single = np.geomspace(kmin, kmax, 1000)
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

names  = ['A1_v', 'b_v']
labels = ['A1_v', 'b_v']

void_bins = [(0, 10), (10, 20), (20, 30)]
Rv_median = {
    'FIDUCIAL': [8.46, 14.72, 23.17],
}

bin_labels       = ['0_10', '10_20', '20_30']
delta_v_bin_labels = ['bin0', 'bin1', 'bin2']

rmax_trunc = 500.0   # Mpc/h — match shell profile truncation

chi2_results = {mode: {'bin_label': [], 'chi2_red': []} for mode in modes}

# Define coarser bins for the fit (adjust edges to taste)
fit_rp_edges = np.logspace(np.log10(6.0), np.log10(500.0), 12)  # ~11 bins

def rebin_measurement(rp, wvp, cov, edges):
    """
    Average wvp into coarser bins defined by edges.
    Propagates the full covariance correctly.
    """
    n_out = len(edges) - 1
    wvp_rebinned  = np.zeros(n_out)
    rp_rebinned   = np.zeros(n_out)
    cov_rebinned  = np.zeros((n_out, n_out))

    # Build the averaging matrix A: shape (n_out, n_in)
    A = np.zeros((n_out, len(rp)))
    for j in range(n_out):
        mask = (rp >= edges[j]) & (rp < edges[j+1])
        if mask.sum() > 0:
            A[j, mask] = 1.0 / mask.sum()

    wvp_rebinned = A @ wvp
    rp_rebinned  = A @ rp
    cov_rebinned = A @ cov @ A.T

    # drop empty bins
    nonempty = np.array([
        (rp >= edges[j]).any() and (rp < edges[j+1]).any()
        for j in range(n_out)
    ])
    return rp_rebinned[nonempty], wvp_rebinned[nonempty], cov_rebinned[np.ix_(nonempty, nonempty)]


for i in range(len(modes)):

    print("-----------------------------")
    print(modes[i])
    print("-----------------------------")

    for bin_idx, (void_bin, bin_label, delta_bin_label) in enumerate(
        zip(void_bins, bin_labels, delta_v_bin_labels)
    ):
        Rv = Rv_median[modes[i]][bin_idx]

        print(f"  -- Void bin: {void_bin}, Rv={Rv}, bin_label={bin_label}")

        lower_rp_VI  = 6.0
        upper_rp_VI  = 2.5 * Rv

        lower_rp_VII = 4.0 * Rv
        upper_rp_VII = 500.0

        # --------------------------------------------------
        # Load xi_vm from measurement and Hankel transform it
        # --------------------------------------------------
        run      = 'L1000N1800'
        redshift = '0.0'

        df_corr = pd.read_hdf(
            f'/disks/shear16/herle/correlations1/'
            f'correlations_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical_morerp.h5',
            key='a'
        )

        r_xi  = df_corr['xi_vm_r'].values    # Mpc/h
        xi_vm = df_corr['xi_vm'].values       # dimensionless

        # Interpolate xi_vm onto a fine r-grid before transforming
        # (xi_vm only has ~25 points; need density for accurate Hankel)
        r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 1000)
        xi_fine = interp1d(r_xi, xi_vm, kind='cubic')(r_fine)

        # Truncate at rmax
        mask_r   = r_fine <= rmax_trunc
        r_trunc  = r_fine[mask_r]
        xi_trunc = xi_fine[mask_r]

        # Hankel transform — identical unit convention to shell profile code:
        # r in Mpc/h, k_native in h/Mpc -> amplitude in (Mpc/h)^3, then /h^3 -> Mpc^3
        delta_v = hankel_transform(r_trunc, xi_trunc, k_native)
        delta_v /= h**3

        # Interpolate onto k_array_single
        logk_void   = np.log10(k_native)
        logk_target = np.log10(k_array_single)

        interp_to_target = interp1d(
            logk_void, delta_v, kind="linear",
            bounds_error=False, fill_value=0.0
        )
        delta_v_on_target = interp_to_target(logk_target)

        # Smooth on k_array_single's log grid
        logk = logk_target

        N_log        = 20000
        logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

        interp_func    = interp1d(logk, delta_v_on_target, kind="linear")
        delta_uniform  = interp_func(logk_uniform)

        window_length = 3001
        polyorder     = 3
        if window_length % 2 == 0:
            window_length += 1

        delta_uniform_smooth = savgol_filter(delta_uniform, window_length, polyorder)

        interp_back      = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")
        delta_v_smoothed = interp_back(logk)

        # Pack into pk_input
        pk_input_1v = np.zeros((len(k_array_single), 1, len(redshift_list)))
        for idx_z, z in enumerate(redshift_list):
            pk_input_1v[:, 0, idx_z] = -np.pi**2 * delta_v_smoothed

        # --------------------------------------------------
        # Everything below is unchanged from original
        # --------------------------------------------------

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

        wVII = w_2v.copy()
        wVII *= scaling

        wVI = w_1v.copy()
        wVI *= scaling

        model_rp = rp_array

        model_mask_VI  = (model_rp >= lower_rp_VI)  & (model_rp <= upper_rp_VI)
        model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

        # -------------------------------------------------
        # load measurement + JK realizations
        # -------------------------------------------------
        meas_file = (
            f'/disks/shear16/herle/correlations1/'
            f'correlations_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical_morerp.h5'
        )

        wvp_JK_measured = np.load(
            f'/disks/shear16/herle/correlations1/'
            f'wvp_xip_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical_morerp.npy'
        )

        # reuse df_corr since it is the same file
        rp_meas  = df_corr['wvp_rp'].values
        wvp_meas = df_corr['wvp_xip'].values

        mask_VI  = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
        mask_VII = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)

        mask_total = mask_VI | mask_VII

        rp_cut = rp_meas[mask_total]
        corr   = wvp_meas[mask_total]

        wvp_JK_cut = wvp_JK_measured[:, mask_total]

        wvp_mean = np.mean(wvp_JK_cut, axis=0)

        wvp_diff = pd.DataFrame(
            wvp_JK_cut - wvp_mean,
            columns=rp_cut
        )

        NPatches = wvp_JK_cut.shape[0]
        print('Npatches: ', NPatches)

        cov_mat = ((NPatches - 1) / NPatches) * np.sum(
            np.einsum('ij,ik->ijk', wvp_diff, wvp_diff),
            axis=0
        )

        cov_mat = pd.DataFrame(
            data=cov_mat,
            columns=rp_cut
        )
        cov_mat.set_index(rp_cut, inplace=True)

        # fig, ax = plt.subplots()
        # im = ax.imshow(np.log10(np.abs(cov_mat)), origin='lower')
        # fig.colorbar(im, ax=ax)
        # plt.tight_layout()
        # plt.savefig(
        #     f'/disks/shear16/herle/code/plots/'
        #     f'cov_void_corr_{modes[i]}_{bin_label}_particles_env_dep_analytical_xivmhankel.png'
        # )
        # plt.close()

        cov_arr = cov_mat.values

        # rebin for the fit
        rp_cut, corr, cov_arr = rebin_measurement(rp_cut, corr, cov_arr, fit_rp_edges)
        inv_cov = np.linalg.pinv(cov_arr)

        # recompute VI/VII masks on the rebinned rp_cut
        mask_data_VI  = (rp_cut >= lower_rp_VI)  & (rp_cut <= upper_rp_VI)
        mask_data_VII = (rp_cut >= lower_rp_VII) & (rp_cut <= upper_rp_VII)

        def model_wVI(p):
            A1_v = p[0]
            wVI_model = wVI[model_mask_VI].copy()
            wVI_model *= A1_v
            return model_rp[model_mask_VI], wVI_model

        def model_wVII(p):
            A1_v = p[0]
            b_v  = p[1]
            wVII_model = wVII[model_mask_VII].copy()
            wVII_model *= A1_v * b_v
            return model_rp[model_mask_VII], wVII_model

        def log_likelihood(p):
            rp_model_VI,  wVI_model  = model_wVI(p)
            rp_model_VII, wVII_model = model_wVII(p)

            rp_model_total   = np.concatenate([rp_model_VI,  rp_model_VII])
            corr_model_total = np.concatenate([wVI_model, wVII_model])

            corr_model_interpol = np.interp(rp_cut, rp_model_total, corr_model_total)

            delta = corr - corr_model_interpol
            chisq = delta.dot(inv_cov.dot(delta.T))
            ll = -0.5 * chisq
            if np.isnan(ll):
                return -np.inf, chisq
            return ll, chisq

        prior_obj = Prior()

        param_names = ['A1_v', 'b_v']
        prior_obj.add_parameter('A1_v', dist=(-10.0, 10.0))
        prior_obj.add_parameter('b_v',  dist=(-10.0, 10.0))

        def loglikelihood_fn(p_dict):
            p = np.array([p_dict[key] for key in param_names])
            logl, _ = log_likelihood(p)
            return logl

        box_size = args.box_size
        n_cores  = args.n_cores

        def run():
            n_live = 5000
            filename = (
                f'/disks/shear16/herle/models/models/'
                f'nautilus_chain_revolver_{modes[i]}_voids_{box_size}'
                f'_Rv{bin_label}Mpc_seperate_env_dep_analytical_xivmhankel_morerp_rebinned.h5'
            )
            sampler = Sampler(
                prior_obj,
                loglikelihood_fn,
                n_live=n_live,
                pool=n_cores if n_cores > 1 else None,
                filepath=filename,
                resume=False,
                n_networks=32,
            )
            sampler.run(verbose=True, discard_exploration=True)
            points, log_w, log_l = sampler.posterior()
            weights = np.exp(log_w)
            logz    = sampler.log_z
            output_file = (
                f'/disks/shear16/herle/models/models/'
                f'nautilus_chain_revolver_{modes[i]}_voids_{box_size}'
                f'_Rv{bin_label}Mpc_seperate_env_dep_analytical_xivmhankel_morerp_rebinned.npz'
            )
            np.savez(output_file,
                     samples=points,
                     logl=log_l,
                     weights=weights,
                     logz=logz)
            print(f"Saved results to {output_file}")

        if __name__ == '__main__':
            run()

        filename = (
            f'/disks/shear16/herle/models/models/'
            f'nautilus_chain_revolver_{modes[i]}_voids_{box_size}'
            f'_Rv{bin_label}Mpc_seperate_env_dep_analytical_xivmhankel_morerp_rebinned.npz'
        )
        samples, weights, logl = read_nautilus_samples(filename)
        getdist = MCSamples(samples=samples,
                            weights=weights,
                            names=names,
                            labels=labels,
                            ignore_rows=0)

        chi2_eff, chi2_red = compute_chi2(logl, samples, corr)
        print(f"[{modes[i]}] Reduced Chi2: ", chi2_red)

        chi2_results[modes[i]]['bin_label'].append(bin_label)
        chi2_results[modes[i]]['chi2_red'].append(chi2_red)

        d = getdist.get1DDensity('A1_v')
        xyind = np.where(d.P == np.amax(d.P))
        A1v = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)
        A1v_error = (errors_density[1] - errors_density[0]) / 2.0

        d = getdist.get1DDensity('b_v')
        xyind = np.where(d.P == np.amax(d.P))
        bv = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)
        bv_error = (errors_density[1] - errors_density[0]) / 2.0

        map_idx  = np.argmax(logl)
        A1_best  = samples[map_idx, 0]
        bv_best  = samples[map_idx, 1]

        wVI_bestfit  = wVI  * A1_best
        wVII_bestfit = wVII * A1_best * bv_best

        Nsamp        = 500
        weights_norm = weights / np.sum(weights)
        inds         = np.random.choice(len(samples), size=Nsamp, p=weights_norm)
        posterior_draws = samples[inds]

        A1_draws = posterior_draws[:, 0]
        bv_draws = posterior_draws[:, 1]

        wVI_ens  = []
        wVII_ens = []

        for A1, bv_d in zip(A1_draws, bv_draws):
            wVI_ens.append(wVI * A1)
            wVII_ens.append(wVII * A1 * bv_d)

        wVI_ens  = np.array(wVI_ens)
        wVII_ens = np.array(wVII_ens)

        p16_VI,  p50_VI,  p84_VI  = np.percentile(wVI_ens,  [16, 50, 84], axis=0)
        p16_VII, p50_VII, p84_VII = np.percentile(wVII_ens, [16, 50, 84], axis=0)

        err_cut = np.sqrt(np.diag(cov_arr))

        # mask_data_VI  = mask_VI[mask_total]
        # mask_data_VII = mask_VII[mask_total]

        color = 'k'
        label = modes[i]

        fig, (ax_main, ax_res) = plt.subplots(
            2, 1,
            figsize=(14, 14),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
        )

        ax_main.plot(model_rp[model_mask_VII], model_rp[model_mask_VII] * wVII_bestfit[model_mask_VII], color=color)
        ax_main.plot(model_rp[model_mask_VI],  model_rp[model_mask_VI]  * wVI_bestfit[model_mask_VI],  color=color)

        ax_main.errorbar(rp_cut, rp_cut * corr, yerr=rp_cut * err_cut,
                         fmt='o', capsize=3, color=color, label=label)

        ax_main.fill_between(
            model_rp[model_mask_VI],
            model_rp[model_mask_VI] * p16_VI[model_mask_VI],
            model_rp[model_mask_VI] * p84_VI[model_mask_VI],
            alpha=0.25, color=color
        )
        ax_main.fill_between(
            model_rp[model_mask_VII],
            model_rp[model_mask_VII] * p16_VII[model_mask_VII],
            model_rp[model_mask_VII] * p84_VII[model_mask_VII],
            alpha=0.25, color=color
        )

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

        ax_res.errorbar(rp_cut, residual, yerr=frac_err,
                        fmt='o', capsize=3, color=color)

        ax_main.axvspan(lower_rp_VI,  upper_rp_VI,  color='grey', alpha=0.08, zorder=0)
        ax_main.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)
        ax_res.axvspan(lower_rp_VI,   upper_rp_VI,  color='grey', alpha=0.08, zorder=0)
        ax_res.axvspan(lower_rp_VII,  upper_rp_VII, color='grey', alpha=0.08, zorder=0)

        ax_main.axhline(0, color='k', ls=':')
        ax_main.set_xscale('log')
        ax_main.set_xlim(1, 600)
        ax_main.set_ylabel(r'$r_p w_{\mathrm{V}+}\,[\mathrm{Mpc}^2/h^2]$')
        ax_main.legend(frameon=False)
        ax_main.set_title(
            f'{modes[i]}  |  $R_v = {void_bin[0]}$–${void_bin[1]}$ Mpc/$h$  '
            f'|  $\\langle R_v \\rangle = {Rv:.2f}$ Mpc/$h$',
            fontsize=12
        )

        ax_res.axhline(0, color='k', ls=':')
        ax_res.text(
            0.02, 0.92,
            f'$\\chi^2_{{\\rm red}} = {chi2_red:.2f}$',
            transform=ax_res.transAxes,
            fontsize=12,
            va='top'
        )
        ax_res.set_xscale('log')
        ax_res.set_ylim(-200, 200)
        ax_res.set_xlabel(r'$r_p$ [Mpc/h]')
        ax_res.set_ylabel(r'$(\mathrm{Data}-\mathrm{Model})/\mathrm{Data}$ [%]')

        plt.tight_layout()
        plt.savefig(
            f'/disks/shear16/herle/code/plots/'
            f'check_fits_{modes[i]}_{bin_label}_seperate_env_dep_analytical_xivmhankel_morerp_rebinned.png'
        )

        map_idx = np.argmax(logl)

        print("=" * 50)

        for param_idx, name in enumerate(names):
            d = getdist.get1DDensity(name)
            xyind = np.where(d.P == np.amax(d.P))
            marginal_mode = d.x[xyind[0]][0]
            errors_density = d.getLimits(0.68)
            lower_error = errors_density[0]
            upper_error = errors_density[1]
            joint_map_val = samples[map_idx, param_idx]
            print(f"\n{name}:")
            print(f"  Marginalised mode: {marginal_mode:.4f}  "
                  f"+{upper_error - marginal_mode:.4f} / -{marginal_mode - lower_error:.4f}")
            print(f"  Joint MAP value:   {joint_map_val:.4f}")

        g = plots.get_subplot_plotter()
        g.settings.fig_width_inch  = 12
        g.settings.legend_fontsize = 25
        g.settings.fontsize        = 25
        g.settings.axes_labelsize  = 30
        g.settings.axes_fontsize   = 25

        g.triangle_plot(
            [getdist],
            ['A1_v', 'b_v'],
            filled=[True],
            marker_args={'lw': 1.5},
            contour_colors=['k'],
            legend_kwargs={'frameon': False, 'ncol': 2}
        )

        plt.savefig(
            f'/disks/shear16/herle/code/plots/'
            f'check_contours_{modes[i]}_{bin_label}_seperate_env_dep_analytical_xivmhankel_morerp_rebinned.png'
        )
        plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for mode in modes:
    ax.plot(
        chi2_results[mode]['bin_label'],
        chi2_results[mode]['chi2_red'],
        marker='o', linewidth=2, label=mode
    )
ax.axhline(1, ls='--', lw=1, color='k')
ax.set_xlabel(r'$R_v$ bin')
ax.set_ylabel(r'$\chi^2_{\rm red}$')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    '/disks/shear16/herle/code/plots/chi2red_vs_Rvbin_env_dep_analytical_xivmhankel_morerp_rebinned.png',
    dpi=300, bbox_inches='tight'
)
plt.close()