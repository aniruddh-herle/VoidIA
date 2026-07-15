import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mnras import *

from getdist import plots, MCSamples
import pyccl as ccl
import glob
plt.rcdefaults()
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 22


def read_nautilus_samples(npz_filename):
    data = np.load(npz_filename)
    samples = data['samples']        # shape (Nsamples, Nparams)
    weights = data['weights']        # shape (Nsamples,)
    logl = data['logl']              # shape (Nsamples,)
    return samples, weights, logl

def compute_chi2(logl, samples, corr):
    logL_max = np.max(logl)   # MAP = maximum likelihood for flat priors
    chi2_eff = -2 * logL_max

    Ndata = len(corr)
    Nparams_flat = samples.shape[1]

    Ndof = Ndata - Nparams_flat   # e.g. Nparams_flat = 5 for TATT
    chi2_red = chi2_eff / Ndof
    return chi2_eff, chi2_red


names =  ['A1_v', 'b_v']
labels = ['A1_v', 'b_v']

# modes = [
#     'FIDUCIAL',
#     'WEAK_AGN',
#     'STRONG_AGN',
#     'STRONGER_AGN',
#     'STRONGEST_AGN',
#     'STRONG_SUPERNOVA'
# ]

modes = [
    'FIDUCIAL'
]



FLAMINGO_colors = [
    '#117733',
    '#ABD0E6',
    '#6AAED6',
    '#3787C0',
    '#105BA4',
    '#FF8C40'
]

FLAMINGO_labels = [
    r"L1_m9",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    r"M_*-\sigma$"
]

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

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
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
pi_max = 100.0  # in Mpc


# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts).
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T

h = 0.681

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo_ccl['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo_ccl, scale_factor)

Rv = 15.0
# masks
lower_rp_VI = 6.0
upper_rp_VI = 2.5*Rv

lower_rp_VII = 4.0*Rv
upper_rp_VII = 500.0

run = 'L1000N1800'
redshift = '0.0'

Nsamp = 500

# -------------------------------------------------
# Storage across modes
# -------------------------------------------------
getdist_samples_all = []
best_fit_results_all = {}
chi2_results_all = {}
model_results_all = {}

for i, mode in enumerate(modes):

    # -------------------------------------------------
    # Posterior samples + best fit (per mode)
    # -------------------------------------------------
    filename = f'/disks/shear16/herle/models/models/nautilus_chain_revolver_{mode}_voids_1000_Rv10_20Mpc_pimax100_0.0_JK125.npz'
    samples, weights, logl = read_nautilus_samples(filename)
    getdist_samples = MCSamples(samples=samples,
                                weights=weights,
                                names=names,
                                labels=labels,
                                ignore_rows=0,
                                label=FLAMINGO_labels[i])

    getdist_samples_all.append(getdist_samples)

    best_fit_results = {}

    for name in names:
        d = getdist_samples.get1DDensity(name)
        xyind = np.where(d.P == np.amax(d.P))
        max_density = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)

        lower_error = errors_density[0]
        upper_error = errors_density[1]

        best_fit_results[name] = {
            'best_fit': float(max_density),
            'lower_68': float(lower_error),
            'upper_68': float(upper_error)
        }

    best_fit_results_all[mode] = best_fit_results

    # -------------------------------------------------
    # 2-void term model (per mode)
    # -------------------------------------------------
    pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

    # Fill pk_input with g+ and gg power spectra
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

    # -------------------------------------------------
    # 1-void term model (per mode)
    # -------------------------------------------------
    delta_v = np.load(f'/disks/shear16/herle/code/zobov_{modes[i]}_void_delta_k_bin0_raw.npy')

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

    wVII = w_2v.copy()
    wVII *= scaling
    wVII *= h

    wVI = w_1v.copy()
    wVI *= scaling
    wVI *= h

    model_rp = rp_array * h

    # model_rp = model_rp /h  # in Mpc
    print(f"[{mode}] Model rp range: ", model_rp.min(), model_rp.max())

    model_mask_VI = (model_rp >= lower_rp_VI) & (model_rp <= upper_rp_VI)
    model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

    # # -------------------------------------------------
    # # Data (per mode)
    # # -------------------------------------------------
    meas_file = f"/disks/shear16/herle/correlations1/correlations_revolver_10_20Mpc_{run}_HYDRO_{modes[i]}_{redshift}_JK.h5"
    df = pd.read_hdf(meas_file, key='a')
    rp_meas = df['wgp_rp'].values
    wvp_meas = df['wgp_xip'].values
    err_meas = df['wgp_std_from_cov'].values

    # # bootstrap version

    # # -------------------------------------------------
    # # Data (per mode)
    # # -------------------------------------------------

    # meas_file = (
    #     f"/disks/shear16/herle/correlations1/"
    #     f"correlations_revolver_10_20Mpc_{run}_HYDRO_{modes[i]}_{redshift}.h5"
    # )

    # df = pd.read_hdf(meas_file, key='a')

    # rp_meas = df['wgp_rp'].values
    # wvp_meas = df['wgp_xip'].values


    # # -------------------------------------------------
    # # Load bootstrap realizations
    # # -------------------------------------------------

    # boot_files = sorted(
    #     glob.glob(
    #         f"/disks/shear16/herle/correlations1/bootstrap1/"
    #         f"correlations_revolver_10_20Mpc_{run}_HYDRO_{modes[i]}_{redshift}_bootstrap*.h5"
    #     )
    # )

    # print(f"[{mode}] Found {len(boot_files)} bootstrap realizations.")

    # bootstrap_vectors = []

    # for f in boot_files:
    #     df_boot = pd.read_hdf(f, key='a')
    #     bootstrap_vectors.append(
    #         df_boot['wgp_xip'].values
    #     )

    # bootstrap_vectors = np.asarray(bootstrap_vectors)

    # print(f"[{mode}] Bootstrap shape:", bootstrap_vectors.shape)


    # # -------------------------------------------------
    # # Bootstrap covariance
    # # -------------------------------------------------

    # K = bootstrap_vectors.shape[0]

    # bootstrap_mean = np.mean(
    #     bootstrap_vectors,
    #     axis=0
    # )

    # delta_boot = bootstrap_vectors - bootstrap_mean

    # bootstrap_cov = (
    #     delta_boot.T @ delta_boot
    # ) / (K - 1)


    # # Bootstrap errors
    # err_meas = np.sqrt(
    #     np.diag(bootstrap_cov)
    # )

    # -------------------------------------------------
    # build masks (same logic as before)
    # -------------------------------------------------
    mask_data_VI  = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
    mask_data_VII = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)

    mask_data_total = mask_data_VI | mask_data_VII

    corr   = wvp_meas[mask_data_total]

    chi2_eff, chi2_red = compute_chi2(logl, samples, corr)
    print(f'[{mode}] Reduced Chi2: ', chi2_red)
    chi2_results_all[mode] = {'chi2_eff': chi2_eff, 'chi2_red': chi2_red}

    # -------------------------------------------------
    # best-fit amplitudes (marginal getdist mode)
    # -------------------------------------------------
    A1_best = best_fit_results['A1_v']['best_fit']
    b_best  = best_fit_results['b_v']['best_fit']

    print(f'[{mode}] A1_v best fit: ', A1_best)
    print(f'[{mode}] b_v best fit: ', b_best)

    wVII_bestfit = wVII * A1_best * b_best
    wVI_bestfit  = wVI  * A1_best

    # -------------------------------------------------
    # Draw posterior samples (unchanged)
    # -------------------------------------------------
    weights_norm = weights / np.sum(weights)
    inds = np.random.choice(len(samples), size=Nsamp, p=weights_norm)

    posterior_draws = samples[inds]

    A1_draws = posterior_draws[:, 0]
    b_draws  = posterior_draws[:, 1]

    wVI_ens = []
    wVII_ens = []

    base_wVI = wVI.copy()
    base_wVII = wVII.copy()

    for A1, b in zip(A1_draws, b_draws):
        wVI_ens.append(base_wVI * A1)
        wVII_ens.append(base_wVII * A1 * b)

    wVI_ens = np.array(wVI_ens)
    wVII_ens = np.array(wVII_ens)

    p16_VI, p50_VI, p84_VI = np.percentile(wVI_ens, [16, 50, 84], axis=0)
    p16_VII, p50_VII, p84_VII = np.percentile(wVII_ens, [16, 50, 84], axis=0)

    # -------------------------------------------------
    # stash everything this mode needs for the combined plot
    # -------------------------------------------------
    model_results_all[mode] = dict(
        model_rp=model_rp,
        model_mask_VI=model_mask_VI,
        model_mask_VII=model_mask_VII,
        wVI_bestfit=wVI_bestfit,
        wVII_bestfit=wVII_bestfit,
        p16_VI=p16_VI, p84_VI=p84_VI,
        p16_VII=p16_VII, p84_VII=p84_VII,
        rp_meas=rp_meas,
        wvp_meas=wvp_meas,
        err_meas=err_meas,
        mask_data_VI=mask_data_VI,
        mask_data_VII=mask_data_VII,
        color=FLAMINGO_colors[i],
        label=FLAMINGO_labels[i],
    )


# =====================================================
# Combined posterior triangle plot (all modes together)
# =====================================================

g = plots.get_subplot_plotter()
g.settings.fig_width_inch = 12
g.settings.legend_fontsize = 25
g.settings.fontsize = 25
g.settings.axes_labelsize = 30
g.settings.axes_fontsize = 25


g.triangle_plot(
    getdist_samples_all,
    ['A1_v', 'b_v'],
    filled=[True]*len(modes),
    marker_args={'lw':1.5},
    contour_colors=FLAMINGO_colors,
    legend_labels=FLAMINGO_labels,
    legend_kwargs={'frameon': False, 'ncol': 2}
    )


plt.savefig('/disks/shear16/herle/code/plots/zobov_all_modes_void_model.png')


# =====================================================
# Combined model vs data plot (all modes together, with residuals)
# =====================================================

fig, (ax_main, ax_res) = plt.subplots(
    2, 1,
    figsize=(14, 14),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)

for mode in modes:
    res = model_results_all[mode]

    model_rp = res['model_rp']
    model_mask_VI = res['model_mask_VI']
    model_mask_VII = res['model_mask_VII']
    wVI_bestfit = res['wVI_bestfit']
    wVII_bestfit = res['wVII_bestfit']
    p16_VI, p84_VI = res['p16_VI'], res['p84_VI']
    p16_VII, p84_VII = res['p16_VII'], res['p84_VII']
    rp_meas = res['rp_meas']
    wvp_meas = res['wvp_meas']
    err_meas = res['err_meas']
    mask_data_VI = res['mask_data_VI']
    mask_data_VII = res['mask_data_VII']
    color = res['color']
    label = res['label']

    # -------------------------------------------------
    # Main panel
    # -------------------------------------------------
    ax_main.plot(model_rp, model_rp * wVII_bestfit, alpha=0.2, ls='--', color=color)
    ax_main.plot(model_rp, model_rp * wVI_bestfit, alpha=0.2, ls='--', color=color)

    ax_main.errorbar(
        rp_meas,
        rp_meas * wvp_meas,
        yerr=rp_meas * err_meas,
        fmt='o',
        capsize=3,
        color=color,
        label=label
    )

    line_VII, = ax_main.plot(
        model_rp[model_mask_VII],
        model_rp[model_mask_VII] * wVII_bestfit[model_mask_VII],
        color=color
    )

    line_VI, = ax_main.plot(
        model_rp[model_mask_VI],
        model_rp[model_mask_VI] * wVI_bestfit[model_mask_VI],
        color=color
    )

    ax_main.fill_between(
        model_rp[model_mask_VI],
        model_rp[model_mask_VI] * p16_VI[model_mask_VI],
        model_rp[model_mask_VI] * p84_VI[model_mask_VI],
        alpha=0.25,
        color=color
    )

    ax_main.fill_between(
        model_rp[model_mask_VII],
        model_rp[model_mask_VII] * p16_VII[model_mask_VII],
        model_rp[model_mask_VII] * p84_VII[model_mask_VII],
        alpha=0.25,
        color=color
    )

    # -------------------------------------------------
    # Residuals
    # -------------------------------------------------
    model_at_data = np.full_like(rp_meas, np.nan)

    model_at_data[mask_data_VI] = np.interp(
        rp_meas[mask_data_VI],
        model_rp[model_mask_VI],
        wVI_bestfit[model_mask_VI]
    )

    model_at_data[mask_data_VII] = np.interp(
        rp_meas[mask_data_VII],
        model_rp[model_mask_VII],
        wVII_bestfit[model_mask_VII]
    )

    residual = 100.0 * (wvp_meas - model_at_data) / wvp_meas
    frac_err = 100.0 * err_meas / np.abs(wvp_meas)

    ax_res.errorbar(
        rp_meas,
        residual,
        yerr=frac_err,
        fmt='o',
        capsize=3,
        color=color
    )

# shaded 1-void/2-void regions (mode-independent, drawn once)
ax_main.axvspan(lower_rp_VI, upper_rp_VI, color='grey', alpha=0.08, zorder=0)
ax_main.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)

ax_res.axvspan(lower_rp_VI, upper_rp_VI, color='grey', alpha=0.08, zorder=0)
ax_res.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)

ax_main.axhline(0, color='k', ls=':')

ax_main.set_xscale('log')
ax_main.set_xlim(1, 600)
ax_main.set_ylim(-7, 7)

ax_main.set_ylabel(r'$r_p w_{\mathrm{V}+}\,[\mathrm{Mpc}^2/h^2]$')

correct_tick_marks(ax_main)

ax_main.legend(frameon=False)

ax_res.axhline(0, color='k', ls=':')

ax_res.set_xscale('log')
ax_res.set_ylim(-200, 200)

ax_res.set_xlabel(r'$r_p$ [Mpc/h]')
ax_res.set_ylabel(r'$(\mathrm{Data}-\mathrm{Model})/\mathrm{Data}$ [%]')

plt.savefig('/disks/shear16/herle/code/plots/zobov_all_modes_fit.png')