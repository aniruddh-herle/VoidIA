"""
Standalone plotting script — combined 2×3 figure of void–galaxy shape
correlation fits (wV+) and void–matter correlation (xi_vm), with residuals.

Joint 3-parameter model: A1_v (1-void wvp), A2_v (2-void wvp), b_v (xi_vm).
Reads pre-computed Nautilus posteriors (.npz) and HDF5 measurement files.

Usage:
    python plot_combined_fits_2x3_fullt.py [-box_size 2800] [-run L2800N5040]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from getdist import MCSamples
from mcfit import P2xi
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mnras import *

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("-box_size", type=float, default=2800.0)
parser.add_argument("-run",      type=str,   default="L2800N5040")
args = parser.parse_args()

BOX_SIZE = float(args.box_size)
RUN      = args.run

os.environ["OMP_NUM_THREADS"] = "1"

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOTPATH = Path("/disks/shear16/herle/code/GNAT")
if str(ROOTPATH) not in sys.path:
    sys.path.append(str(ROOTPATH))

CORR_DIR   = Path("/disks/shear16/herle/correlations1")
MODELS_DIR = Path("/disks/shear16/herle/models/models")
PLOTS_DIR  = Path("/disks/shear16/herle/code/plots")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── GNAT imports ───────────────────────────────────────────────────────────────
import pyccl as ccl
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def j0(x):
    out = np.ones_like(x)
    mask = x != 0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def hankel_transform(r, delta_r, k_vals):
    """Spherical Bessel j0 Hankel transform (real-space → k-space)."""
    dr  = np.gradient(r)
    kr  = np.outer(k_vals, r)
    integrand = j0(kr) * (r**2 * dr * delta_r)
    return 4 * np.pi * integrand.sum(axis=1)


def rebin_measurement(rp, wvp, cov, edges):
    """Average wvp/cov into coarser bins; propagates full covariance."""
    n_out = len(edges) - 1
    A = np.zeros((n_out, len(rp)))
    for j in range(n_out):
        mask = (rp >= edges[j]) & (rp < edges[j + 1])
        if mask.sum() > 0:
            A[j, mask] = 1.0 / mask.sum()
    wvp_rb = A @ wvp
    rp_rb  = A @ rp
    cov_rb = A @ cov @ A.T
    nonempty = np.array(
        [(rp >= edges[j]).any() and (rp < edges[j + 1]).any()
         for j in range(n_out)]
    )
    return rp_rb[nonempty], wvp_rb[nonempty], cov_rb[np.ix_(nonempty, nonempty)]


def read_nautilus_samples(path):
    data = np.load(path)
    return data["samples"], data["weights"], data["logl"]


def compute_chi2_red(logl, samples, n_data):
    chi2_eff = -2 * np.max(logl)
    ndof     = n_data - samples.shape[1]
    return chi2_eff / ndof


# ═══════════════════════════════════════════════════════════════════════════════
# Cosmology & grids
# ═══════════════════════════════════════════════════════════════════════════════

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306 - 0.0486,
    Omega_b=0.0486,
    h=0.681,
    sigma8=0.807,
    n_s=0.967,
)

H             = 0.681
redshift_list = np.array([0])
kmin, kmax    = 1e-5, 1e2
pi_max        = 100.0
rmax_trunc    = 400.0               # Mpc/h

rp_array       = np.logspace(-1, np.log10(900), 200)
k_array_single = np.geomspace(kmin, kmax, 1000)
k_array        = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
k_native       = np.logspace(-5, 2, 1_000_001)   # h/Mpc

C1rhocrit = 0.0134
z_i       = 0.0
a_i       = 1.0 / (1.0 + z_i)
scaling   = (C1rhocrit * cosmo_ccl["Omega_m"]
             / np.pi**2
             / ccl.growth_factor(cosmo_ccl, a_i))

fit_rp_edges  = np.logspace(np.log10(6.0), np.log10(500.0), 12)   # ~11 bins
SAVGOL_WINDOW = 1501                                               # must be odd

# ── Fiducial (matter–matter) power spectrum → w_2v ────────────────────────────
pk_input_mm = np.zeros((len(k_array_single), 1, len(redshift_list)))
psHandler   = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm = psHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl, k_input=k_array_single,
        redshift=z, pk_type="nonlinear_matter",
    )
    pk_input_mm[:, 0, idx_z] = -np.pi**2 * pkmm

predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl, redshift_list=redshift_list, logger=logger,
)
predictor.initialise_power_spectra(
    k_input=k_array, pk_input=pk_input_mm, probetype_order=["g+"],
)
predictor.calculate_estimators(
    rp_support=rp_array, beta_shape=0, beta_density=0, pi_max=pi_max,
)
w_arr, _ = predictor.return_all_estimators_as_arrays(
    rp_support=rp_array, r_support=rp_array,
)
w_2v = w_arr[:, 0, 0]   # matter–matter estimator (region VII)

# ── xi_mm via mcfit (computed once) ───────────────────────────────────────────
a_target  = 1.0 / (1.0 + 0.0)
k_hmpc    = np.geomspace(1e-5, 1e3, 8192)
pk_ccl_mm = ccl.nonlin_matter_power(cosmo_ccl, k_hmpc * H, a_target)
pk_hmpc   = pk_ccl_mm * H**3

r_mm, xi_mm_raw = P2xi(k_hmpc, lowring=True)(pk_hmpc, extrap=True)
keep            = (r_mm >= 0.1) & (r_mm <= 300.0)
r_mm, xi_mm_raw = r_mm[keep], xi_mm_raw[keep]
xi_mm_interp    = interp1d(r_mm, xi_mm_raw, kind="cubic",
                            bounds_error=False, fill_value=0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Bin definitions
# ═══════════════════════════════════════════════════════════════════════════════

MODE     = "FIDUCIAL"
REDSHIFT = "0.0"

bin_labels = ["0_10",  "10_15", "15_20", "20_25", "25_30", "30_50"]
void_bins  = [(0,10),  (10,15), (15,20), (20,25), (25,30), (30,50)]
Rv_medians = [8.23,    12.49,   17.17,   21.94,   26.76,   31.92  ]

names  = ["A1_v", "A2_v", "b_v"]
labels = ["A1_v", "A2_v", "b_v"]

# ═══════════════════════════════════════════════════════════════════════════════
# Per-bin computation
# ═══════════════════════════════════════════════════════════════════════════════

bin_data = []

for bin_idx, (void_bin, bin_label, Rv) in enumerate(
        zip(void_bins, bin_labels, Rv_medians)):

    logger.info(f"Processing bin {bin_label}  (Rv={Rv:.2f} Mpc/h) …")

    lower_VI  = 6.0
    upper_VI  = 2.5 * Rv
    lower_VII = 4.0 * Rv
    upper_VII = 500.0

    xi_rmin = max(3.0 * Rv, 150.0)   # well beyond compensation wall
    xi_rmax = 300.0

    # ── Load xi_vm → Hankel → delta_v → w_1v ─────────────────────────────────
    corr_stem = (f"correlations_joint_revolver_{bin_label}hMpc_{RUN}"
                 f"_HYDRO_{MODE}_{REDSHIFT}_env_dep_analytical_morerp_fullt")
    df_corr = pd.read_hdf(CORR_DIR / f"{corr_stem}.h5", key="a")

    r_xi  = df_corr["xi_vm_r"].values
    xi_vm = df_corr["xi_vm"].values

    r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 1000)
    xi_fine = interp1d(r_xi, xi_vm, kind="cubic")(r_fine)
    mask_r  = r_fine <= rmax_trunc
    r_trunc, xi_trunc = r_fine[mask_r], xi_fine[mask_r]

    delta_v = hankel_transform(r_trunc, xi_trunc, k_native) / H**3

    logk_tgt    = np.log10(k_array_single)
    delta_v_tgt = interp1d(
        np.log10(k_native), delta_v, kind="linear",
        bounds_error=False, fill_value=0.0,
    )(logk_tgt)

    logk_uni     = np.linspace(logk_tgt.min(), logk_tgt.max(), 20_000)
    delta_uni    = interp1d(logk_tgt, delta_v_tgt, kind="linear")(logk_uni)
    delta_smooth = savgol_filter(delta_uni, window_length=SAVGOL_WINDOW, polyorder=3)
    delta_v_smoothed = interp1d(logk_uni, delta_smooth, kind="linear")(logk_tgt)

    pk_void = np.zeros_like(pk_input_mm)
    for idx_z in range(len(redshift_list)):
        pk_void[:, 0, idx_z] = -np.pi**2 * delta_v_smoothed

    predictor.initialise_power_spectra(
        k_input=k_array, pk_input=pk_void, probetype_order=["g+"],
    )
    predictor.calculate_estimators(
        rp_support=rp_array, beta_shape=0, beta_density=0, pi_max=pi_max,
    )
    w_arr, _ = predictor.return_all_estimators_as_arrays(
        rp_support=rp_array, r_support=rp_array,
    )
    w_1v = w_arr[:, 0, 0]

    wVI  = w_1v * scaling    # 1-void template (region VI)
    wVII = w_2v * scaling    # 2-void template (region VII)

    model_rp  = rp_array
    mask_mVI  = (model_rp >= lower_VI)  & (model_rp <= upper_VI)
    mask_mVII = (model_rp >= lower_VII) & (model_rp <= upper_VII)

    # ── wvp measurement + JK covariance (rebinned) ───────────────────────────
    jk_stem = (f"wvp_xip_jk_voids_joint_revolver_{bin_label}hMpc_{RUN}"
               f"_HYDRO_{MODE}_{REDSHIFT}_env_dep_analytical_morerp_fullt")
    wvp_JK = np.load(CORR_DIR / f"{jk_stem}.npy")

    rp_meas  = df_corr["wvp_rp"].values
    wvp_meas = df_corr["wvp_xip"].values

    mask_total = (
        ((rp_meas >= lower_VI)  & (rp_meas <= upper_VI)) |
        ((rp_meas >= lower_VII) & (rp_meas <= upper_VII))
    )
    rp_cut   = rp_meas[mask_total]
    corr_wvp = wvp_meas[mask_total]
    jk_cut   = wvp_JK[:, mask_total]
    NPatches = jk_cut.shape[0]

    diff_wvp = jk_cut - jk_cut.mean(axis=0)
    cov_wvp  = ((NPatches - 1) / NPatches) * (diff_wvp.T @ diff_wvp)

    rp_cut, corr_wvp, cov_wvp = rebin_measurement(
        rp_cut, corr_wvp, cov_wvp, fit_rp_edges)
    err_wvp = np.sqrt(np.diag(cov_wvp))

    mask_rVI  = (rp_cut >= lower_VI)  & (rp_cut <= upper_VI)
    mask_rVII = (rp_cut >= lower_VII) & (rp_cut <= upper_VII)

    # ── xi_vm measurement + JK covariance ────────────────────────────────────
    xi_jk_stem = (f"xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_{RUN}"
                  f"_HYDRO_{MODE}_{REDSHIFT}_env_dep_analytical_morerp_fullt")
    xi_vm_JK = np.load(CORR_DIR / f"{xi_jk_stem}.npy")

    rp_xi_vm   = df_corr["xi_vm_r"].values
    xi_vm_meas = df_corr["xi_vm"].values

    # fit-range cut (used in likelihood and residuals)
    mask_xi      = (rp_xi_vm >= xi_rmin) & (rp_xi_vm <= xi_rmax)
    rp_xi_cut    = rp_xi_vm[mask_xi]
    xi_vm_cut    = xi_vm_meas[mask_xi]
    xi_vm_JK_cut = xi_vm_JK[:, mask_xi]

    diff_xi = xi_vm_JK_cut - xi_vm_JK_cut.mean(axis=0)
    cov_xi  = ((NPatches - 1) / NPatches) * (diff_xi.T @ diff_xi)
    err_xi  = np.sqrt(np.diag(cov_xi))

    # full-range arrays (used only for plotting the xi_vm main panel)
    rp_xi_full  = rp_xi_vm
    xi_vm_full  = xi_vm_meas
    diff_xi_full = xi_vm_JK - xi_vm_JK.mean(axis=0)
    err_xi_full  = np.sqrt(np.diag(
        ((NPatches - 1) / NPatches) * (diff_xi_full.T @ diff_xi_full)
    ))

    # ── Load posterior samples ────────────────────────────────────────────────
    npz_path = (MODELS_DIR
                / f"nautilus_chain_revolver_{MODE}_voids_{BOX_SIZE}"
                  f"_Rv{bin_label}Mpc_env_dep_analytical"
                  f"_xivmhankel_morerp_fullt_wl1501.npz")
    samples, weights, logl = read_nautilus_samples(npz_path)

    n_data   = len(rp_cut) + len(rp_xi_cut)
    chi2_red = compute_chi2_red(logl, samples, n_data)

    map_idx = np.argmax(logl)
    A1_best = samples[map_idx, 0]
    A2_best = samples[map_idx, 1]
    bv_best = samples[map_idx, 2]

    wVI_bf  = wVI  * A1_best
    wVII_bf = wVII * A2_best * bv_best
    xi_bf      = bv_best * xi_mm_interp(rp_xi_cut)    # fit range (residuals)
    xi_bf_full = bv_best * xi_mm_interp(rp_xi_full)   # full range (plot)

    logger.info(f"  MAP: A1={A1_best:.3f}  A2={A2_best:.3f}  bv={bv_best:.3f}"
                f"  chi2_red={chi2_red:.3f}")

    # ── Posterior envelope (500 draws) ────────────────────────────────────────
    rng       = np.random.default_rng(42)
    w_norm    = weights / weights.sum()
    idx_draws = rng.choice(len(samples), size=500, p=w_norm)
    A1_d = samples[idx_draws, 0]
    A2_d = samples[idx_draws, 1]
    bv_d = samples[idx_draws, 2]

    wVI_ens  = wVI[None, :]  * A1_d[:, None]
    wVII_ens = wVII[None, :] * (A2_d * bv_d)[:, None]
    xi_ens      = bv_d[:, None] * xi_mm_interp(rp_xi_cut)[None, :]   # fit range
    xi_ens_full = bv_d[:, None] * xi_mm_interp(rp_xi_full)[None, :]  # full range

    p16_VI,  p84_VI  = np.percentile(wVI_ens,       [16, 84], axis=0)
    p16_VII, p84_VII = np.percentile(wVII_ens,      [16, 84], axis=0)
    p16_xi,  p84_xi  = np.percentile(xi_ens,        [16, 84], axis=0)
    p16_xi_full, p84_xi_full = np.percentile(xi_ens_full, [16, 84], axis=0)

    # ── Residuals ─────────────────────────────────────────────────────────────
    m_at_data = np.full_like(rp_cut, np.nan)
    m_at_data[mask_rVI]  = np.interp(
        rp_cut[mask_rVI],  model_rp[mask_mVI],  wVI_bf[mask_mVI])
    m_at_data[mask_rVII] = np.interp(
        rp_cut[mask_rVII], model_rp[mask_mVII], wVII_bf[mask_mVII])

    residual_wvp = 100.0 * (corr_wvp - m_at_data)  / np.abs(corr_wvp)
    frac_err_wvp = 100.0 * err_wvp                  / np.abs(corr_wvp)

    residual_xi  = 100.0 * (xi_vm_cut - xi_bf) / np.abs(xi_vm_cut)
    frac_err_xi  = 100.0 * err_xi              / np.abs(xi_vm_cut)

    bin_data.append(dict(
        void_bin=void_bin, bin_label=bin_label, Rv=Rv,
        lower_VI=lower_VI, upper_VI=upper_VI,
        lower_VII=lower_VII, upper_VII=upper_VII,
        model_rp=model_rp,
        mask_mVI=mask_mVI, mask_mVII=mask_mVII,
        wVI_bf=wVI_bf, wVII_bf=wVII_bf,
        p16_VI=p16_VI, p84_VI=p84_VI,
        p16_VII=p16_VII, p84_VII=p84_VII,
        rp_cut=rp_cut, corr_wvp=corr_wvp, err_wvp=err_wvp,
        mask_rVI=mask_rVI, mask_rVII=mask_rVII,
        residual_wvp=residual_wvp, frac_err_wvp=frac_err_wvp,
        rp_xi_cut=rp_xi_cut, xi_vm_cut=xi_vm_cut, err_xi=err_xi,
        xi_bf=xi_bf, p16_xi=p16_xi, p84_xi=p84_xi,
        residual_xi=residual_xi, frac_err_xi=frac_err_xi,
        rp_xi_full=rp_xi_full, xi_vm_full=xi_vm_full, err_xi_full=err_xi_full,
        xi_bf_full=xi_bf_full, p16_xi_full=p16_xi_full, p84_xi_full=p84_xi_full,
        xi_rmin=xi_rmin, xi_rmax=xi_rmax,
        chi2_red=chi2_red,
        A1_best=A1_best, A2_best=A2_best, bv_best=bv_best,
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# Combined 2 × 3 figure
# ═══════════════════════════════════════════════════════════════════════════════
#
# Outer: 2 rows × 3 cols (row 0 → bins 0,1,2 | row 1 → bins 3,4,5)
# Inner: 4 stacked panels per cell
#   [0] wvp main          height 3
#   [1] wvp residuals     height 1
#   [2] xi_vm main        height 2
#   [3] xi_vm residuals   height 1
#
# ═══════════════════════════════════════════════════════════════════════════════

COLOR = "#332288"
GREY  = "#999999"
SHADE = "#cccccc"

fig = plt.figure(figsize=(18, 22), layout="constrained")
fig.patch.set_facecolor("white")

outer = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.12,
    wspace=0.30,
)

for panel_idx, bd in enumerate(bin_data):
    row_o = panel_idx // 3
    col_o = panel_idx % 3

    inner = gridspec.GridSpecFromSubplotSpec(
        4, 1,
        subplot_spec=outer[row_o, col_o],
        height_ratios=[3, 1, 2, 1],
        hspace=0.08,
    )
    ax_w   = fig.add_subplot(inner[0])
    ax_wr  = fig.add_subplot(inner[1], sharex=ax_w)
    ax_xi  = fig.add_subplot(inner[2])
    ax_xir = fig.add_subplot(inner[3], sharex=ax_xi)

    mrp  = bd["model_rp"]
    mVI  = bd["mask_mVI"]
    mVII = bd["mask_mVII"]
    rxi  = bd["rp_xi_cut"]

    # ── wvp: fit-range shading ────────────────────────────────────────────────
    for ax in (ax_w, ax_wr):
        ax.axvspan(bd["lower_VI"],  bd["upper_VI"],  color=SHADE, alpha=0.35, zorder=0, lw=0)
        ax.axvspan(bd["lower_VII"], bd["upper_VII"], color=SHADE, alpha=0.35, zorder=0, lw=0)
        ax.set_xscale("log")

    # ── wvp: posterior band ───────────────────────────────────────────────────
    ax_w.fill_between(
        mrp[mVI], mrp[mVI] * bd["p16_VI"][mVI], mrp[mVI] * bd["p84_VI"][mVI],
        color=COLOR, alpha=0.18, zorder=1,
    )
    ax_w.fill_between(
        mrp[mVII], mrp[mVII] * bd["p16_VII"][mVII], mrp[mVII] * bd["p84_VII"][mVII],
        color=COLOR, alpha=0.18, zorder=1,
    )

    # ── wvp: best-fit curves (both solid) ────────────────────────────────────
    ax_w.plot(mrp[mVI],  mrp[mVI]  * bd["wVI_bf"][mVI],
              color=COLOR, lw=2.0, ls="-", zorder=3, solid_capstyle="round")
    ax_w.plot(mrp[mVII], mrp[mVII] * bd["wVII_bf"][mVII],
              color=COLOR, lw=2.0, ls="-", zorder=3, solid_capstyle="round")

    # ── wvp: data ─────────────────────────────────────────────────────────────
    ax_w.errorbar(
        bd["rp_cut"], bd["rp_cut"] * bd["corr_wvp"],
        yerr=bd["rp_cut"] * bd["err_wvp"],
        fmt="o", ms=4, capsize=2.5, elinewidth=1.0, color=COLOR, zorder=4,
    )
    ax_w.axhline(0, color=GREY, ls=":", lw=0.8)

    # ── wvp: axes ─────────────────────────────────────────────────────────────
    ax_w.set_xlim(3, 700)
    ax_w.tick_params(axis="x", which="both", labelbottom=False)
    ax_w.tick_params(axis="both", labelsize=11)
    correct_tick_marks(ax_w)

    vb  = bd["void_bin"]
    ttl = (
        r"$R_v = $" + f"{vb[0]}–{vb[1]}" + r" Mpc/$h$"
        + r"    $\langle R_v\rangle = $" + f"{bd['Rv']:.2f}" + r" Mpc/$h$"
        + "\n"
        + r"$A_1^{\rm MAP}=$" + f"{bd['A1_best']:.3f}"
        + r"   $A_2^{\rm MAP}=$" + f"{bd['A2_best']:.3f}"
        + r"   $b_v^{\rm MAP}=$" + f"{bd['bv_best']:.3f}"
    )
    ax_w.set_title(ttl, fontsize=11, pad=4, linespacing=1.5)

    if col_o == 0:
        ax_w.set_ylabel(r"$r_p\,w_{\mathrm{V}+}\ [\mathrm{Mpc}^2 h^{-2}]$", fontsize=12)

    # ── wvp residuals ─────────────────────────────────────────────────────────
    ax_wr.errorbar(
        bd["rp_cut"], bd["residual_wvp"], yerr=bd["frac_err_wvp"],
        fmt="o", ms=4, capsize=2.5, elinewidth=1.0, color=COLOR, zorder=3,
    )
    ax_wr.axhline(0, color=COLOR, ls="--", lw=1.2)
    ax_wr.set_ylim(-50, 50)
    ax_wr.tick_params(axis="both", labelsize=10)
    correct_tick_marks(ax_wr)
    ax_wr.set_xlabel(r"$r_p\ [\mathrm{Mpc}\,h^{-1}]$", fontsize=12)

    if col_o == 0:
        ax_wr.set_ylabel(r"$\Delta/|D|\ [\%]$", fontsize=11)

    # ── xi_vm: fit-regime shading ─────────────────────────────────────────────
    rxi_full = bd["rp_xi_full"]
    ax_xi.axvspan(bd["xi_rmin"], bd["xi_rmax"],
                  color=SHADE, alpha=0.35, zorder=0, lw=0)

    # ── xi_vm: posterior band (full range) ────────────────────────────────────
    ax_xi.fill_between(
        rxi_full, rxi_full**2 * bd["p16_xi_full"], rxi_full**2 * bd["p84_xi_full"],
        color=COLOR, alpha=0.18, zorder=1,
    )

    # ── xi_vm: best-fit curve (full range) ───────────────────────────────────
    ax_xi.plot(rxi_full, rxi_full**2 * bd["xi_bf_full"],
               color=COLOR, lw=2.0, zorder=3)

    # ── xi_vm: data (full range) ──────────────────────────────────────────────
    ax_xi.errorbar(
        rxi_full, rxi_full**2 * bd["xi_vm_full"], yerr=rxi_full**2 * bd["err_xi_full"],
        fmt="o", ms=4, capsize=2.5, elinewidth=1.0, color=COLOR, zorder=4,
    )
    ax_xi.axhline(0, color=GREY, ls=":", lw=0.8)
    ax_xi.set_xscale("log")
    ax_xi.tick_params(axis="x", which="both", labelbottom=False)
    ax_xi.tick_params(axis="both", labelsize=11)
    correct_tick_marks(ax_xi)

    if col_o == 0:
        ax_xi.set_ylabel(r"$r^2\,\xi_{vm}(r)\ [\mathrm{Mpc}^2 h^{-2}]$", fontsize=12)

    # ── xi_vm residuals ───────────────────────────────────────────────────────
    ax_xir.errorbar(
        rxi, bd["residual_xi"], yerr=bd["frac_err_xi"],
        fmt="o", ms=4, capsize=2.5, elinewidth=1.0, color=COLOR, zorder=3,
    )
    ax_xir.axhline(0, color=COLOR, ls="--", lw=1.2)
    ax_xir.set_ylim(-160, 160)
    ax_xir.set_xscale("log")
    ax_xir.tick_params(axis="both", labelsize=10)
    correct_tick_marks(ax_xir)
    ax_xir.set_xlabel(r"$r\ [\mathrm{Mpc}\,h^{-1}]$", fontsize=12)

    if col_o == 0:
        ax_xir.set_ylabel(r"$\Delta/|D|\ [\%]$", fontsize=11)

    # ── chi2 annotation (bottom-right of xi_vm residual panel) ───────────────
    ax_xir.text(
        0.97, 0.88,
        rf"$\chi^2_{{\rm red}} = {bd['chi2_red']:.2f}$",
        transform=ax_xir.transAxes,
        fontsize=11, va="top", ha="right", color=COLOR,
    )

out_path = (PLOTS_DIR
            / f"combined_fits_2x3_{MODE}_{BOX_SIZE}_{RUN}"
              f"_env_dep_analytical_xivmhankel_morerp_fullt_wl1501.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
logger.info(f"Saved → {out_path}")