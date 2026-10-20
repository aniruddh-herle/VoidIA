"""
Standalone plotting script: combined 2×3 figure of void–galaxy shape
correlation fits (wV+) with residual panels.

Reads pre-computed Nautilus posterior chains (.npz) and HDF5 measurement
files; recomputes the model curves from scratch before plotting.

Usage:
    python plot_combined_fits_2x3.py [-box_size 2800] [-run L2800N5040]
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
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mnras import *

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("-box_size", type=float, default=2800.0)
parser.add_argument("-run",      type=str,   default="L2800N5040")
args = parser.parse_args()

BOX_SIZE = float(args.box_size)   # ensure float even if default was used
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
    kr  = np.outer(k_vals, r)          # (Nk, Nr)
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


def compute_chi2_red(logl, samples, corr):
    chi2_eff = -2 * np.max(logl)
    ndof     = len(corr) - samples.shape[1]
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

H           = 0.681               # little-h
redshift_list = np.array([0])
kmin, kmax  = 1e-5, 1e2
pi_max      = 100.0
rmax_trunc  = 400.0               # Mpc/h

rp_array       = np.logspace(-1, np.log10(900), 200)
k_array_single = np.geomspace(kmin, kmax, 1000)
k_array        = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
k_native       = np.logspace(-5, 2, 1_000_001)   # h/Mpc, fine grid for Hankel

C1rhocrit  = 0.0134
z_i        = 0.0
a_i        = 1.0 / (1.0 + z_i)
scaling    = (C1rhocrit * cosmo_ccl["Omega_m"]
              / np.pi**2
              / ccl.growth_factor(cosmo_ccl, a_i))

fit_rp_edges = np.logspace(np.log10(6.0), np.log10(500.0), 12)   # ~11 bins

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
w_2v = w_arr[:, 0, 0]   # matter–matter estimator (for region VII)

# ═══════════════════════════════════════════════════════════════════════════════
# Bin definitions
# ═══════════════════════════════════════════════════════════════════════════════

MODE      = "FIDUCIAL"
REDSHIFT  = "0.0"

bin_labels       = ["0_10",  "10_15", "15_20", "20_25", "25_30", "30_50"]
void_bins        = [(0,10),  (10,15), (15,20), (20,25), (25,30), (30,50)]
Rv_medians       = [8.23,    12.49,   17.28,   22.12,   26.98,   32.56  ]

names  = ["A1_v", "b_v"]
labels = ["A1_v", "b_v"]

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

    # ── Load xi_vm ────────────────────────────────────────────────────────────
    corr_stem = (f"correlations_joint_revolver_{bin_label}hMpc_{RUN}"
                 f"_HYDRO_{MODE}_{REDSHIFT}_env_dep_analytical_morerp_fullt")
    df_corr = pd.read_hdf(CORR_DIR / f"{corr_stem}.h5", key="a")

    r_xi  = df_corr["xi_vm_r"].values
    xi_vm = df_corr["xi_vm"].values

    # Interpolate to fine grid, truncate, then Hankel-transform
    r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 1000)
    xi_fine = interp1d(r_xi, xi_vm, kind="cubic")(r_fine)
    mask_r  = r_fine <= rmax_trunc
    r_trunc, xi_trunc = r_fine[mask_r], xi_fine[mask_r]

    delta_v = hankel_transform(r_trunc, xi_trunc, k_native) / H**3

    # Resample onto k_array_single and smooth
    logk_tgt = np.log10(k_array_single)
    delta_v_tgt = interp1d(
        np.log10(k_native), delta_v, kind="linear",
        bounds_error=False, fill_value=0.0,
    )(logk_tgt)

    N_log        = 20_000
    logk_uni     = np.linspace(logk_tgt.min(), logk_tgt.max(), N_log)
    delta_uni    = interp1d(logk_tgt, delta_v_tgt, kind="linear")(logk_uni)
    win_len      = 1501   # ensure odd
    delta_smooth = savgol_filter(delta_uni, win_len, polyorder=3)
    delta_v_smoothed = interp1d(logk_uni, delta_smooth, kind="linear")(logk_tgt)

    # ── Void power spectrum → w_1v (region VI model) ──────────────────────────
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

    wVI  = w_1v  * scaling    # void profile model
    wVII = w_2v  * scaling    # matter–matter model

    model_rp      = rp_array
    mask_mVI      = (model_rp >= lower_VI)  & (model_rp <= upper_VI)
    mask_mVII     = (model_rp >= lower_VII) & (model_rp <= upper_VII)

    # ── Measurement + JK covariance ───────────────────────────────────────────
    jk_stem = (f"wvp_xip_jk_voids_joint_revolver_{bin_label}hMpc_{RUN}"
               f"_HYDRO_{MODE}_{REDSHIFT}_env_dep_analytical_morerp_fullt")
    wvp_JK = np.load(CORR_DIR / f"{jk_stem}.npy")

    rp_meas  = df_corr["wvp_rp"].values
    wvp_meas = df_corr["wvp_xip"].values

    mask_d_VI  = (rp_meas >= lower_VI)  & (rp_meas <= upper_VI)
    mask_d_VII = (rp_meas >= lower_VII) & (rp_meas <= upper_VII)
    mask_total = mask_d_VI | mask_d_VII

    rp_cut   = rp_meas[mask_total]
    corr_vec = wvp_meas[mask_total]
    jk_cut   = wvp_JK[:, mask_total]

    NP       = jk_cut.shape[0]
    diff     = jk_cut - jk_cut.mean(axis=0)
    cov_mat  = (NP - 1) / NP * (diff.T @ diff)

    rp_cut, corr_vec, cov_arr = rebin_measurement(
        rp_cut, corr_vec, cov_mat, fit_rp_edges)
    inv_cov  = np.linalg.pinv(cov_arr)
    err_cut  = np.sqrt(np.diag(cov_arr))

    mask_rVI  = (rp_cut >= lower_VI)  & (rp_cut <= upper_VI)
    mask_rVII = (rp_cut >= lower_VII) & (rp_cut <= upper_VII)

    # ── Posterior samples ─────────────────────────────────────────────────────
    npz_path = (MODELS_DIR
                / f"nautilus_chain_revolver_{MODE}_voids_{BOX_SIZE}"
                  f"_Rv{bin_label}Mpc_seperate_env_dep_analytical"
                  f"_xivmhankel_morerp_rebinned_wl1501_fullt.npz")
    samples, weights, logl = read_nautilus_samples(npz_path)
    chi2_red = compute_chi2_red(logl, samples, corr_vec)

    map_idx = np.argmax(logl)
    A1_best = samples[map_idx, 0]
    bv_best = samples[map_idx, 1]

    wVI_bf  = wVI  * A1_best
    wVII_bf = wVII * A1_best * bv_best

    # ── Marginal statistics (for annotation) ─────────────────────────────────
    getdist_obj = MCSamples(
        samples=samples, weights=weights,
        names=names, labels=labels, ignore_rows=0,
    )
    def _marginal_mode(param):
        d   = getdist_obj.get1DDensity(param)
        idx = np.argmax(d.P)
        lim = d.getLimits(0.68)
        val = d.x[idx]
        return val, lim[1] - val, val - lim[0]   # mode, +err, -err

    A1_mode, A1_plus, A1_minus = _marginal_mode("A1_v")
    bv_mode, bv_plus,  bv_minus = _marginal_mode("b_v")

    # ── Posterior envelope (500 draws) ────────────────────────────────────────
    rng          = np.random.default_rng(42)
    w_norm       = weights / weights.sum()
    idx_draws    = rng.choice(len(samples), size=500, p=w_norm)
    A1_d = samples[idx_draws, 0]
    bv_d = samples[idx_draws, 1]

    wVI_ens  = wVI[None, :]  * A1_d[:, None]
    wVII_ens = wVII[None, :] * (A1_d * bv_d)[:, None]

    p16_VI,  p84_VI  = np.percentile(wVI_ens,  [16, 84], axis=0)
    p16_VII, p84_VII = np.percentile(wVII_ens, [16, 84], axis=0)

    # ── Model evaluated at rebinned data rp (for residuals) ───────────────────
    m_at_data = np.full_like(rp_cut, np.nan)
    m_at_data[mask_rVI]  = np.interp(
        rp_cut[mask_rVI],  model_rp[mask_mVI],  wVI_bf[mask_mVI])
    m_at_data[mask_rVII] = np.interp(
        rp_cut[mask_rVII], model_rp[mask_mVII], wVII_bf[mask_mVII])

    residual = 100.0 * (corr_vec - m_at_data) / np.abs(corr_vec)
    frac_err = 100.0 * err_cut / np.abs(corr_vec)

    bin_data.append(dict(
        void_bin=void_bin, bin_label=bin_label, Rv=Rv,
        lower_VI=lower_VI, upper_VI=upper_VI,
        lower_VII=lower_VII, upper_VII=upper_VII,
        model_rp=model_rp,
        mask_mVI=mask_mVI, mask_mVII=mask_mVII,
        wVI_bf=wVI_bf, wVII_bf=wVII_bf,
        p16_VI=p16_VI, p84_VI=p84_VI,
        p16_VII=p16_VII, p84_VII=p84_VII,
        rp_cut=rp_cut, corr=corr_vec, err_cut=err_cut,
        mask_rVI=mask_rVI, mask_rVII=mask_rVII,
        residual=residual, frac_err=frac_err,
        chi2_red=chi2_red,
        A1_best=A1_best, bv_best=bv_best,
        A1_mode=A1_mode, A1_plus=A1_plus, A1_minus=A1_minus,
        bv_mode=bv_mode, bv_plus=bv_plus, bv_minus=bv_minus,
    ))
    logger.info(f"  chi2_red = {chi2_red:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Combined 2 × 3 figure
# ═══════════════════════════════════════════════════════════════════════════════
#
# Layout: outer 2×3 grid; each cell contains a stacked [fit / residual] pair.
#   outer row 0 → bins 0,1,2   |   outer row 1 → bins 3,4,5
#   inner height ratio: 3:1 (fit : residual)
#
# ═══════════════════════════════════════════════════════════════════════════════

COLOR  = "#332288"      # data + model colour
GREY   = "#999999"      # zero lines
SHADE  = "#cccccc"      # fit-range shading

fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor("white")

outer = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.42,    # vertical gap between the two outer rows
    wspace=0.30,    # horizontal gap between columns
)

for panel_idx, bd in enumerate(bin_data):
    row_o = panel_idx // 3
    col_o = panel_idx % 3

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[row_o, col_o],
        height_ratios=[3.2, 1],
        hspace=0.06,
    )
    ax_m = fig.add_subplot(inner[0])
    ax_r = fig.add_subplot(inner[1], sharex=ax_m)

    mrp  = bd["model_rp"]
    mVI  = bd["mask_mVI"]
    mVII = bd["mask_mVII"]

    # ── fit-range shading ─────────────────────────────────────────────────────
    for ax in (ax_m, ax_r):
        ax.axvspan(bd["lower_VI"],  bd["upper_VI"],  color=SHADE, alpha=0.35, zorder=0, lw=0)
        ax.axvspan(bd["lower_VII"], bd["upper_VII"], color=SHADE, alpha=0.35, zorder=0, lw=0)
        ax.set_xscale("log")

    # ── posterior band ────────────────────────────────────────────────────────
    ax_m.fill_between(
        mrp[mVI],
        mrp[mVI] * bd["p16_VI"][mVI],
        mrp[mVI] * bd["p84_VI"][mVI],
        color=COLOR, alpha=0.18, zorder=1,
    )
    ax_m.fill_between(
        mrp[mVII],
        mrp[mVII] * bd["p16_VII"][mVII],
        mrp[mVII] * bd["p84_VII"][mVII],
        color=COLOR, alpha=0.18, zorder=1,
    )

    # ── best-fit model curves ─────────────────────────────────────────────────
    ax_m.plot(mrp[mVI],  mrp[mVI]  * bd["wVI_bf"][mVI],
              color=COLOR, lw=2.0, zorder=3, solid_capstyle="round")
    ax_m.plot(mrp[mVII], mrp[mVII] * bd["wVII_bf"][mVII],
              color=COLOR, lw=2.0, zorder=3, solid_capstyle="round")

    # ── data ──────────────────────────────────────────────────────────────────
    ax_m.errorbar(
        bd["rp_cut"],
        bd["rp_cut"] * bd["corr"],
        yerr=bd["rp_cut"] * bd["err_cut"],
        fmt="o", ms=4, capsize=2.5, elinewidth=1.0,
        color=COLOR, zorder=4,
    )

    # ── reference line ────────────────────────────────────────────────────────
    ax_m.axhline(0, color=GREY, ls=":", lw=0.8)

    # ── axes cosmetics (main) ─────────────────────────────────────────────────
    ax_m.set_xlim(3, 700)
    ax_m.tick_params(axis="x", which="both", labelbottom=False)
    ax_m.tick_params(axis="both", labelsize=8)
    correct_tick_marks(ax_m)

    if col_o == 0:
        ax_m.set_ylabel(
            r"$r_p\,w_{\mathrm{V}+}\ [\mathrm{Mpc}^2 h^{-2}]$",
            fontsize=9,
        )

    # ── title ─────────────────────────────────────────────────────────────────
    vb  = bd["void_bin"]
    ttl = (
        r"$R_v = $"
        f"{vb[0]}–{vb[1]}"
        r" Mpc/$h$"
        f"    "
        r"$\langle R_v\rangle = $"
        f"{bd['Rv']:.2f}"
        r" Mpc/$h$"
        "\n"
        r"$A_{1,v}^{\rm MAP} = $"
        f"{bd['A1_best']:.3f}"
        r"   $b_v^{\rm MAP} = $"
        f"{bd['bv_best']:.3f}"
    )
    ax_m.set_title(ttl, fontsize=8.5, pad=4, linespacing=1.5)

    # ── residuals ─────────────────────────────────────────────────────────────
    ax_r.errorbar(
        bd["rp_cut"],
        bd["residual"],
        yerr=bd["frac_err"],
        fmt="o", ms=4, capsize=2.5, elinewidth=1.0,
        color=COLOR, zorder=3,
    )
    ax_r.axhline(0, color=COLOR, ls="--", lw=1.2)
    ax_r.set_ylim(-60, 60)
    ax_r.tick_params(axis="both", labelsize=7.5)
    correct_tick_marks(ax_r)

    if col_o == 0:
        ax_r.set_ylabel(r"$\Delta/|D|\ [\%]$", fontsize=8)
    ax_r.set_xlabel(r"$r_p\ [\mathrm{Mpc}\,h^{-1}]$", fontsize=8.5)

    # ── chi2 annotation ───────────────────────────────────────────────────────
    ax_r.text(
        0.97, 0.88,
        rf"$\chi^2_{{\rm red}} = {bd['chi2_red']:.2f}$",
        transform=ax_r.transAxes,
        fontsize=8.5, va="top", ha="right",
        color=COLOR,
    )

plt.tight_layout()

out_path = (PLOTS_DIR
            / f"combined_fits_2x3_{MODE}_{BOX_SIZE}_{RUN}"
              f"_env_dep_analytical_xivmhankel_morerp_rebinned.pdf")
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
logger.info(f"Saved → {out_path}")