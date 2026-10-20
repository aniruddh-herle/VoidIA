# import numpy as np
# from scipy.interpolate import interp1d
# import pyccl as ccl
# import pyccl.ccllib as lib
# import pyccl.nl_pt as pt
# import glob
# import pandas as pd
# from nautilus import Sampler, Prior
# from multiprocessing import Pool

# import argparse

# import os
# os.environ["OMP_NUM_THREADS"] = "1"

# parser = argparse.ArgumentParser()
# parser.add_argument('-box_size', type=float, default=1000, help='Size of the box.')
# parser.add_argument('-n_cores', type=int, default=128, help='Number of CPU cores to use for multiprocessing.')
# args = parser.parse_args()

# from scipy.signal import savgol_filter
# from scipy.interpolate import interp1d

# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# if not logger.hasHandlers():
#     ch = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

# import pyccl as ccl
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(script_dir, ".."))

# if root_dir not in sys.path:
#     sys.path.append(root_dir)

# rootpath = Path('/disks/shear16/herle/code/GNAT')
# if str(rootpath) not in os.sys.path:
#     os.sys.path.append(str(rootpath))

# from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor
# from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum

# import pandas as pd

# # =====================================================
# # Hankel transform
# # =====================================================
# def j0(x):
#     out = np.ones_like(x)
#     mask = x != 0
#     out[mask] = np.sin(x[mask]) / x[mask]
#     return out

# def hankel_transform(r, delta_r, k_vals):
#     dr = np.gradient(r)
#     kr = np.outer(k_vals, r)      # shape (Nk, Nr)
#     kernel = j0(kr)
#     integrand = kernel * (r**2 * dr * delta_r)
#     return 4*np.pi * integrand.sum(axis=1)

# # =====================================================

# cosmo_ccl = ccl.Cosmology(
#     Omega_c=0.306-0.0486,
#     Omega_b=0.0486,
#     h=0.681,
#     sigma8=0.807,
#     n_s=0.967
# )

# redshift_list = np.array([0])

# kmin, kmax = 1e-5, 1e2

# rp_values2v = np.logspace(-1, np.log10(900), 200)

# rp_array = rp_values2v
# r_array  = rp_array

# pi_max = 100.0

# k_array_single = np.geomspace(kmin, kmax, 1000)
# k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
# pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))


# h = 0.681


# k_native = np.logspace(-5, 2, 1000001)   # h/Mpc


# bin_labels       = ['0_10', '10_15', '15_20', '20_25', '25_30', '30_50']
# delta_v_bin_labels = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5']

# void_bins = [(0,10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
# Rv_median =  [8.23, 12.49, 17.28, 22.12, 26.98, 32.56]

# rmax_trunc = 400.0   # Mpc/h — match shell profile truncation

# # Savitzky-Golay window lengths: 30 values from 101 to 3001
# window_lengths = np.linspace(101, 3001, 30).astype(int)

# # Ensure all window lengths are odd and unique
# # window_lengths = np.unique(window_lengths + (window_lengths % 2 == 0))

# # print(f"Number of window lengths: {len(window_lengths)}")
# # print(window_lengths)

# window_length = [ 101, 501, 1001, 1401, 1501, 1601, 1701, 1801, 1901, 2001, 2501, 2801, 2901, 3001]
# # Store results for each smoothing window
# results = {}
# unsmoothed_results = {}


# for bin_idx, (void_bin, bin_label, delta_bin_label) in enumerate(
#     zip(void_bins, bin_labels, delta_v_bin_labels)
# ):
#     Rv = Rv_median[bin_idx]

#     print(f"  -- Void bin: {void_bin}, Rv={Rv}, bin_label={bin_label}")

#     # --------------------------------------------------
#     # Load xi_vm from measurement and Hankel transform it
#     # --------------------------------------------------
#     # run      = 'L1000N1800'
#     run      = 'L2800N5040'
#     redshift = '0.0'

#     df_corr = pd.read_hdf(
#         f'/disks/shear16/herle/correlations1/'
#         f'correlations_joint_revolver_{bin_label}hMpc_{run}'
#         f'_HYDRO_FIDUCIAL_{redshift}_env_dep_analytical_morerp_fullt.h5',
#         key='a'
#     )

#     r_xi  = df_corr['xi_vm_r'].values    # Mpc/h
#     xi_vm = df_corr['xi_vm'].values      # dimensionless

#     # Interpolate xi_vm onto a fine r-grid before transforming
#     r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 1000)
#     xi_fine = interp1d(r_xi, xi_vm, kind='cubic')(r_fine)

#     # Truncate at rmax
#     mask_r   = r_fine <= rmax_trunc
#     r_trunc  = r_fine[mask_r]
#     xi_trunc = xi_fine[mask_r]

#     # Hankel transform
#     delta_v = hankel_transform(r_trunc, xi_trunc, k_native)
#     delta_v /= h**3

#     # Interpolate onto k_array_single
#     logk_void   = np.log10(k_native)
#     logk_target = np.log10(k_array_single)

#     interp_to_target = interp1d(
#         logk_void, delta_v, kind="linear",
#         bounds_error=False, fill_value=0.0
#     )
#     delta_v_on_target = interp_to_target(logk_target)

#     # Store the unsmoothed Delta_v(k)
#     unsmoothed_results[bin_idx] = delta_v_on_target.copy()

#     # Smooth on k_array_single's log grid
#     logk = logk_target

#     N_log        = 20000
#     logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

#     interp_func   = interp1d(logk, delta_v_on_target, kind="linear")
#     delta_uniform = interp_func(logk_uniform)

#     polyorder = 3

#     for window_length in window_lengths:

#         if window_length % 2 == 0:
#             window_length += 1

#         delta_uniform_smooth = savgol_filter(
#             delta_uniform, window_length, polyorder
#         )

#         interp_back = interp1d(
#             logk_uniform, delta_uniform_smooth, kind="linear"
#         )
#         delta_v_smoothed = interp_back(logk)

#         # Store the smoothed result for each void bin and window length
#         results[(bin_idx, window_length)] = delta_v_smoothed.copy()


# # =====================================================
# # Save smoothing results
# # =====================================================
output_file = (
    '/disks/shear16/herle/code/plots/'
    'smoothing_window_results.npz'
)

# delta_smoothed_array = np.array([
#     [
#         results[(bin_idx, wl)]
#         for wl in window_lengths
#     ]
#     for bin_idx in range(len(void_bins))
# ])

# delta_unsmoothed_array = np.array([
#     unsmoothed_results[bin_idx]
#     for bin_idx in range(len(void_bins))
# ])

# np.savez_compressed(
#     output_file,
#     k=k_array_single,
#     window_lengths=window_lengths,
#     void_bins=np.array(void_bins),
#     Rv_median=np.array(Rv_median),
#     delta_smoothed=delta_smoothed_array,
#     delta_unsmoothed=delta_unsmoothed_array
# )

# print(f"Saved results to {output_file}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mnras import *

# =====================================================
# Load saved results
# =====================================================
input_file = (
    '/disks/shear16/herle/code/plots/'
    'smoothing_window_results.npz'
)

data = np.load(input_file)

k = data['k']
window_lengths = data['window_lengths']
void_bins = data['void_bins']
Rv_median = data['Rv_median']

delta_smoothed = data['delta_smoothed']
delta_unsmoothed = data['delta_unsmoothed']

# =====================================================
# Plot settings
# =====================================================
cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(
    vmin=window_lengths.min(),
    vmax=window_lengths.max()
)

fig = plt.figure(figsize=(10, 27))

# Four rows: main panel + ratio panel, repeated twice
gs = fig.add_gridspec(
    6, 2,
    height_ratios=[3, 1, 3, 1, 3, 1],
    hspace=0.12,
    wspace=0.32
)

# =====================================================
# One main panel and one ratio panel per void bin
# =====================================================
# Before the loop
ax_main_ref = None

for bin_idx, void_bin in enumerate(void_bins):

    row = (bin_idx // 2) * 2
    col = bin_idx % 2

    ax = fig.add_subplot(
        gs[row, col],
        sharex=ax_main_ref
    )

    if ax_main_ref is None:
        ax_main_ref = ax

    ax_ratio = fig.add_subplot(
        gs[row + 1, col],
        sharex=ax
    )

    baseline = delta_unsmoothed[bin_idx]

    # Mask ratios where the baseline is too close to zero
    threshold = 1e-3 * np.max(np.abs(baseline))
    valid = np.abs(baseline) > threshold

    for j, wl in enumerate(window_lengths):

        color = cmap(norm(wl))
        smoothed = delta_smoothed[bin_idx, j]

        # Main panel: smoothed Delta_v(k)
        ax.plot(
            k, smoothed,
            color=color,
            linewidth=1.2
        )

        # Lower panel: ratio to unsmoothed Delta_v(k)
        ratio = np.full_like(baseline, np.nan)
        ratio[valid] = smoothed[valid] / baseline[valid]

        ax_ratio.plot(
            k, ratio,
            color=color,
            linewidth=1.0
        )

    # Main panel formatting
    ax.set_xscale('log')
    ax.set_title(
        rf'$R_{{\rm v}} \in '
        rf'[{void_bin[0]}, {void_bin[1]}]'
        rf'\,h^{{-1}}\,\mathrm{{Mpc}}$'
    )
    ax.set_ylabel(r'$\Delta_v(k)$')
    ax.tick_params(labelbottom=False)

    # Ratio panel formatting
    ax_ratio.set_xscale('log')
    ax_ratio.axhline(
        1.0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )
    if bin_idx >= 3:
        ax_ratio.set_xlabel(r'$k\,[h\,\mathrm{Mpc}^{-1}]$')
    else:
        ax_ratio.tick_params(labelbottom=False)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_ylim(0.8, 1.2)

    correct_tick_marks(ax_ratio)
    correct_tick_marks(ax)

# =====================================================
# Shared colour bar for window length
# =====================================================
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=fig.axes,
    location='right',
    pad=0.025,
    fraction=0.025
)
cbar.set_label('Savitzky-Golay window length')


fig.savefig(
    '/disks/shear16/herle/code/plots/'
    'smoothing_window_ratios.png',
    dpi=200,
    bbox_inches='tight'
)
