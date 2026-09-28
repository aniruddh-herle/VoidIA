import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from mnras import *
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

plt.rcParams['text.usetex'] = False
h = 0.681

# =====================================================
# Settings
# =====================================================

def get_radius_from_name(fname):
    base = os.path.basename(fname)
    return float(base.split("Rv_")[1].split("_")[0])

def stack_stats(arr):
    med = np.median(arr, axis=0)
    p16 = np.percentile(arr, 16, axis=0)
    p84 = np.percentile(arr, 84, axis=0)
    return med, p16, p84

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
    return 4 * np.pi * integrand.sum(axis=1)

# =====================================================
# Configuration
# =====================================================

modes        = ['FIDUCIAL']
mean_density = 1.846582e-01

log10kmin = -5
log10kmax = 2
k_native  = np.logspace(log10kmin, log10kmax, 1000001)
k_plot    = k_native * h

bin_edges  = [(0, 10), (10, 20), (20, 30), (30, 50)]
bin_labels = ['bin0', 'bin1', 'bin2', 'bin3']
bin_titles = ["0–10 Mpc/h", "10–20 Mpc/h", "20–30 Mpc/h", "30–50 Mpc/h"]

rmaxes = [100, 150, 200, 250, 300, 400]

profile_dirs = {
    'lowres': "/disks/shear16/herle/void_profiles/zobov_hydro_particles/",
    'hres':   "/disks/shear16/herle/void_profiles/zobov_hydro_particles_hres/",
}

temp_dir = "/disks/shear16/herle/code/temp_data/"
plot_dir = "/disks/shear16/herle/code/plots/"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# =====================================================
# MAIN LOOP
# =====================================================

# for mode in modes:
#     for res_label, profile_base in profile_dirs.items():

#         profile_dir = os.path.join(profile_base, mode)
#         files = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
#         print(f"\n[{mode} | {res_label}] Found {len(files)} void profiles in {profile_dir}")

#         # --------------------------------------------------
#         # Load and bin profiles
#         # --------------------------------------------------
#         bins   = [[] for _ in range(4)]
#         r_phys = None

#         for f in files:
#             d        = np.load(f)
#             r_phys   = d["r_phys"]
#             rho_phys = d["density_phys"]
#             delta    = rho_phys / mean_density - 1
#             Rv       = get_radius_from_name(f)
#             for b, (lo, hi) in enumerate(bin_edges):
#                 if lo <= Rv < hi:
#                     bins[b].append(delta)

#         bin_meds = []
#         for b in range(4):
#             arr = np.array(bins[b])
#             med, _, _ = stack_stats(arr)
#             bin_meds.append(med)
#             print(f"  {bin_titles[b]}: {len(bins[b])} voids")

#         # --------------------------------------------------
#         # Hankel transform for each rmax, save to temp_data
#         # --------------------------------------------------
#         for rmax in rmaxes:

#             mask   = r_phys <= rmax
#             r_trunc = r_phys[mask]

#             for b, (bin_label, med) in enumerate(zip(bin_labels, bin_meds)):

#                 delta_trunc = med[mask]
#                 delta_k     = hankel_transform(r_trunc, delta_trunc, k_native)
#                 delta_k    /= h**3

#                 fname = (
#                     f"{temp_dir}"
#                     f"zobov_{mode}_{res_label}_void_Rv{bin_label}"
#                     f"_particles_rmax{rmax}.npy"
#                 )
#                 np.save(fname, delta_k)

#             print(f"  Saved rmax={rmax}")

# =====================================================
# PLOTTING — 2x4 grid, all rmax on one figure per (mode, res)
# =====================================================

colors_rmax = plt.cm.viridis(np.linspace(0, 0.9, len(rmaxes)))

for mode in modes:
    for res_label in profile_dirs:

        fig, axes = plt.subplots(
            2, 4, figsize=(22, 8),
            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08, 'wspace': 0.25}
        )

        for b, (bin_label, bin_title) in enumerate(zip(bin_labels, bin_titles)):

            ax_main = axes[0, b]
            ax_rat  = axes[1, b]

            # load all rmax arrays for this bin
            curves = {}
            for rmax in rmaxes:
                fname = (
                    f"{temp_dir}"
                    f"zobov_{mode}_{res_label}_void_Rv{bin_label}"
                    f"_particles_rmax{rmax}.npy"
                )
                curves[rmax] = np.load(fname)

            ref = curves[rmaxes[0]]

            for rmax, col in zip(rmaxes, colors_rmax):
                delta_k = curves[rmax]
                ax_main.plot(k_plot, delta_k / 1e3, color=col, label=f"rmax={rmax}")

                ratio = np.where(np.abs(ref) > 0, delta_k / ref, np.nan)
                ax_rat.plot(k_plot, ratio, color=col)

            # main panel
            ax_main.set_xscale('log')
            ax_main.axhline(0, ls='--', color='k', lw=0.8)
            ax_main.set_xlim(1e-5, 1e2)
            ax_main.set_title(bin_title)
            ax_main.set_xticklabels([])
            if b == 0:
                ax_main.set_ylabel(r"$\Delta_v(k)\times10^3$")
            if b == 3:
                ax_main.legend(frameon=False, fontsize=8, loc='lower right')
            correct_tick_marks(ax_main)

            # ratio panel
            ax_rat.set_xscale('log')
            ax_rat.axhline(1, ls='--', color='k', lw=0.8)
            ax_rat.set_xlim(1e-5, 1e2)
            ax_rat.set_ylim(0.5, 1.5)
            ax_rat.set_xlabel(r"$k\;[\mathrm{Mpc}^{-1}]$")
            if b == 0:
                ax_rat.set_ylabel(f"ratio to rmax={rmaxes[0]}")
            correct_tick_marks(ax_rat)

        fig.suptitle(f"{mode} [{res_label}] — effect of rmax on $\\Delta_v(k)$", fontsize=13)

        plt.savefig(
            f"{plot_dir}"
            f"zobov_{mode}_{res_label}_delta_k_rmax_comparison_allbins.png",
            bbox_inches='tight'
        )
        plt.close()
        print(f"Saved plot: {mode} {res_label} all bins")