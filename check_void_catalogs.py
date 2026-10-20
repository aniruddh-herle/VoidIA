import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mnras import *

h = 0.681

base_path = '/disks/shear16/herle/voids/'

files = [
    'zobov-Flamingo_output_fulltracerFIDUCIAL_cat.txt',
    'zobov-Flamingo_output_fulltracerWEAK_AGN_cat.txt',
    'zobov-Flamingo_output_fulltracerSTRONG_AGN_cat.txt',
    'zobov-Flamingo_output_fulltracerSTRONGER_AGN_cat.txt',
    'zobov-Flamingo_output_fulltracerSTRONGEST_AGN_cat.txt',
    'zobov-Flamingo_output_fulltracerSTRONG_SUPERNOVA_cat.txt',
    'zobov-Flamingo_output_fulltracerJETS_published_cat.txt',
    'zobov-Flamingo_output_fulltracerSTRONG_JETS_published_cat.txt',
    'zobov-Flamingo_output_fulltracerFIDUCIAL_2gigabox_cat.txt',
]

# ── Color scheme ──────────────────────────────────────────────
# FIDUCIAL anchors : green (L1) / dark purple (L2p8)
# AGN series       : blue gradient, light → dark  =  weak → strong AGN
# Supernova        : burnt orange
# Jets             : teal gradient, light → dark
FLAMINGO_colors = [
    '#117733',  # L1 FIDUCIAL      – dark green
    '#C6DBEF',  # WEAK_AGN         – very pale blue   (fgas +2σ)
    '#6BAED6',  # STRONG_AGN       – light blue       (fgas −2σ)
    '#2171B5',  # STRONGER_AGN     – medium blue      (fgas −4σ)
    '#084594',  # STRONGEST_AGN    – dark navy        (fgas −8σ)
    '#D94801',  # STRONG_SUPERNOVA – burnt orange     (M*−σ)
    '#41B6C4',  # JETS             – light teal
    '#0B5E6B',  # STRONG_JETS      – dark teal        (JETS + fgas −4σ)
    '#332288',  # L2p8 FIDUCIAL    – dark purple
]

FLAMINGO_labels = [r"L1$\_$m9", r"fgas$+2\sigma$", r"fgas$-2\sigma$", r"fgas$-4\sigma$", r"fgas$-8\sigma$",
                   r"M*$-\sigma$", "Jets", r"Jets fgas$-4\sigma$", r"L2p8$\_$m9"]

box_sizes = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2800]  # Mpc

# NN mean inter-tracer distance per simulation (Mpc/h), measured from KD-tree
# Order matches files / labels above
NN_MEAN = 0.9488
R_MIN_FACTOR = 2.5   # void radius cut in units of NN mean

# =============================================================
# Load catalogs + apply per-simulation radius cut
# =============================================================
catalogs_raw = []
for f in files:
    v = np.loadtxt(os.path.join(base_path, f))
    catalogs_raw.append(v)

catalogs = []
print("=" * 65)
print(f"VOID RADIUS CUT:  R_v > {R_MIN_FACTOR} × NN_mean  (from 2p8 box)")
print("=" * 65)
for cat, label in zip(catalogs_raw, FLAMINGO_labels):
    r_min    = R_MIN_FACTOR * NN_MEAN
    mask     = cat[:, 4] > r_min
    cat_cut  = cat[mask]
    catalogs.append(cat_cut)
    print(f"\n  {label}  →  R_min = {R_MIN_FACTOR} × {NN_MEAN:.4f} = {r_min:.3f} Mpc/h")
    print(f"    Before : {len(cat):>9,}")
    print(f"    After  : {len(cat_cut):>9,}  ({100*mask.mean():.1f}% retained,"
          f"  {(~mask).sum():,} removed)")

# =============================================================
# 1. Number of voids + ratio
# =============================================================
n_voids = [len(c) for c in catalogs]
volume_ratio = (2800 / 1000) ** 3
observed_ratio = n_voids[1] / n_voids[0]

print("=" * 55)
print("VOID COUNTS")
print("=" * 55)
for label, n in zip(FLAMINGO_labels, n_voids):
    print(f"  {label}: {n} voids")
print(f"\n  Observed ratio  (L2p8 / L1):  {observed_ratio:.2f}")
print(f"  Expected ratio  (V2p8 / V1):  {volume_ratio:.2f}")
print(f"  Deficit factor:               {volume_ratio / observed_ratio:.2f}x fewer than expected")

# =============================================================
# 2. Min / max void radii
# =============================================================
print("\n" + "=" * 55)
print("VOID RADII")
print("=" * 55)
for label, cat in zip(FLAMINGO_labels, catalogs):
    rv = cat[:, 4]
    print(f"  {label}: min={rv.min():.2f}  max={rv.max():.2f}  "
          f"median={np.median(rv):.2f}  mean={rv.mean():.2f}  Mpc/h")

# =============================================================
# Figure 1: Radius distribution (original, density=True)
# =============================================================
fig, ax = plt.subplots(figsize=(6, 6))
for cat, color, label in zip(catalogs, FLAMINGO_colors, FLAMINGO_labels):
    ax.hist(
        cat[:, 4],
        bins=50, range=(0, 50),
        density=True,
        histtype='step', linewidth=2,
        color=color, label=label
    )

correct_tick_marks(ax)
ax.set_xlabel(r'Void Radius [Mpc/$h$]')
ax.set_ylabel('Probability Density')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    '/disks/shear16/herle/code/plots/voids_radius_pdf_1gpc_2p8gpc.pdf', bbox_inches='tight'
)
plt.close()

# # =============================================================
# # Figure 2: Void positions (x, y, z) — 3 subpanels
# # =============================================================
# coord_labels  = ['x', 'y', 'z']
# latex_coords  = [r'$x$', r'$y$', r'$z$']
# coord_cols   = [1, 2, 3]

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for ax, col, coord, latex in zip(axes, coord_cols, coord_labels, latex_coords):
#     for cat, color, label, L in zip(catalogs, FLAMINGO_colors, FLAMINGO_labels, box_sizes):
#         ax.hist(
#             cat[:, col],
#             bins=50, range=(0, L),
#             density=True,
#             histtype='step', linewidth=2,
#             color=color, label=label
#         )
#     ax.set_xlabel(fr'{latex} [Mpc/$h$]')
#     ax.set_ylabel('Probability Density')
#     ax.legend(frameon=False, fontsize=9)

# plt.suptitle('Void centre positions', y=1.01)
# plt.tight_layout()
# plt.savefig(
#     '/disks/shear16/herle/code/plots/voids_positions_1gpc_2p8gpc.png',
#     dpi=300, bbox_inches='tight'
# )
# plt.close()

# # =============================================================
# # Figure 2b: Void positions scatter — x vs y plane
# # =============================================================
# fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# for ax, cat, color, label, L in zip(axes, catalogs, FLAMINGO_colors, FLAMINGO_labels, box_sizes):
#     # subsample for plotting speed if needed
#     Nplot = min(10000, len(cat))
#     idx = np.random.choice(len(cat), size=Nplot, replace=False)

#     ax.scatter(
#         cat[idx, 1],  # x
#         cat[idx, 2],  # y
#         s=1,
#         alpha=0.3,
#         color=color,
#         rasterized=True
#     )
#     ax.set_xlim(0, L*h)
#     ax.set_ylim(0, L*h)
#     ax.set_xlabel(r'$x$ [Mpc/$h$]')
#     ax.set_ylabel(r'$y$ [Mpc/$h$]')
#     ax.set_title(label)
#     ax.set_aspect('equal')

# plt.suptitle('Void centre positions — $x$–$y$ plane', y=1.01)
# plt.tight_layout()
# plt.savefig(
#     '/disks/shear16/herle/code/plots/voids_positions_scatter_xy_1gpc_2p8gpc.png',
#     dpi=300, bbox_inches='tight'
# )
# plt.close()

# =============================================================
# Figure 3: Void size function  n(Rv)  [counts per bin]
# =============================================================
rv_bins = np.linspace(0, 50, 51)
rv_centres = 0.5 * (rv_bins[:-1] + rv_bins[1:])

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# --- left: raw counts ---
ax = axes[0]
for cat, color, label in zip(catalogs, FLAMINGO_colors, FLAMINGO_labels):
    counts, _ = np.histogram(cat[:, 4], bins=rv_bins)
    ax.step(rv_centres, counts, where='mid', linewidth=2, color=color, label=label)
ax.set_xlabel(r'$R_v$ [Mpc/$h$]')
ax.set_ylabel(r'$N_{\rm voids}$')
ax.set_yscale('log')
correct_tick_marks(ax)
ax.legend(frameon=False, loc='upper right')
#ax.set_title('Raw counts')

# --- right: volume-normalised (void number density per bin) ---
ax = axes[1]
for cat, color, label, L in zip(catalogs, FLAMINGO_colors, FLAMINGO_labels, box_sizes):
    counts, _ = np.histogram(cat[:, 4], bins=rv_bins)
    dRv = np.diff(rv_bins)
    V = L ** 3
    dn_dRv = counts / (V * dRv)   # [h/Mpc]^3 / [Mpc/h] = [h/Mpc]^4
    ax.step(rv_centres, dn_dRv, where='mid', linewidth=2, color=color, label=label)
ax.set_xlabel(r'$R_v$ [Mpc/$h$]')
ax.set_ylabel(r'${\rm d}n/{\rm d}R_v$ [$h^4\,{\rm Mpc}^{-4}$]')
ax.set_yscale('log')
correct_tick_marks(ax)

#ax.set_title('Volume-normalised (void size function)')

plt.tight_layout()
plt.savefig(
    '/disks/shear16/herle/code/plots/voids_size_function_1gpc_2p8gpc.pdf', bbox_inches='tight'
)
plt.close()

print("\nAll plots saved.")


# voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_outputFIDUCIAL_2gigabox_cat.txt')
# print(voids[:10, 4])   # Rv of first 10 catalog entries
# print(voids[-10:, 4])  # Rv of last 10 catalog entries