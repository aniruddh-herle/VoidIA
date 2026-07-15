import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# --------------------------------------------------
# Paths to profile directories
# --------------------------------------------------
base_dir = "/disks/shear16/herle/void_profiles/combined2/"

catalog_dirs = {
    "Disperse": os.path.join(base_dir, "Disperse"),
    "Voxel": os.path.join(base_dir, "Voxel"),
    "Zobov": os.path.join(base_dir, "Zobov"),
    "VERSUS": os.path.join(base_dir, "VERSUS"),
}

# --------------------------------------------------
# Extract radius from filename
# --------------------------------------------------
radius_pattern = re.compile(r"void_Rv_([0-9.]+)_id")

def load_radii(directory):
    files = glob.glob(os.path.join(directory, "*.npz"))
    radii = []

    for f in files:
        match = radius_pattern.search(os.path.basename(f))
        if match:
            radii.append(float(match.group(1)))

    return np.array(radii)

# --------------------------------------------------
# Load all radii
# --------------------------------------------------
all_radii = {}

for name, path in catalog_dirs.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing directory: {path}")
        continue

    radii = load_radii(path)
    all_radii[name] = radii

    print(f"{name}: {len(radii)} voids")

# --------------------------------------------------
# Plot histograms
# --------------------------------------------------
plt.figure(figsize=(7, 6))

bins = np.linspace(0, 80, 40)

colors = {
    "Disperse": "C0",
    "Voxel": "C1",
    "Zobov": "C2",
    "VERSUS": "C3",
}

for name, radii in all_radii.items():
    if len(radii) == 0:
        continue
    

    print(f"{name}: min radius = {np.min(radii):.2f} Mpc, max radius = {np.max(radii):.2f} Mpc")

    plt.hist(
        radii,
        bins=bins,
        histtype="step",
        density=True,
        linewidth=2,
        label=name,
        color=colors.get(name, None),
    )

plt.xlabel("Void Radius [Mpc]")
plt.ylabel("Probability Density")
plt.legend(frameon=False)
plt.tight_layout()

outpath = "/disks/shear16/herle/code/plots/void_radius_histograms_all.png"
plt.savefig(outpath, dpi=300)

print(f"Saved: {outpath}")

import pandas as pd

h=0.681

voxel_voids = np.loadtxt('/disks/shear16/herle/voids/voxel-Flamingo_output_cat.txt') #in Mpc/h
zobov_voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_output_cat.txt') #in Mpc/h

voxel_voids[:, 1:4] /= h
zobov_voids[:, 1:4] /= h


disperse_voids = pd.read_hdf('/disks/shear16/herle/voids/voids_L1000N1800_HYDRO_FIDUCIAL_0.0.h5') # in Mpc
versus_voids = np.loadtxt('/disks/shear16/herle/code/void_catalog_versus_L1000N1800_HYDRO_FIDUCIAL_0.0.txt') # in Mpc/h

versus_voids[:, 0:3] /= h   # positions: Mpc/h → Mpc
versus_voids[:, 3]  /= h    # radius:    Mpc/h → Mpc


plt.figure(figsize=(6,6))

plt.hist(voxel_voids[:, 4], range=(0, 50), bins=30, histtype='step', density=True, label='Revolver (Voxel)')
plt.hist(zobov_voids[:, 4], range=(0, 50), bins=30, histtype='step', density=True, label='Revolver (Zobov)')
plt.hist(disperse_voids['void_radius'], range=(0, 50), bins=30, histtype='step', density=True, label='Disperse')
plt.hist(versus_voids[:, 3], range=(0, 50), bins=30, histtype='step', density=True, label='VERSUS')

plt.legend(frameon=False)

plt.xlabel('Void Radius [Mpc]')
plt.ylabel('Probability Density')
plt.savefig('/disks/shear16/herle/code/plots/Voids_radius_distribution.png')