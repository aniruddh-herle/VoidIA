import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mnras import *

h=0.681

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

base_path = '/disks/shear16/herle/voids/'

files = [
    'zobov-Flamingo_outputFIDUCIAL_cat.txt',
    'zobov-Flamingo_outputWEAK_AGN_cat.txt',
    'zobov-Flamingo_outputSTRONG_AGN_cat.txt',
    'zobov-Flamingo_outputSTRONGER_AGN_cat.txt',
    'zobov-Flamingo_outputSTRONGEST_AGN_cat.txt',
    'zobov-Flamingo_outputSTRONG_SUPERNOVA_cat.txt',
    'zobov-Flamingo_outputJETS_published_cat.txt',
    'zobov-Flamingo_outputSTRONG_JETS_published_cat.txt',
    'zobov-Flamingo_outputFIDUCIAL_2gigabox_cat.txt'
]

FLAMINGO_colors = [
    '#117733', '#ABD0E6', '#6AAED6', '#3787C0',
    '#105BA4', '#FF8C40', '#7EFF4B', '#55E18E', 
    '#332288'
]

FLAMINGO_labels = [
    r"L1$_m9$",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    r"M_*-\sigma$",
    "JETS",
    r"JETS fgas$-4\sigma$",
    r"L2p8$_m9$",
]

rcParams['text.usetex'] = False

plt.figure(figsize=(6, 6))

for filename, color, label in zip(files, FLAMINGO_colors, FLAMINGO_labels):

    voids = np.loadtxt(os.path.join(base_path, filename))

    plt.hist(
        voids[:, 4],                # radius column
        bins=30,
        range=(0, 50),
        density=True,
        histtype='step',
        linewidth=2,
        color=color,
        label=label
    )

plt.xlabel(r'Void Radius [Mpc/$h$]')
plt.ylabel('Probability Density')
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()

plt.savefig(
    '/disks/shear16/herle/code/plots/voxel_voids_radius_distribution_all_hydro.png',
    dpi=300,
    bbox_inches='tight'
)


box_size = 1000 * h
# box_size = 2800 * h

# # -------------------------------
# # Hydro runs
# # -------------------------------
hydro_modes = {
    "FIDUCIAL": (
        "zobov-Flamingo_outputFIDUCIAL_cat.txt",
        "L1000N1800_HYDRO_FIDUCIAL_0.0stellar_mass_cut.h5",
        r"L1$_{\rm m9}$",
        "#117733"
    ),

    "WEAK_AGN": (
        "zobov-Flamingo_outputWEAK_AGN_cat.txt",
        "L1000N1800_HYDRO_WEAK_AGN_0.0stellar_mass_cut.h5",
        r"fgas$+2\sigma$",
        "#ABD0E6"
    ),

    "STRONG_AGN": (
        "zobov-Flamingo_outputSTRONG_AGN_cat.txt",
        "L1000N1800_HYDRO_STRONG_AGN_0.0stellar_mass_cut.h5",
        r"fgas$-2\sigma$",
        "#6AAED6"
    ),

    "STRONGER_AGN": (
        "zobov-Flamingo_outputSTRONGER_AGN_cat.txt",
        "L1000N1800_HYDRO_STRONGER_AGN_0.0stellar_mass_cut.h5",
        r"fgas$-4\sigma$",
        "#3787C0"
    ),

    "STRONGEST_AGN": (
        "zobov-Flamingo_outputSTRONGEST_AGN_cat.txt",
        "L1000N1800_HYDRO_STRONGEST_AGN_0.0stellar_mass_cut.h5",
        r"fgas$-8\sigma$",
        "#105BA4"
    ),

    "STRONG_SUPERNOVA": (
        "zobov-Flamingo_outputSTRONG_SUPERNOVA_cat.txt",
        "L1000N1800_HYDRO_STRONG_SUPERNOVA_0.0stellar_mass_cut.h5",
        r"$M_*-\sigma$",
        "#FF8C40"
    ),

    "JETS": (
        "zobov-Flamingo_outputJETS_published_cat.txt",
        "L1000N1800_HYDRO_JETS_published_0.0stellar_mass_cut.h5",
        "JETS",
        "#7EFF4B"
    ),

    "STRONG_JETS": (
        "zobov-Flamingo_outputSTRONG_JETS_published_cat.txt",
        "L1000N1800_HYDRO_STRONG_JETS_published_0.0stellar_mass_cut.h5",
        r"JETS fgas$-4\sigma$",
        "#55E18E"
    )
}

# -------------------------------
# Hydro runs
# -------------------------------
# hydro_modes = {
#     "FIDUCIAL_2p8": (
#         "zobov-Flamingo_outputFIDUCIAL_2gigabox_cat.txt",
#         "L2800N5040_HYDRO_FIDUCIAL_0.0stellar_mass_cut.h5",
#         r"L2p8$_{\rm m9}$",
#         "#332288"
#     )

# }


# -------------------------------
# Bins
# -------------------------------
r_bins_phys   = np.linspace(0, 200, 80)
r_bins_scaled = np.linspace(0, 5, 60)

r_cent_phys   = 0.5*(r_bins_phys[:-1] + r_bins_phys[1:])
r_cent_scaled = 0.5*(r_bins_scaled[:-1] + r_bins_scaled[1:])

# -------------------------------
# Worker function for a single void (extracted from the loop body, unchanged logic)
# -------------------------------
def _process_single_void(i, pos, Rv, gx, gy, gz, box_size, r_bins_phys, r_bins_scaled, r_cent_phys, r_cent_scaled, save_dir):

    xv, yv, zv = pos

    dx = np.minimum(np.abs(gx - xv), box_size - np.abs(gx - xv))
    dy = np.minimum(np.abs(gy - yv), box_size - np.abs(gy - yv))
    dz = np.minimum(np.abs(gz - zv), box_size - np.abs(gz - zv))

    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    counts_phys, _ = np.histogram(dist, bins=r_bins_phys)
    shell_vol_phys = (4/3)*np.pi*(r_bins_phys[1:]**3 - r_bins_phys[:-1]**3)
    density_phys = counts_phys / shell_vol_phys

    dist_scaled = dist / Rv
    counts_scaled, _ = np.histogram(dist_scaled, bins=r_bins_scaled)
    shell_vol_scaled = (
        (4/3)*np.pi*(r_bins_scaled[1:]**3 - r_bins_scaled[:-1]**3)
        * Rv**3
    )

    density_scaled = counts_scaled / shell_vol_scaled

    fname = f"{save_dir}/void_Rv_{Rv:.2f}_id_{i}.npz"

    np.savez(
        fname,
        r_phys=r_cent_phys,
        density_phys=density_phys,
        r_scaled=r_cent_scaled,
        density_scaled=density_scaled,
        Rv=Rv
    )

    return density_phys, density_scaled


# -------------------------------
# Function to compute void profiles (now parallelised over voids with joblib)
# -------------------------------
def compute_void_profiles(void_positions, void_radii, save_dir, n_jobs=-1):

    os.makedirs(save_dir, exist_ok=True)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_void)(
            i,
            void_positions[i],
            void_radii[i],
            gx, gy, gz,
            box_size,
            r_bins_phys, r_bins_scaled,
            r_cent_phys, r_cent_scaled,
            save_dir
        )
        for i in tqdm(range(len(void_radii)), desc="Computing void profiles")
    )

    profiles_phys = [r[0] for r in results]
    profiles_scaled = [r[1] for r in results]

    return np.array(profiles_phys), np.array(profiles_scaled)


save_base_dir = "/disks/shear16/herle/void_profiles/zobov_hydro4/"

all_profiles_phys = {}
all_profiles_scaled = {}
colors = {}
mean_densities = {}
labels = {}

# -------------------------------
# Compute profiles for all hydro runs

for mode, (void_file, galaxy_file, label, color) in hydro_modes.items():

    print(f"Processing {label}")

    # -------------------------------
    # Read void catalogue
    # -------------------------------
    zobov_voids = np.loadtxt(
        f"/disks/shear16/herle/voids/{void_file}"
    )

    # zobov_voids = zobov_voids[
    #     (zobov_voids[:,4] >= 10) &
    #     (zobov_voids[:,4] <= 12)
    # ]


    # -------------------------------
    # Read galaxy catalogue
    # -------------------------------
    galaxy_data = pd.read_hdf(
        "/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/"
        + galaxy_file
    )

    for c in ['halo_centre_x','halo_centre_y','halo_centre_z']:
        galaxy_data[c] /= 1e3
        galaxy_data[c] *= h


    gx = galaxy_data["halo_centre_x"].values
    gy = galaxy_data["halo_centre_y"].values
    gz = galaxy_data["halo_centre_z"].values

    # print min and max of galaxy positions
    print(f"Galaxy positions: x = [{gx.min():.2f}, {gx.max():.2f}] Mpc/h")
    print(f"Galaxy positions: y = [{gy.min():.2f}, {gy.max():.2f}] Mpc/h")
    print(f"Galaxy positions: z = [{gz.min():.2f}, {gz.max():.2f}] Mpc/h")
    volume = box_size**3
    mean_densities[label] = len(galaxy_data) / volume

    print(f"Mean density for {label}: {mean_densities[label]:.6e} Mpc^-3/h^3")

    save_dir = os.path.join(save_base_dir, mode)

    phys, scaled = compute_void_profiles(
        zobov_voids[:,1:4],
        zobov_voids[:,4],
        save_dir
    )

    all_profiles_phys[mode] = phys
    all_profiles_scaled[mode] = scaled
    colors[mode] = color
    labels[mode] = label

# -------------------------------
# Plot stacked median + percentiles
# -------------------------------
plt.rcParams["figure.figsize"] = (14,6)
plt.rcParams['text.usetex'] = False

fig, ax = plt.subplots(1,2)

# Physical
for name, profiles in all_profiles_phys.items():

    median = np.median(profiles, axis=0)
    p16 = np.percentile(profiles, 16, axis=0)
    p84 = np.percentile(profiles, 84, axis=0)

    ax[0].plot(
        r_cent_phys,
        median,
        label=name,
        color=colors[name]
    )

    ax[0].fill_between(
        r_cent_phys,
        p16,
        p84,
        alpha=0.3,
        color=colors[name]
    )

# Scaled
for name, profiles in all_profiles_scaled.items():

    median = np.median(profiles, axis=0)
    p16 = np.percentile(profiles, 16, axis=0)
    p84 = np.percentile(profiles, 84, axis=0)

    ax[1].plot(
        r_cent_scaled,
        median,
        label=labels[name],
        color=colors[name]
    )

    ax[1].fill_between(
        r_cent_scaled,
        p16,
        p84,
        alpha=0.3,
        color=colors[name]
    )

fid_density = mean_densities[r"L1$_{\rm m9}$"]

ax[0].axhline(fid_density, ls="--", color="k", label="Mean density")
ax[1].axhline(fid_density, ls="--", color="k")

ax[0].set_xlabel("r [Mpc/h]")
ax[0].set_ylabel("Galaxy density [Mpc$^{-3}/h^3$]")
ax[0].set_yscale("log")
ax[0].set_title("Physical")

ax[1].set_xlabel("r / Rv")
ax[1].set_yscale("log")
ax[1].set_title("Scaled")
ax[1].legend(frameon=False)

correct_tick_marks(ax[0])
correct_tick_marks(ax[1])

plt.tight_layout()

plt.savefig(
    "/disks/shear16/herle/code/plots/zobov_void_profiles_2p8.png"
)
