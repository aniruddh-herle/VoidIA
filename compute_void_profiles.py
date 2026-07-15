import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mnras import *

h=0.681

voxel_voids = np.loadtxt('/disks/shear16/herle/voids/voxel-Flamingo_output_cat.txt') #in Mpc/h
zobov_voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_output_cat.txt') #in Mpc/h

voxel_voids[:, 1:4] /= h
zobov_voids[:, 1:4] /= h


disperse_voids = pd.read_hdf('/disks/shear16/herle/voids/voids_L1000N1800_HYDRO_FIDUCIAL_0.0.h5') # in Mpc
versus_voids = np.loadtxt('/disks/shear16/herle/code/void_catalog_versus_L1000N1800_HYDRO_FIDUCIAL_0.0.txt') # in Mpc/h

versus_voids[:, 0:3] /= h   # positions: Mpc/h → Mpc
versus_voids[:, 3]  /= h    # radius:    Mpc/h → Mpc



# plt.figure(figsize=(6,6))

# plt.hist(voxel_voids[:, 4], range=(0, 50), bins=30, histtype='step', density=True, label='Revolver (Voxel)')
# plt.hist(zobov_voids[:, 4], range=(0, 50), bins=30, histtype='step', density=True, label='Revolver (Zobov)')
# plt.hist(disperse_voids['void_radius'], range=(0, 50), bins=30, histtype='step', density=True, label='Disperse')
# plt.hist(versus_voids[:, 3], range=(0, 50), bins=30, histtype='step', density=True, label='VERSUS')

# plt.legend(frameon=False)


# plt.xlabel('Void Radius [Mpc]')
# plt.ylabel('Probability Density')
# plt.savefig('/disks/shear16/herle/code/plots/voxel_voids_radius_distribution.png')

# only select voids with radius between 10 and 30 Mpc/h
# disperse_voids = disperse_voids[(disperse_voids['void_radius'] >= 10) & (disperse_voids['void_radius'] <= 30)]
# voxel_voids = voxel_voids[(voxel_voids[:, 4] >= 10) & (voxel_voids[:, 4] <= 30)]
# zobov_voids = zobov_voids[(zobov_voids[:, 4] >= 10) & (zobov_voids[:, 4] <= 30)]

# subsample N randomly to 5000 voids each for faster computation
# N = 50
# voxel_voids = voxel_voids[np.random.choice(len(voxel_voids), N, replace=False)]
# zobov_voids = zobov_voids[np.random.choice(len(zobov_voids), N, replace=False)]
# disperse_voids = disperse_voids.sample(N, replace=False)

print(f"Number of Disperse voids: {len(disperse_voids)}")
print(f"Number of Voxel voids: {len(voxel_voids)}")
print(f"Number of Zobov voids: {len(zobov_voids)}")
print(f"Number of VERSUS voids: {len(versus_voids)}")

box_size = 1000

# -------------------------------
# Read galaxy data
# -------------------------------
galaxy_data = pd.read_hdf(
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_FIDUCIAL_0.0stellar_mass_cut.h5'
)

# convert to Mpc
for c in ['halo_centre_x','halo_centre_y','halo_centre_z']:
    galaxy_data[c] /= 1e3

gx = galaxy_data["halo_centre_x"].values
gy = galaxy_data["halo_centre_y"].values
gz = galaxy_data["halo_centre_z"].values

# -------------------------------
# Global density
# -------------------------------
volume = box_size**3
N_total = len(galaxy_data)
universal_mean_density = N_total / volume
print(f"Mean density: {universal_mean_density:.6e}")

# -------------------------------
# Bins
# -------------------------------
r_bins_phys   = np.linspace(0, 200, 80)   # Mpc
r_bins_scaled = np.linspace(0, 5, 60)     # r/Rv

r_cent_phys   = 0.5*(r_bins_phys[:-1] + r_bins_phys[1:])
r_cent_scaled = 0.5*(r_bins_scaled[:-1] + r_bins_scaled[1:])

# -------------------------------
# Function to compute void profiles
# -------------------------------
def compute_void_profiles(void_positions, void_radii, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    profiles_phys = []
    profiles_scaled = []

    for i, (pos, Rv) in enumerate(zip(void_positions, void_radii)):
        xv, yv, zv = pos

        # periodic distances
        dx = np.minimum(np.abs(gx - xv), box_size - np.abs(gx - xv))
        dy = np.minimum(np.abs(gy - yv), box_size - np.abs(gy - yv))
        dz = np.minimum(np.abs(gz - zv), box_size - np.abs(gz - zv))
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Physical profile
        counts_phys, _ = np.histogram(dist, bins=r_bins_phys)
        shell_vol_phys = (4/3)*np.pi*(r_bins_phys[1:]**3 - r_bins_phys[:-1]**3)
        density_phys = counts_phys / shell_vol_phys
        profiles_phys.append(density_phys)

        # Scaled profile
        dist_scaled = dist / Rv
        counts_scaled, _ = np.histogram(dist_scaled, bins=r_bins_scaled)
        shell_vol_scaled = (4/3)*np.pi*(r_bins_scaled[1:]**3 - r_bins_scaled[:-1]**3) * Rv**3
        density_scaled = counts_scaled / shell_vol_scaled
        profiles_scaled.append(density_scaled)

        # Save individual void profile
        fname = f"{save_dir}/void_Rv_{Rv:.2f}_id_{i}.npz"
        np.savez(fname,
                 r_phys=r_cent_phys,
                 density_phys=density_phys,
                 r_scaled=r_cent_scaled,
                 density_scaled=density_scaled,
                 Rv=Rv)
        
    return np.array(profiles_phys), np.array(profiles_scaled)

# -------------------------------
# Prepare catalogs dictionary
# -------------------------------
catalogs = {
    "Disperse": (disperse_voids[['x','y','z']].values, disperse_voids['void_radius'].values),
    "Voxel": (voxel_voids[:, 1:4], voxel_voids[:, 4]),
    "Zobov": (zobov_voids[:, 1:4], zobov_voids[:, 4]),
    "VERSUS": (versus_voids[:, 0:3], versus_voids[:, 3])
}


save_base_dir = "/disks/shear16/herle/void_profiles/combined/"

all_profiles_phys = {}
all_profiles_scaled = {}

# -------------------------------
# Compute profiles for all catalogs
# -------------------------------
for name, (positions, radii) in catalogs.items():
    save_dir = os.path.join(save_base_dir, name)
    phys, scaled = compute_void_profiles(positions, radii, save_dir)
    all_profiles_phys[name] = phys
    all_profiles_scaled[name] = scaled

# for name in catalogs.keys():
#     save_dir = os.path.join(save_base_dir, name)

#     profiles_phys = []
#     profiles_scaled = []

#     # load every saved .npz file for this catalog
#     for fname in sorted(os.listdir(save_dir)):
#         if not fname.endswith(".npz"):
#             continue

#         data = np.load(os.path.join(save_dir, fname))
#         profiles_phys.append(data["density_phys"])
#         profiles_scaled.append(data["density_scaled"])

#         # (optional) recover bin centers from first file
#         if "r_cent_phys" not in globals():
#             r_cent_phys = data["r_phys"]
#             r_cent_scaled = data["r_scaled"]

#     all_profiles_phys[name] = np.array(profiles_phys)
#     all_profiles_scaled[name] = np.array(profiles_scaled)

# -------------------------------
# Plot stacked median + percentiles
# -------------------------------
plt.rcParams["figure.figsize"] = (14,6)
plt.rcParams['text.usetex'] = False
fig, ax = plt.subplots(1,2)

colors = {"Disperse": "C0", "Voxel": "C1", "Zobov": "C2", "VERSUS": "C3"}

# Physical
for name, profiles in all_profiles_phys.items():
    median = np.median(profiles, axis=0)
    p16 = np.percentile(profiles, 16, axis=0)
    p84 = np.percentile(profiles, 84, axis=0)
    ax[0].plot(r_cent_phys, median, label=name, color=colors[name])
    ax[0].fill_between(r_cent_phys, p16, p84, alpha=0.3, color=colors[name])

ax[0].axhline(universal_mean_density, ls="--", color="k", label="Mean density")
ax[0].set_xlabel("r [Mpc]")
ax[0].set_ylabel("Galaxy density [Mpc$^{-3}$]")
ax[0].set_yscale("log")
ax[0].set_title("Physical")

# Scaled
for name, profiles in all_profiles_scaled.items():
    median = np.median(profiles, axis=0)
    p16 = np.percentile(profiles, 16, axis=0)
    p84 = np.percentile(profiles, 84, axis=0)
    ax[1].plot(r_cent_scaled, median, label=name, color=colors[name])
    ax[1].fill_between(r_cent_scaled, p16, p84, alpha=0.3, color=colors[name])

ax[1].axhline(universal_mean_density, ls="--", color="k", label="Mean density")
ax[1].set_xlabel("r / Rv")
ax[1].set_yscale("log")
ax[1].set_title("Scaled")
ax[1].legend(frameon=False)
correct_tick_marks(ax[0])
correct_tick_marks(ax[1])


plt.tight_layout()
plt.savefig("/disks/shear16/herle/code/plots/void_profiles_all_catalogs.png")
