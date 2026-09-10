import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from mnras import *
plt.rcParams['text.usetex'] = False
h = 0.681
# =====================================================
# Settings
# =====================================================
profile_dir = "/disks/shear16/herle/void_profiles/zobov_hydro/FIDUCIAL/"
mean_density = 1.383265e-02 # Mpc^-3/h^3

print("Mean density (Mpc^-3/h^3): ", mean_density)

# mean_density = 4.3685e-3 # Mpc^-3

# print("Mean density (Mpc^-3): ", mean_density)

# =====================================================
# Find all saved profiles
# =====================================================
files = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
print(f"Found {len(files)} void profiles")


# =====================================================
# Storage
# =====================================================
all_profiles = []


bin_0  = []   # 10–20 Mpc/h

r_phys = None


# =====================================================
# Helper to extract Rv from filename
# filename looks like: void_Rv_22.50_id_17.npz
# =====================================================
def get_radius_from_name(fname):
    base = os.path.basename(fname)
    return float(base.split("Rv_")[1].split("_")[0])


# =====================================================
# Load each void and compute delta
# =====================================================
for f in files:

    d = np.load(f)

    r_phys = d["r_phys"]
    rho_phys = d["density_phys"]

    delta = rho_phys / mean_density - 1

    Rv = get_radius_from_name(f)

    all_profiles.append(delta)


    if 10 <= Rv < 20:
        bin_0.append(delta)

    # if 30 <= Rv < 40:
    #     bin_1.append(delta)
    
    # if 40 <= Rv < 100:
    #     bin_2.append(delta)


# convert to arrays
all_profiles = np.array(all_profiles)
bin_0 = np.array(bin_0)
# bin_1   = np.array(bin_1)
# bin_2   = np.array(bin_2)


# =====================================================
# Stack statistics
# =====================================================
def stack_stats(arr):
    med = np.median(arr, axis=0)
    p16 = np.percentile(arr, 16, axis=0)
    p84 = np.percentile(arr, 84, axis=0)
    return med, p16, p84

# print(len(bin_0), len(bin_1), len(bin_2))
# print(bin_0.shape, bin_1.shape, bin_2.shape)
print(len(bin_0))
print(bin_0.shape)

med_all, p16_all, p84_all = stack_stats(all_profiles)

med_0, _, _ = stack_stats(bin_0)
# med_1, _, _ = stack_stats(bin_1)
# med_2, _, _ = stack_stats(bin_2)

print("Number of voids in each bin:")
print(f"Rv 10 < Rv < 20:  {len(bin_0)}")
# print(f"Rv 20 < Rv < 40: {len(bin_1)}")
# print(f"Rv 40 < Rv < 100: {len(bin_2)}")


# =====================================================
# Plot
# =====================================================
plt.rcParams["figure.figsize"] = (10, 5)

fig, ax = plt.subplots(1, 2)


# =====================================================
# Left: all voids (with errors)
# =====================================================
ax[0].plot(r_phys, med_all, label="Median")
ax[0].fill_between(r_phys, p16_all, p84_all, alpha=0.3)

ax[0].axhline(0, ls="--")

ax[0].set_xlabel("r [Mpc/h]")
ax[0].set_ylabel(r"$\Delta_v(r)$")
ax[0].set_title("All voids")
ax[0].legend(frameon=False)
correct_tick_marks(ax[0])


# =====================================================
# Right: radius bins (medians only)
# =====================================================
ax[1].plot(r_phys, med_0, label="10–20 Mpc/h")
# ax[1].plot(r_phys, med_1, label="20–40 Mpc")
# ax[1].plot(r_phys, med_2, label="40–100 Mpc")

ax[1].axhline(0, ls="--")

ax[1].set_xlabel("r [Mpc/h]")
ax[1].set_title("By void radius")
ax[1].legend(frameon=False)
correct_tick_marks(ax[1])


plt.tight_layout()
plt.savefig("/disks/shear16/herle/code/plots/void_delta_profiles_zobov_fiducial.png")

# =====================================================
# Fourier transform of profiles (all bins, one plot)
# =====================================================

# truncate integration range
rmax = 100
mask = r_phys <= rmax

r_trunc = r_phys[mask]

delta_0_trunc = med_0[mask]
# delta_1_trunc = med_1[mask]
# delta_2_trunc = med_2[mask]


# spherical Bessel j0
def j0(x):
    out = np.ones_like(x)
    mask = x != 0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


# Hankel transform (vectorized)
def hankel_transform(r, delta_r, k_vals):
    dr = np.gradient(r)
    kr = np.outer(k_vals, r)      # shape (Nk, Nr)
    kernel = j0(kr)
    integrand = kernel * (r**2 * dr * delta_r)
    return 4*np.pi * integrand.sum(axis=1)



log10kmin = -5
log10kmax = 2
k_native = np.logspace(log10kmin, log10kmax, 1000001)   # h/Mpc

delta_k_0 = hankel_transform(r_trunc, delta_0_trunc, k_native)
delta_k_0 /= h**3        # amplitude: (Mpc/h)^3 -> Mpc^3

k = k_native * h

# delta_k_1 = hankel_transform(r_trunc, delta_1_trunc, k)
# delta_k_2 = hankel_transform(r_trunc, delta_2_trunc, k)

np.save("/disks/shear16/herle/code/temp_data/trial_zobov_fiducial_void_Rvbin01.npy", delta_k_0)
# np.save("/disks/shear16/herle/code/zobov_fiducial_void_Rvbin1.npy", delta_k_1)
# np.save("/disks/shear16/herle/code/zobov_fiducial_void_Rvbin2.npy", delta_k_2)

# plot all together
plt.figure(figsize=(5,5))
plt.plot(k, delta_k_0/1e3, label="10–20 Mpc/h")
# plt.plot(k, delta_k_1/1e3, label="20–40 Mpc")  
# plt.plot(k, delta_k_2/1e3, label="40–100 Mpc")


plt.xscale("log")
plt.axhline(0, ls="--")

plt.xlabel(r"$k \; [\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$\Delta_v(k) \ \times 10^3$")
plt.legend(frameon=False)

ax = plt.gca()
correct_tick_marks(ax)

plt.xlim(10**(-5), 1000)


plt.tight_layout()
plt.savefig("/disks/shear16/herle/code/plots/zobov_fiducial_void_delta_k_profiles.png")

# =====================================================
# Savitzky–Golay smoothing (TUNE THESE)
# =====================================================
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


# ---------------------------------
# work in log10(k) space
# ---------------------------------
logk = np.log10(k)


# build uniform grid in log-space
N_log = 20000   # resolution of smoothing grid (10–50k is good)

logk_uniform = np.linspace(logk.min(), logk.max(), N_log)


def smooth(delta_v, window_length=1001, polyorder=3):


    # interpolate raw signal onto uniform log grid
    interp_func = interp1d(logk, delta_v, kind="linear")

    delta_uniform = interp_func(logk_uniform)


    # ---------------------------------
    # Savitzky–Golay parameters (TUNE)
    # ---------------------------------
    if window_length % 2 == 0:
        window_length += 1


    delta_uniform_smooth = savgol_filter(
        delta_uniform,
        window_length,
        polyorder
    )


    # ---------------------------------
    # interpolate back to original k
    # ---------------------------------
    interp_back = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")

    return interp_back(logk)

smooth_delta_0 = smooth(delta_k_0)
# smooth_delta_1 = smooth(delta_k_1)
# smooth_delta_2 = smooth(delta_k_2)

# -----------------------------
# save arrays
# -----------------------------
np.save("/disks/shear16/herle/code/temp_data/trial_zobov_fiducial_void_delta_k_bin0_raw.npy", delta_k_0)
np.save("/disks/shear16/herle/code/temp_data/trial_zobov_fiducial_void_delta_k_bin0_smooth.npy", smooth_delta_0)
# np.save("/disks/shear16/herle/code/zobov_fiducial_void_delta_k_bin1_raw.npy", delta_k_1)
# np.save("/disks/shear16/herle/code/zobov_fiducial_void_delta_k_bin1_smooth.npy", smooth_delta_1)
# np.save("/disks/shear16/herle/code/zobov_fiducial_void_delta_k_bin2_raw.npy", delta_k_2)
# np.save("/disks/shear16/herle/code/zobov_fiducial_void_delta_k_bin2_smooth.npy", smooth_delta_2)


# =====================================================
# Plot comparison
# =====================================================
plt.figure(figsize=(5,5))

# plt all bins

plt.plot(k, smooth_delta_0/1e3, label="10–20 Mpc/h (smoothed)", color="C0")
plt.plot(k, delta_k_0/1e3, ls="--", label="10–20 Mpc/h (raw)", color="C0", alpha=0.5)
# plt.plot(k, smooth_delta_1/1e3, label="20–40 Mpc (smoothed)", color="C1")
# plt.plot(k, smooth_delta_2/1e3, label="40–100 Mpc (smoothed)", color="C2")   
# plt.plot(k, delta_k_1/1e3, ls="--", label="30–40 Mpc (raw)", color="C1", alpha=0.5)
# plt.plot(k, delta_k_2/1e3, ls="--", label="40–100 Mpc (raw)", color="C2", alpha=0.5)

plt.xscale("log")
plt.axhline(0, ls="--")

plt.xlabel(r"$k \; [\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$\Delta_v(k) \times 10^3$")

ax = plt.gca()
correct_tick_marks(ax)



plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("/disks/shear16/herle/code/plots/zobov_fiducial_void_delta_bins_smoothed.png")
