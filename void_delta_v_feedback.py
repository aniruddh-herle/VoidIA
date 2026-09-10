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

def get_radius_from_name(fname):
    base = os.path.basename(fname)
    return float(base.split("Rv_")[1].split("_")[0])

def stack_stats(arr):
    med = np.median(arr, axis=0)
    p16 = np.percentile(arr, 16, axis=0)
    p84 = np.percentile(arr, 84, axis=0)
    return med, p16, p84

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

# =====================================================
# Savitzky–Golay smoothing (TUNE THESE)
# =====================================================
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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

    interp_back = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")

    return interp_back(logk)

# modes = ['FIDUCIAL', 'WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA']
# mean_densities = [1.383265e-02, 1.383626e-02, 1.420568e-02, 1.438745e-02, 1.495675e-02, 1.225041e-02]

modes = ['FIDUCIAL']
mean_densities = [1.383265e-02]


# modes = ['FIDUCIAL_2p8']
# mean_densities = [2.014269e-02]


for i in range(len(modes)):

    print(modes[i])

    profile_dir = f"/disks/shear16/herle/void_profiles/zobov_hydro4/{modes[i]}/"
    mean_density = mean_densities[i]

    print("Mean density (Mpc^-3/h^3): ", mean_density)


    files = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
    print(f"Found {len(files)} void profiles")

    print("Min Rv:", min(get_radius_from_name(f) for f in files))
    print("Max Rv:", max(get_radius_from_name(f) for f in files))


    all_profiles = []


    bin_0  = []   
    bin_1  = []   
    bin_2  = []   
    bin_3  = []   

    # NEW: track radii per bin
    Rv_bin_0 = []
    Rv_bin_1 = []
    Rv_bin_2 = []
    Rv_bin_3 = []

    r_phys = None

    for f in files:

        d = np.load(f)

        r_phys = d["r_phys"]
        rho_phys = d["density_phys"]

        delta = rho_phys / mean_density - 1

        Rv = get_radius_from_name(f)

        all_profiles.append(delta)


        if 0 <= Rv < 10:
            bin_0.append(delta)
            Rv_bin_0.append(Rv)

        if 10 <= Rv < 20:
            bin_1.append(delta)
            Rv_bin_1.append(Rv)
        
        if 20 <= Rv < 30:
            bin_2.append(delta)
            Rv_bin_2.append(Rv)

        if 30 <= Rv < 50:
            bin_3.append(delta)
            Rv_bin_3.append(Rv)


    # convert to arrays
    all_profiles = np.array(all_profiles)
    bin_0 = np.array(bin_0)
    bin_1 = np.array(bin_1)
    bin_2 = np.array(bin_2)
    bin_3 = np.array(bin_3)


    med_all, p16_all, p84_all = stack_stats(all_profiles)

    med_0, _, _ = stack_stats(bin_0)
    med_1, _, _ = stack_stats(bin_1)
    med_2, _, _ = stack_stats(bin_2)
    med_3, _, _ = stack_stats(bin_3)

    print("Number of voids in each bin:")
    print(f"Rv 0 < Rv < 10:  {len(bin_0)}")
    print(f"Rv 10 < Rv < 20: {len(bin_1)}")
    print(f"Rv 20 < Rv < 30: {len(bin_2)}")
    print(f"Rv 30 < Rv < 50: {len(bin_3)}")

    med_all, p16_all, p84_all = stack_stats(all_profiles)

    med_0, _, _ = stack_stats(bin_0)
    med_1, _, _ = stack_stats(bin_1)
    med_2, _, _ = stack_stats(bin_2)
    med_3, _, _ = stack_stats(bin_3)

    # NEW: median void radius per bin
    Rv_med_0 = np.median(Rv_bin_0)
    Rv_med_1 = np.median(Rv_bin_1)
    Rv_med_2 = np.median(Rv_bin_2)
    Rv_med_3 = np.median(Rv_bin_3)

    print("Median Rv in each bin:")
    print(f"Rv 0 < Rv < 10:  {Rv_med_0:.2f}")
    print(f"Rv 10 < Rv < 20: {Rv_med_1:.2f}")
    print(f"Rv 20 < Rv < 30: {Rv_med_2:.2f}")
    print(f"Rv 30 < Rv < 50: {Rv_med_3:.2f}")


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
    ax[1].plot(r_phys, med_0, label="0–10 Mpc/h")
    ax[1].plot(r_phys, med_1, label="10–20 Mpc")
    ax[1].plot(r_phys, med_2, label="20–30 Mpc")
    ax[1].plot(r_phys, med_3, label="30–50 Mpc")

    ax[1].axhline(0, ls="--")

    ax[1].set_xlabel("r [Mpc/h]")
    ax[1].set_title("By void radius")
    ax[1].legend(frameon=False)
    correct_tick_marks(ax[1])


    plt.tight_layout()
    plt.savefig(f"/disks/shear16/herle/code/plots/void_delta_profiles_zobov_{modes[i]}.png")

    # =====================================================
    # Fourier transform of profiles (all bins, one plot)
    # =====================================================

    # truncate integration range
    rmax = 100
    mask = r_phys <= rmax

    r_trunc = r_phys[mask]

    delta_0_trunc = med_0[mask]
    delta_1_trunc = med_1[mask]
    delta_2_trunc = med_2[mask]
    delta_3_trunc = med_3[mask]


    delta_k_0 = hankel_transform(r_trunc, delta_0_trunc, k_native)
    delta_k_0 /= h**3        # amplitude: (Mpc/h)^3 -> Mpc^3

    delta_k_1 = hankel_transform(r_trunc, delta_1_trunc, k_native)
    delta_k_1 /= h**3        # amplitude: (Mpc/h)^3 -> Mpc^3

    delta_k_2 = hankel_transform(r_trunc, delta_2_trunc, k_native)
    delta_k_2 /= h**3        # amplitude: (Mpc/h)^3 -> Mpc^3

    delta_k_3 = hankel_transform(r_trunc, delta_3_trunc, k_native)
    delta_k_3 /= h**3        # amplitude: (Mpc/h)^3 -> Mpc^3

    k = k_native * h
    #k = k_native

    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_Rvbin0.npy", delta_k_0)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_Rvbin1.npy", delta_k_1)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_Rvbin2.npy", delta_k_2)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_Rvbin3.npy", delta_k_3)

    # plot all together
    plt.figure(figsize=(5,5))
    plt.plot(k, delta_k_0/1e3, label="0–10 Mpc/h")
    plt.plot(k, delta_k_1/1e3, label="10–20 Mpc")  
    plt.plot(k, delta_k_2/1e3, label="20–30 Mpc")
    plt.plot(k, delta_k_3/1e3, label="30–50 Mpc")


    plt.xscale("log")
    plt.axhline(0, ls="--")

    plt.xlabel(r"$k \; [\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$\Delta_v(k) \ \times 10^3$")
    plt.legend(frameon=False)

    ax = plt.gca()
    correct_tick_marks(ax)

    plt.xlim(10**(-5), 1000)


    plt.tight_layout()
    plt.savefig(f"/disks/shear16/herle/code/plots/zobov_{modes[i]}_void_delta_k_profiles.png")

    


    # ---------------------------------
    # work in log10(k) space
    # ---------------------------------
    logk = np.log10(k)


    # build uniform grid in log-space
    N_log = 20000   # resolution of smoothing grid (10–50k is good)

    logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

    smooth_delta_0 = smooth(delta_k_0)
    smooth_delta_1 = smooth(delta_k_1)
    smooth_delta_2 = smooth(delta_k_2)
    smooth_delta_3 = smooth(delta_k_3)

    # -----------------------------
    # save arrays
    # -----------------------------
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin0_raw.npy", delta_k_0)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin0_smooth.npy", smooth_delta_0)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin1_raw.npy", delta_k_1)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin1_smooth.npy", smooth_delta_1)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin2_raw.npy", delta_k_2)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin2_smooth.npy", smooth_delta_2)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin3_raw.npy", delta_k_3)
    np.save(f"/disks/shear16/herle/code/temp_data/zobov_{modes[i]}_void_delta_k_bin3_smooth.npy", smooth_delta_3)


    # =====================================================
    # Plot comparison
    # =====================================================
    plt.figure(figsize=(5,5))

    # plt all bins

    plt.plot(k, smooth_delta_0/1e3, label="0–10 Mpc/h (smoothed)", color="C0")
    plt.plot(k, delta_k_0/1e3, ls="--", label="0–10 Mpc/h (raw)", color="C0", alpha=0.5)
    plt.plot(k, smooth_delta_1/1e3, label="10–20 Mpc/h (smoothed)", color="C0")
    plt.plot(k, delta_k_1/1e3, ls="--", label="10–20 Mpc/h (raw)", color="C0", alpha=0.5)
    plt.plot(k, smooth_delta_2/1e3, label="20–30 Mpc/h (smoothed)", color="C0")
    plt.plot(k, delta_k_2/1e3, ls="--", label="20–30 Mpc/h (raw)", color="C0", alpha=0.5)
    plt.plot(k, smooth_delta_3/1e3, label="30–50 Mpc/h (smoothed)", color="C0")
    plt.plot(k, delta_k_3/1e3, ls="--", label="30–50 Mpc/h (raw)", color="C0", alpha=0.5)


    plt.xscale("log")
    plt.axhline(0, ls="--")

    plt.xlabel(r"$k \; [\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$\Delta_v(k) \times 10^3$")

    ax = plt.gca()
    correct_tick_marks(ax)



    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(f"/disks/shear16/herle/code/plots/zobov_{modes[i]}_void_delta_bins_smoothed.png")
