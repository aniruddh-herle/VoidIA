import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from mnras import *
from scipy.interpolate import interp1d

plt.rcParams['text.usetex'] = False
h = 0.681

# ── helpers ───────────────────────────────────────────────────────────────────

def get_radius_from_name(fname):
    base = os.path.basename(fname)
    return float(base.split("Rv_")[1].split("_")[0])

def stack_stats_sem(arr):
    """Mean profile with error = sigma / sqrt(Nv)."""
    arr   = np.array(arr)
    Nv    = arr.shape[0]
    mean  = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0, ddof=1)
    sem   = sigma / np.sqrt(Nv)
    return mean, mean - sem, mean + sem

def j0(x):
    out = np.ones_like(x)
    mask = x != 0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def hankel_transform(r, delta_r, k_vals):
    dr        = np.gradient(r)
    kr        = np.outer(k_vals, r)
    integrand = j0(kr) * (r**2 * dr * delta_r)
    return 4 * np.pi * integrand.sum(axis=1)

# ── settings ──────────────────────────────────────────────────────────────────

run           = 'L2800N5040'
mode          = 'FIDUCIAL'
redshift      = '0.0'
mean_density  = 1.846582e-01

corr_dir    = '/disks/shear16/herle/correlations1/'
temp_dir    = '/disks/shear16/herle/code/temp_data/'
plots_dir   = '/disks/shear16/herle/code/plots/'
profile_dir = f"/disks/shear16/herle/void_profiles/zobov_hydro_particles/{mode}/"

bin_labels = ['0_10', '10_15', '15_20', '20_25', '25_30', '30_50']
bin_edges  = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
bin_titles = ["0–10 Mpc/h", "10–15 Mpc/h", "15–20 Mpc/h",
              "20–25 Mpc/h", "25–30 Mpc/h", "30–50 Mpc/h"]
shell_keys = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5']

n_bins = len(bin_labels)   # ← drives everything below

xi_suffix  = '_env_dep_analytical_morerp'

log10kmin  = -5
log10kmax  = 2
k_native   = np.logspace(log10kmin, log10kmax, 1000001)
k          = k_native * h
rmax_trunc = 400.0

col_shell = 'C0'
col_xi    = 'C2'

# ── load shell profiles ───────────────────────────────────────────────────────

files  = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
r_phys = None
bins   = [[] for _ in range(n_bins)]   # was hardcoded to 3

for f in files:
    d        = np.load(f)
    r_phys   = d["r_phys"]
    rho_phys = d["density_phys"]
    delta    = rho_phys / mean_density - 1
    Rv       = get_radius_from_name(f)
    for b, (lo, hi) in enumerate(bin_edges):
        if lo <= Rv < hi:
            bins[b].append(delta)

bin_stats = [stack_stats_sem(b) for b in bins]
bin_Nv    = [len(b) for b in bins]

# ── load xi_vm ────────────────────────────────────────────────────────────────

def load_xi_vm(bin_label):
    # bin_label is e.g. '10_15', matching {Rmin}_{Rmax} in the saved filenames
    meas_file = (
        f'{corr_dir}correlations_joint_revolver_{bin_label}hMpc_'
        f'{run}_HYDRO_{mode}_{redshift}{xi_suffix}.h5'
    )
    jk_file = (
        f'{corr_dir}xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_'
        f'{run}_HYDRO_{mode}_{redshift}{xi_suffix}.npy'
    )
    # JK array is always saved; compute mean from it rather than relying on
    # xi_vm columns in the h5 (which may not be present for all runs)
    jk         = np.load(jk_file)          # shape (NPatches, Nr)
    NPatches   = jk.shape[0]
    xi         = np.mean(jk, axis=0)       # mean over JK patches
    diff       = jk - xi
    cov        = ((NPatches - 1) / NPatches) * (diff.T @ diff)
    err        = np.sqrt(np.diag(cov))

    # r-grid: prefer xi_vm_r from h5; fall back to xi_vm_r column if named differently
    df = pd.read_hdf(meas_file, key='a')
    if 'xi_vm_r' in df.columns:
        r = df['xi_vm_r'].values
    else:
        # last resort: pick the first column that looks like a separation axis
        r_cols = [c for c in df.columns if 'xi_vm' in c and '_r' in c]
        if r_cols:
            r = df[r_cols[0]].values
        else:
            raise KeyError(f"Cannot find xi_vm r-column in {meas_file}. "
                           f"Available columns: {list(df.columns)}")
    return r, xi, err

xi_data = []
for bl in bin_labels:
    try:
        xi_data.append(load_xi_vm(bl))
    except Exception as e:
        print(f"Could not load xi_vm for bin {bl}: {e}")
        xi_data.append(None)

# ── Hankel transforms ─────────────────────────────────────────────────────────

def hankel_shell(bin_idx):
    delta_k = np.load(
        f'{temp_dir}zobov_FIDUCIAL_2p8_void_delta_k_{shell_keys[bin_idx]}'
        f'_raw_particles_rmax400_mean.npy'
    )
    return delta_k

def hankel_xi(bin_idx):
    entry = xi_data[bin_idx]
    if entry is None:
        return None
    r_xi, xi_mean, _ = entry
    r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 2000)
    xi_fine = interp1d(r_xi, xi_mean, kind='cubic')(r_fine)
    mask    = r_fine <= rmax_trunc
    delta_k = hankel_transform(r_fine[mask], xi_fine[mask], k_native) / h**3
    return delta_k

# ── plot: 2 rows × n_bins cols ───────────────────────────────────────────────

fig, axes = plt.subplots(2, n_bins, figsize=(4.5 * n_bins, 9))

for b in range(n_bins):   # was range(3)

    # ── top row: real-space profiles ──────────────────────────────────────────
    ax = axes[0, b]
    mean, lo, hi = bin_stats[b]
    Nv = bin_Nv[b]

    ax.plot(r_phys, mean, color=col_shell, lw=1.5,
            label=rf"$\Delta_v(r)$ shells ($N_v={Nv}$)")
    ax.fill_between(r_phys, lo, hi, alpha=0.25, color=col_shell)

    entry = xi_data[b]
    if entry is not None:
        r, xi, err = entry
        ax.errorbar(r, xi, yerr=err,
                    fmt='s', ms=3, capsize=2, alpha=0.8,
                    color=col_xi, label=r"$\xi_{vm}$ more rp")

    ax.set_xscale('log')
    ax.axhline(0, ls='--', color='k', lw=0.8)
    ax.set_xlabel(r"$r \; [\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\xi_{vm}(r)$" if b == 0 else "")
    ax.set_title(bin_titles[b])
    ax.legend(frameon=False, fontsize=8)
    correct_tick_marks(ax)

    # ── bottom row: Hankel transforms ─────────────────────────────────────────
    ax = axes[1, b]

    try:
        dk_shell = hankel_shell(b)
        ax.plot(k, dk_shell / 1e3, color=col_shell, lw=1.5, ls='--',
                label=r"$\Delta_v(k)$ shells")
    except Exception as e:
        print(f"Could not load shell Hankel for bin {b}: {e}")

    dk_xi = hankel_xi(b)
    if dk_xi is not None:
        ax.plot(k, dk_xi / 1e3, color=col_xi, lw=1.5,
                label=r"$\xi_{vm}$ more rp")

    ax.axhline(0, ls=':', color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1e3)
    ax.set_xlabel(r"$k \; [\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta_v(k) \times 10^3$" if b == 0 else "")
    ax.set_title(bin_titles[b])
    ax.legend(frameon=False, fontsize=8)
    correct_tick_marks(ax)

fig.tight_layout()
fig.savefig(
    f'{plots_dir}zobov_{mode}_profiles_hankel_2cases_2x{n_bins}.png',
    dpi=300, bbox_inches='tight'
)