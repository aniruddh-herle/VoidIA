import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from mnras import *
from scipy.interpolate import interp1d
plt.rcParams['text.usetex'] = False
h = 0.681

def get_radius_from_name(fname):
    base = os.path.basename(fname)
    return float(base.split("Rv_")[1].split("_")[0])

def stack_stats(arr):
    med = np.median(arr, axis=0)
    p16 = np.percentile(arr, 16, axis=0)
    p84 = np.percentile(arr, 84, axis=0)
    return med, p16, p84

modes        = ['FIDUCIAL']
mean_densities = [1.846582e-01]

run      = 'L1000N1800'
redshift = '0.0'

bin_labels = ['0_10', '10_20', '20_30']
bin_edges  = [(0, 10), (10, 20), (20, 30)]
bin_titles = ["0–10 Mpc/h", "10–20 Mpc/h", "20–30 Mpc/h"]

# =====================================================
# Helper: load xi_vm from a given h5 + jk npy suffix
# =====================================================
def load_xi_vm(bin_label, suffix):
    meas_file = (
        f'/disks/shear16/herle/correlations1/'
        f'correlations_joint_revolver_{bin_label}hMpc_{run}'
        f'_HYDRO_FIDUCIAL_{redshift}_env_dep_analytical{suffix}.h5'
    )
    jk_file = (
        f'/disks/shear16/herle/correlations1/'
        f'xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
        f'_HYDRO_FIDUCIAL_{redshift}_env_dep_analytical{suffix}.npy'
    )
    df          = pd.read_hdf(meas_file, key='a')
    r           = df['xi_vm_r'].values
    xi          = df['xi_vm'].values
    jk          = np.load(jk_file)
    NPatches    = jk.shape[0]
    xi_mean_jk  = np.mean(jk, axis=0)
    diff        = jk - xi_mean_jk
    cov         = ((NPatches - 1) / NPatches) * (diff.T @ diff)
    err         = np.sqrt(np.diag(cov))
    return r, xi, err

# Three xi_vm variants
suffixes = {
    'original':   '',
    'more rp':    '_morerp',
    'more rp\nless rands': '_morerp_lessrands',
}

for i in range(len(modes)):

    # --------------------------------------------------
    # Shell profiles
    # --------------------------------------------------
    profile_dir  = f"/disks/shear16/herle/void_profiles/zobov_hydro_particles/{modes[i]}/"
    mean_density = mean_densities[i]

    files  = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
    r_phys = None
    bins   = [[] for _ in range(3)]

    for f in files:
        d        = np.load(f)
        r_phys   = d["r_phys"]
        rho_phys = d["density_phys"]
        delta    = rho_phys / mean_density - 1
        Rv       = get_radius_from_name(f)
        for b, (lo, hi) in enumerate(bin_edges):
            if lo <= Rv < hi:
                bins[b].append(delta)

    bin_stats = [stack_stats(np.array(b)) for b in bins]

    # --------------------------------------------------
    # Load all xi_vm variants
    # --------------------------------------------------
    xi_vm_all = {}
    for label, suffix in suffixes.items():
        xi_vm_all[label] = []
        for bin_label in bin_labels:
            try:
                r, xi, err = load_xi_vm(bin_label, suffix)
                xi_vm_all[label].append((r, xi, err))
            except Exception as e:
                print(f"Could not load {label} for bin {bin_label}: {e}")
                xi_vm_all[label].append(None)

    # --------------------------------------------------
    # Plot — 1x3 grid
    # --------------------------------------------------
    colors  = {'original': 'C1', 'more rp': 'C2', 'more rp\nless rands': 'C3'}
    markers = {'original': 'o',  'more rp': 's',  'more rp\nless rands': '^'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for b in range(3):

        ax = axes[b]
        med, p16, p84 = bin_stats[b]

        # Shell profile
        ax.plot(r_phys, med, color='C0', lw=1.5, label=r"$\Delta_v(r)$ shells")
        ax.fill_between(r_phys, p16, p84, alpha=0.2, color='C0')

        # xi_vm variants
        for label, data_list in xi_vm_all.items():
            entry = data_list[b]
            if entry is None:
                continue
            r, xi, err = entry
            ax.errorbar(
                r, xi, yerr=err,
                fmt=markers[label], ms=3, capsize=2, alpha=0.8,
                color=colors[label],
                label=rf"$\xi_{{vm}}$ {label}"
            )

        ax.set_xscale('log')
        ax.axhline(0, ls="--", color='k', lw=0.8)
        ax.set_xlabel("r [Mpc/h]")
        ax.set_ylabel(r"$\xi_{vm}(r)$" if b == 0 else "")
        ax.set_title(bin_titles[b])
        ax.legend(frameon=False, fontsize=8)
        correct_tick_marks(ax)

    plt.tight_layout()
    plt.savefig(
        f"/disks/shear16/herle/code/plots/"
        f"void_delta_vs_xivm_zobov_{modes[i]}_allbins_rpcomparison.png",
        dpi=300, bbox_inches='tight'
    )

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

h = 0.681

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
    return 4*np.pi * integrand.sum(axis=1)

log10kmin = -5
log10kmax = 2
k_native  = np.logspace(log10kmin, log10kmax, 1000001)
k         = k_native * h

run      = 'L1000N1800'
mode     = 'FIDUCIAL'
redshift = '0.0'

corr_dir  = '/disks/shear16/herle/correlations1/'
temp_dir  = '/disks/shear16/herle/code/temp_data/'
plots_dir = '/disks/shear16/herle/code/plots/'

r_bins = [
    (0,  10,  '0_10',  'Rvbin0'),
    (10, 20,  '10_20', 'Rvbin1'),
    (20, 30,  '20_30', 'Rvbin2'),
]

# The 4 cases: (label, suffix for corr files, shell_bin_label or None)
cases = [
    ('$\\xi_{vm}$ original',        '_env_dep_analytical',                  None),
    ('$\\xi_{vm}$ more rp',         '_env_dep_analytical_morerp',           None),
    ('$\\xi_{vm}$ more rp less rands', '_env_dep_analytical_morerp_lessrands', None),
    ('Shell profile',                None,                                   'shell'),
]

case_colors  = ['C1', 'C2', 'C3', 'C0']
case_ls      = ['-',  '-',  '-',  '--']

rmax_trunc = 600.0

n_bins = len(r_bins)
fig, axes = plt.subplots(1, n_bins, figsize=(5*n_bins, 5))

for col, (Rmin, Rmax, bin_label_corr, bin_label_shell) in enumerate(r_bins):

    ax = axes[col]

    for (case_label, suffix, shell_flag), color, ls in zip(cases, case_colors, case_ls):

        # --------------------------------------------------
        # Shell profile case
        # --------------------------------------------------
        if shell_flag == 'shell':
            delta_k = np.load(
                f'{temp_dir}zobov_FIDUCIAL_lowres_void_{bin_label_shell}_particles_rmax100.npy'
            )
            ax.plot(k, delta_k / 1e3, color=color, ls=ls, lw=1.5, label=case_label)
            continue

        # --------------------------------------------------
        # xi_vm cases
        # --------------------------------------------------
        df = pd.read_hdf(
            f'{corr_dir}correlations_joint_revolver_{bin_label_corr}hMpc_'
            f'{run}_HYDRO_FIDUCIAL_{redshift}{suffix}.h5',
            key='a'
        )

        r_xi    = df['xi_vm_r'].values
        xi_mean = df['xi_vm'].values

        # Interpolate onto fine r grid
        r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 2000)
        xi_fine = interp1d(r_xi, xi_mean, kind='cubic')(r_fine)

        # Truncate
        mask_r   = r_fine <= rmax_trunc
        r_trunc  = r_fine[mask_r]
        xi_trunc = xi_fine[mask_r]

        # Hankel transform
        delta_k = hankel_transform(r_trunc, xi_trunc, k_native) / h**3

        ax.plot(k, delta_k / 1e3, color=color, ls=ls, lw=1.5, label=case_label)

    ax.axhline(0, ls=':', color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1e3)
    ax.set_xlabel(r'$k \; [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$\Delta_v(k) \times 10^3$')
    ax.set_title(f'$R_v = {Rmin}$–${Rmax}$ Mpc/$h$')
    ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(
    f'{plots_dir}zobov_{mode}_hankel_4cases_comparison.png',
    dpi=300, bbox_inches='tight'
)