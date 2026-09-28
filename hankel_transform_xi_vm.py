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
k_native = np.logspace(log10kmin, log10kmax, 1000001)
k = k_native * h

run      = 'L1000N1800'
mode     = 'FIDUCIAL'
redshift = '0.0'

r_bins = [
    (0,  10,  'Rvbin0'),
    (10, 20,  'Rvbin1'),
    (20, 30,  'Rvbin2')
]

corr_dir  = '/disks/shear16/herle/correlations1/'
temp_dir  = '/disks/shear16/herle/code/temp_data/'
plots_dir = '/disks/shear16/herle/code/plots/'

rmax_trunc  = 600.0
taper_start = 0.80

n_bins = len(r_bins)

# =====================================================
# Layout: 3 rows x n_bins cols
#   row 0: xi_vm in r-space (raw vs tapered)
#   row 1: zoom on zero crossing
#   row 2: Hankel transform (tapered vs untapered)
# =====================================================
fig, axes = plt.subplots(3, n_bins, figsize=(5*n_bins, 13))

for col, (Rmin, Rmax, bin_label) in enumerate(r_bins):

    ax_xi  = axes[0, col]   # xi_vm full range
    ax_zoo = axes[1, col]   # zero-crossing zoom
    ax_k   = axes[2, col]   # Hankel transforms

    # --------------------------------------------------
    # Load
    # --------------------------------------------------
    xi_vm_jk = np.load(
        f'{corr_dir}xi_vm_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'
        f'{run}_HYDRO_{mode}_{redshift}_env_dep_analytical.npy'
    )

    df = pd.read_hdf(
        f'{corr_dir}correlations_joint_revolver_{Rmin}_{Rmax}hMpc_'
        f'{run}_HYDRO_{mode}_{redshift}_env_dep_analytical.h5',
        key='a'
    )

    r_xi    = df['xi_vm_r'].values
    xi_mean = df['xi_vm'].values

    N_jk          = xi_vm_jk.shape[0]
    xi_vm_mean_jk = xi_vm_jk.mean(axis=0)
    xi_vm_err_jk  = np.sqrt(
        (N_jk - 1) / N_jk
        * np.sum((xi_vm_jk - xi_vm_mean_jk[None, :])**2, axis=0)
    )

    # --------------------------------------------------
    # Interpolate onto fine r grid
    # --------------------------------------------------
    r_fine  = np.logspace(np.log10(r_xi.min()), np.log10(r_xi.max()), 2000)
    xi_fine = interp1d(r_xi, xi_mean, kind='cubic')(r_fine)

    # --------------------------------------------------
    # Truncate (shared for both versions)
    # --------------------------------------------------
    mask_r    = r_fine <= rmax_trunc
    r_trunc   = r_fine[mask_r]
    xi_trunc  = xi_fine[mask_r]      # no taper — raw interpolated

    # --------------------------------------------------
    # Cosine taper
    # --------------------------------------------------
    taper = np.ones_like(r_trunc)
    taper_mask = r_trunc >= taper_start * rmax_trunc
    taper[taper_mask] = 0.5 * (
        1.0 + np.cos(
            np.pi * (r_trunc[taper_mask] - taper_start * rmax_trunc)
                   / (rmax_trunc * (1.0 - taper_start))
        )
    )
    xi_tapered = xi_trunc * taper    # tapered version

    # --------------------------------------------------
    # Hankel transforms — both versions
    # --------------------------------------------------
    delta_k_notaper = hankel_transform(r_trunc, xi_trunc,   k_native) / h**3
    delta_k_tapered = hankel_transform(r_trunc, xi_tapered, k_native) / h**3

    np.save(
        f'{temp_dir}zobov_{mode}_xi_vm_{bin_label}_delta_k_particles_tapered.npy',
        delta_k_tapered
    )

    delta_k_shell = np.load(
        f'{temp_dir}zobov_FIDUCIAL_lowres_void_{bin_label}_particles_rmax100.npy'
    )

    # --------------------------------------------------
    # Row 0: xi_vm — raw measurement + both r-space versions
    # --------------------------------------------------
    ax_xi.errorbar(r_xi, xi_mean, yerr=xi_vm_err_jk,
                   fmt='o', ms=3, capsize=2, color='C0',
                   label=r'$\xi_{vm}$ measured', zorder=3)
    ax_xi.plot(r_trunc, xi_trunc,   color='C1', lw=1.5, ls='--', label='no taper')
    ax_xi.plot(r_trunc, xi_tapered, color='C2', lw=1.5,          label='tapered')
    ax_xi.axvspan(taper_start * rmax_trunc, rmax_trunc,
                  color='grey', alpha=0.12, label='taper region')
    ax_xi.axhline(0, ls=':', color='k', lw=0.8)
    ax_xi.set_xscale('log')
    ax_xi.set_xlabel(r'$r$ [Mpc/$h$]')
    ax_xi.set_ylabel(r'$\xi_{vm}(r)$')
    ax_xi.set_title(f'$R_v = {Rmin}$–${Rmax}$ Mpc/$h$')
    ax_xi.legend(frameon=False, fontsize=8)

    # --------------------------------------------------
    # Row 1: zoom on zero crossing
    # --------------------------------------------------
        # --------------------------------------------------
    # Row 1: zoom on the LARGE-r zero crossing
    # (where xi_vm returns to zero from positive/negative
    #  after the compensation wall, not the first crossing)
    # --------------------------------------------------

    # find all sign changes and take the LAST one
    sign_changes = np.where(np.diff(np.sign(xi_mean)))[0]

    if len(sign_changes) > 0:
        r_zero   = r_xi[sign_changes[-1]]   # <-- last crossing, not first
        zoom_lo  = r_zero * 0.2
        zoom_hi  = min(r_zero * 8.0, r_xi.max())
    else:
        zoom_lo = r_xi[len(r_xi)//2]
        zoom_hi = r_xi[-1]

    zoom_mask      = (r_xi   >= zoom_lo) & (r_xi   <= zoom_hi)
    zoom_mask_fine = (r_trunc >= zoom_lo) & (r_trunc <= zoom_hi)

    ax_zoo.errorbar(r_xi[zoom_mask], xi_mean[zoom_mask],
                    yerr=xi_vm_err_jk[zoom_mask],
                    fmt='o', ms=4, capsize=3, color='C0',
                    label=r'$\xi_{vm}$ measured', zorder=3)
    ax_zoo.plot(r_trunc[zoom_mask_fine], xi_trunc[zoom_mask_fine],
                color='C1', lw=1.5, ls='--', label='no taper')
    ax_zoo.plot(r_trunc[zoom_mask_fine], xi_tapered[zoom_mask_fine],
                color='C2', lw=1.5, label='tapered')
    ax_zoo.axhline(0, ls=':', color='k', lw=0.8)
    ax_zoo.set_xscale('log')
    ax_zoo.set_xlabel(r'$r$ [Mpc/$h$]')
    ax_zoo.set_ylabel(r'$\xi_{vm}(r)$')
    ax_zoo.set_title(f'Zero-crossing zoom  ($R_v = {Rmin}$–${Rmax}$)')
    if len(sign_changes) > 0:
        ax_zoo.axvline(r_zero, ls='--', color='grey', lw=1,
                       label=f'$r_0 \\approx {r_zero:.1f}$ Mpc/$h$')
    ax_zoo.legend(frameon=False, fontsize=8)

    # --------------------------------------------------
    # Row 2: Hankel transforms — tapered vs untapered vs shell
    # --------------------------------------------------
    ax_k.plot(k, delta_k_notaper / 1e3, color='C1', ls='--', lw=1.5, label=r'$\xi_{vm}$ no taper')
    ax_k.plot(k, delta_k_tapered / 1e3, color='C2', lw=1.5,          label=r'$\xi_{vm}$ tapered')
    ax_k.plot(k, delta_k_shell   / 1e3, color='C0', ls=':',  lw=1.5, label='Shell profile')
    ax_k.axhline(0, ls='--', color='k', lw=0.8)
    ax_k.set_xscale('log')
    ax_k.set_xlim(1e-5, 1e3)
    ax_k.set_xlabel(r'$k \; [\mathrm{Mpc}^{-1}]$')
    ax_k.set_ylabel(r'$\Delta_v(k) \times 10^3$')
    ax_k.set_title(f'Hankel  ($R_v = {Rmin}$–${Rmax}$ Mpc/$h$)')
    ax_k.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(
    f'{plots_dir}zobov_{mode}_xi_vm_taper_comparison_rmax{int(rmax_trunc)}.png',
    dpi=300, bbox_inches='tight'
)