# ============================================================
# Generalised ZOBOV void fit + plot script
# One combined measurement+model figure
# Separate triangle plots
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyccl as ccl
from getdist import plots, MCSamples
from mnras import *

plt.rcdefaults()
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 22


# ============================================================
# helpers
# ============================================================

def read_nautilus_samples(npz_filename):
    data = np.load(npz_filename)
    return data['samples'], data['weights'], data['logl']


def compute_chi2(logl, samples, corr):
    logL_max = np.max(logl)
    chi2_eff = -2 * logL_max
    Ndof = len(corr) - samples.shape[1]
    return chi2_eff, chi2_eff / Ndof


# ============================================================
# GLOBAL figure (shared by all bins)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ============================================================
# main function
# ============================================================

def run_void_bin(Rmin, Rmax, color,
                 run='L1000N1800',
                 redshift='0.0'):

    tag = f"{int(Rmin)}_{int(Rmax)}Mpc"

    print(f'\n==============================')
    print(f'Running void bin: {tag}')
    print(f'==============================')

    # --------------------------------------------------------
    # paths
    # --------------------------------------------------------
    chain_file = (
        f'/disks/shear16/herle/models/models/'
        f'nautilus_chain_zobov_voids_1000_Rv{tag}_pimax100_{redshift}.npz'
    )

    meas_file = (
        f'/disks/shear16/herle/correlations1/'
        f'correlations_voids_zobov_{tag}_{run}_HYDRO_FIDUCIAL_{redshift}.h5'
    )

    base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0/'


    # ========================================================
    # chain
    # ========================================================
    samples, weights, logl = read_nautilus_samples(chain_file)

    gd = MCSamples(samples=samples, weights=weights,
                   names=['A1_v', 'b_v'],
                   labels=['A1_v', 'b_v'])


    # best fit
    A1_v = gd.get1DDensity('A1_v').x[np.argmax(gd.get1DDensity('A1_v').P)]
    b_v  = gd.get1DDensity('b_v').x[np.argmax(gd.get1DDensity('b_v').P)]

    print(f'A1_v = {A1_v:.3f}, b_v = {b_v:.3f}')


    # ========================================================
    # triangle plot (separate)
    # ========================================================
    g = plots.get_subplot_plotter()
    g.triangle_plot([gd], ['A1_v', 'b_v'], filled=[True])
    plt.savefig(f'/disks/shear16/herle/code/plots/zobov_triangle_{tag}.png')
    plt.close()


    # ========================================================
    # cosmology
    # ========================================================
    h = 0.681

    cosmo = ccl.Cosmology(
        Omega_c=0.306-0.0486,
        Omega_b=0.0486,
        h=h,
        sigma8=0.807,
        n_s=0.967
    )

    scaling = (
        0.0134 * cosmo['Omega_m'] /
        (np.pi**2) /
        ccl.growth_factor(cosmo, 1.0)
    )


    # ========================================================
    # model
    # ========================================================
    base_wVI  = np.load(base_folder+'zobov_wvI.npy') * scaling / h
    base_wVII = np.load(base_folder+'wvII.npy')      * scaling / h
    rp_model = np.loadtxt(base_folder+'w_gp_z0.0.txt')[:, 0] / h

    wVI  = base_wVI  * A1_v
    wVII = base_wVII * A1_v * b_v


    # ========================================================
    # NEW rp scheme  (YOUR REQUEST)
    # ========================================================
    lower_rp_VI  = 6
    upper_rp_VI  = 2.5 * Rmax
    lower_rp_VII = 4.0 * Rmax
    upper_rp_VII = 600

    mask_VI  = (rp_model >= lower_rp_VI)  & (rp_model <= upper_rp_VI)
    mask_VII = (rp_model >= lower_rp_VII) & (rp_model <= upper_rp_VII)


    # ========================================================
    # measurement
    # ========================================================
    df = pd.read_hdf(meas_file, key='a')

    rp_meas  = df['wgp_rp'].values
    wvp_meas = df['wgp_xip'].values
    err_meas = df['wgp_std_from_cov'].values


    # ========================================================
    # plotting (GLOBAL AXIS)
    # ========================================================

    # measurement
    ax.errorbar(
        rp_meas,
        rp_meas*wvp_meas,
        yerr=rp_meas*err_meas,
        fmt='o',
        capsize=3,
        color=color
    )

    # models (same color, dashed/solid)
    ax.plot(
        rp_model[mask_VI],
        rp_model[mask_VI]*wVI[mask_VI],
        color=color,
        ls='--'
    )

    ax.plot(
        rp_model[mask_VII],
        rp_model[mask_VII]*wVII[mask_VII],
        color=color,
        ls='-',
        label= rf'${Rmin} < R_v < {Rmax}\ \mathrm{{Mpc}}$'

    )


# ============================================================
# RUN ALL BINS
# ============================================================

void_bins = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 50)
]

for i, (Rmin, Rmax) in enumerate(void_bins):
    run_void_bin(Rmin, Rmax, color=color_cycle[i % len(color_cycle)])


# ============================================================
# final formatting (ONE time only)
# ============================================================

ax.set_xscale('log')
ax.set_ylim(-15, 15)
ax.hlines(0, 1, 600, linestyles='dotted')

correct_tick_marks(ax)

ax.set_xlabel(r'$r_p$ [Mpc]')
ax.set_ylabel(r'$r_p w_{\mathrm{V}+} [\mathrm{Mpc}^2]$')

ax.legend(frameon=False, ncol=2)

plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/zobov_Rvbins_meas.png')
plt.close()
