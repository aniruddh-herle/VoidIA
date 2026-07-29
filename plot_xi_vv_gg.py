import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from getdist import plots, MCSamples
import re
import h5py
import pyccl as ccl
from scipy.interpolate import interp1d
from mcfit import P2xi

names = ['bg','b_dummy']  
labels = ['b_1','b_d']
run = 'L1000N1800'
redshift = '0.0'

meas_file = (
    f'/disks/shear16/herle/correlations1/'
    f'correlations_joint_revolver_10_20hMpc_{run}_HYDRO_FIDUCIAL_{redshift}_3d.h5'
)

df = pd.read_hdf(meas_file)

# --------------------------
# Measurements
# --------------------------

rp_xi_vv = df['xi_vv_r'].values
xi_vv = df['xi_vv'].values
xi_vv_err = df['xi_vv_err'].values


rp_xi_gg = df['xi_gg_r'].values
xi_gg = df['xi_gg'].values
xi_gg_err = df['xi_gg_err'].values




# --------------------------
# Figure
# --------------------------

fig, axs = plt.subplots(
    2,
    3,
    figsize=(10, 10),
    sharex=True
)

##########################################################
# xi_vv
##########################################################

ax = axs[0, 0]

ax.errorbar(
    rp_xi_vv,
    xi_vv,
    yerr=xi_vv_err,
    fmt='o',
    color='k',
    capsize=3
)


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$\xi_{\rm vv}$')
ax.set_xlim(1,300)

##########################################################
# xi_gg
##########################################################

ax = axs[0, 1]

ax.errorbar(
    rp_xi_gg,
    xi_gg,
    yerr=xi_gg_err,
    fmt='o',
    color='k',
    capsize=3
)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$\xi_{\rm gg}$')
ax.set_xlim(1,300)

##########################################################
# wvv
##########################################################

ax = axs[1, 2]

mask = (xi_vv > 0) & (xi_gg > 0)

rp = rp_xi_vv[mask]
vv = xi_vv[mask]
gg = xi_gg[mask]
vv_err = xi_vv_err[mask]
gg_err = xi_gg_err[mask]

ratio_sq = vv / gg

ratio_sq_err = np.sqrt(
    (vv_err / gg)**2 +
    (vv * gg_err / gg**2)**2
)

ratio = np.sqrt(ratio_sq)

ratio_err = ratio_sq_err / (2 * ratio)

ax.errorbar(
    rp,
    ratio,
    yerr=ratio_err,
    fmt='o',
    capsize=3
)

# Reference value
ax.axhline(1.575 / 1.611, color='green', ls='--', lw=2)

mean_mask = (rp >= 30) & (rp <= 100)

mean_ratio = np.mean(ratio[mean_mask])
ax.axhline(mean_ratio, color='C1', ls='--',
           label=fr'Mean (30-100) = {mean_ratio:.3f}')
ax.set_xscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$b_v/b_g=\sqrt{\xi_{\rm vv}/\xi_{\rm gg}}$')
ax.set_xlim(1, 300)


## get analytic xi_mm from the flamingo power spectrum
"""
Compute xi_mm(r), the real-space matter correlation function, directly from
P(k) via a Hankel transform (FFTLog, through the `mcfit` package) -- no
particle pair-counting required.

    xi(r) = (1/2pi^2) * Integral[ k^2 P(k) sin(kr)/(kr) dk ]

Requires: pip install mcfit --break-system-packages
"""


cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306 - 0.0486,
    Omega_b=0.0486,
    h=0.681,
    sigma8=0.807,
    n_s=0.967,
)

z_target = 0.0
a_target = 1.0 / (1.0 + z_target)
h = cosmo_ccl['h']
 
# Work natively in h/Mpc: build the k-grid in h/Mpc, convert to Mpc^-1 only
# for the CCL call (CCL's native units), then convert P(k) from Mpc^3 to
# (Mpc/h)^3. The Hankel transform of a (k, P(k)) pair in h/Mpc / (Mpc/h)^3
# then returns r directly in Mpc/h -- xi itself is dimensionless either way.
k_hmpc = np.geomspace(1e-5, 1e3, 8192)          # h/Mpc
pk_ccl = ccl.nonlin_matter_power(cosmo_ccl, k_hmpc * h, a_target)  # Mpc^3
pk_hmpc = pk_ccl * h**3                          # (Mpc/h)^3
 
r_mm, xi_mm = P2xi(k_hmpc, lowring=True)(pk_hmpc, extrap=True)
 
# Keep only the r-range where the transform is trustworthy (see note below).
# Now in Mpc/h.
keep = (r_mm >= 0.1) & (r_mm <= 300.0)
r_mm, xi_mm = r_mm[keep], xi_mm[keep]
 

# ============================================================
# 5. Quick plot
# ============================================================

ax = axs[0, 2]

ax.plot(r_mm, xi_mm)
ax.set_xscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$\xi_{mm}(r)$')
ax.set_yscale('log')




##########################################################
# 
##########################################################
from scipy.interpolate import interp1d


def read_nautilus_samples(npz_filename):
    data = np.load(npz_filename)
    samples = data['samples']        # shape (Nsamples, Nparams)
    weights = data['weights']        # shape (Nsamples,)
    logl = data['logl']              # shape (Nsamples,)
    return samples, weights, logl

filename = '/disks/shear16/herle/models/models/nautilus_chain_voids_hMpc_L1000N1800_HYDRO_FIDUCIAL_0.0_gg2.npz'
samples, weights, logl = read_nautilus_samples(filename)
getdist = MCSamples(samples=samples, 
                            weights=weights,
                            names=names, 
                            labels=labels, 
                            ignore_rows=0)


d = getdist.get1DDensity('bg')
xyind = np.where(d.P == np.amax(d.P))
bg = d.x[xyind[0]][0]
errors_density = d.getLimits(0.68)

lower_error = errors_density[0]
upper_error = errors_density[1]

bg_error = (upper_error - lower_error) / 2.0


ax = axs[1, 0]

# xi_mm is defined on its own (much finer) r_mm grid from the Hankel
# transform -- interpolate it onto the xi_vv measurement bins before
# taking any ratio, since the two arrays are not aligned point-by-point.
xi_mm_interp = interp1d(
    r_mm, xi_mm,
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,   # anything outside r_mm's trusted range -> NaN, not extrapolated
)
xi_mm_on_vv_grid = xi_mm_interp(rp_xi_vv)

mask = (xi_vv > 0) & (xi_mm_on_vv_grid > 0) & ~np.isnan(xi_mm_on_vv_grid)

rp = rp_xi_vv[mask]
vv = xi_vv[mask]
mm = xi_mm_on_vv_grid[mask]
vv_err = xi_vv_err[mask]

ratio = np.sqrt(vv / mm)

mean_mask = (rp >= 30) & (rp <= 100)

mean_ratio = np.mean(ratio[mean_mask])
std_ratio = np.std(ratio[mean_mask], ddof=1)
stderr_ratio = std_ratio / np.sqrt(np.sum(mean_mask))

print(f"sqrt(xi_vv/xi_mm), 30-100 Mpc/h:")
print(f"Mean   = {mean_ratio:.4f}")
print(f"Std    = {std_ratio:.4f}")
print(f"StdErr = {stderr_ratio:.4f}")

ratio_err = vv_err / (2 * mm * ratio)

ax.errorbar(
    rp,
    ratio,
    yerr=ratio_err,
    fmt='o',
    capsize=3,
)

ax.axhline(mean_ratio, color='C1', ls='--',
           label=fr'Mean (30-100) = {mean_ratio:.3f}')
ax.legend(loc='upper left', frameon=False)
ax.set_xscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$b_v = \sqrt{\xi_{\rm vv}/\xi_{\rm mm}}$')
ax.set_xlim(1, 300)



###############################


ax = axs[1, 1]


xi_mm_on_gg_grid = xi_mm_interp(rp_xi_gg)

mask = (xi_gg > 0) & (xi_mm_on_gg_grid > 0) & ~np.isnan(xi_mm_on_gg_grid)

rp = rp_xi_vv[mask]
gg = xi_gg[mask]
mm = xi_mm_on_gg_grid[mask]
gg_err = xi_gg_err[mask]

ratio = np.sqrt(gg / mm)
mean_mask = (rp >= 30) & (rp <= 100)

mean_ratio = np.mean(ratio[mean_mask])
std_ratio = np.std(ratio[mean_mask], ddof=1)
stderr_ratio = std_ratio / np.sqrt(np.sum(mean_mask))

print(f"sqrt(xi_gg/xi_mm), 30-100 Mpc/h:")
print(f"Mean   = {mean_ratio:.4f}")
print(f"Std    = {std_ratio:.4f}")
print(f"StdErr = {stderr_ratio:.4f}")
ratio_err = gg_err / (2 * mm * ratio)

ax.errorbar(
    rp,
    ratio,
    yerr=ratio_err,
    fmt='o',
    capsize=3,
)

ax.axhline(y=bg, color='r', label='Fit from GNAT')
ax.axhline(mean_ratio, color='C1', ls='--',
           label=fr'Mean (30-100) = {mean_ratio:.3f}')
ax.legend(loc='upper left', frameon=False)

ax.set_xscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$b_g = \sqrt{\xi_{\rm gg}/\xi_{\rm mm}}$')
ax.set_xlim(1, 300)


plt.tight_layout()

plt.savefig(
    '/disks/shear16/herle/code/plots/xi_gg_vv.png',
    dpi=300
)

