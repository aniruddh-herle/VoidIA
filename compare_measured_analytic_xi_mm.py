
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from getdist import plots, MCSamples
import re
import h5py
import pyccl as ccl
from scipy.interpolate import interp1d
from mcfit import P2xi

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

fig, ax = plt.subplots()


ax.plot(r_mm, xi_mm, label='analytic')
ax.set_xscale('log')
ax.set_xlabel(r'$r$ [Mpc/h]')
ax.set_ylabel(r'$\xi_{mm}(r)$')
ax.set_yscale('log')

plt.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/compare_xi_mm.png')