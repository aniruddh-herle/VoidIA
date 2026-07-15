# ==========================================================
# IMPORTS
# ==========================================================

from mnras import *
plt.rcParams['text.usetex'] = False
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy import constants, integrate, special
from joblib import Parallel, delayed

import astropy.units as u
import pyccl as ccl

from astropy.cosmology import FlatLambdaCDM

# ==========================================================
# CONFIGURATION
# ==========================================================

output_dir = "/disks/shear16/herle/code/"

onevoids = True
twovoids = True
compute_th = True

# ----------------------------------------------------------
# Cosmology
# ----------------------------------------------------------

om = 0.306
h = .681

cosmo_astropy = FlatLambdaCDM(H0=100 * h, Om0=om)

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
    n_s=0.967
)

# ----------------------------------------------------------
# IA / normalization
# ----------------------------------------------------------

A1 = 1
C1 = 5e-14


# ----------------------------------------------------------
# rp grids
# ----------------------------------------------------------


# rp_values2v = np.logspace(0, np.log10(maxbin), 30) *h
rp_values2v = np.logspace(np.log10(1), np.log10(600), 30)
print("pauline rp:", rp_values2v)
# ----------------------------------------------------------
# k integration limits
# ----------------------------------------------------------

kperp_min = 1e-5
kperp_max = 50

kz_min = 1e-5
kz_max = 50

# kperp_min = 1e-5
# kperp_max = 1e3

# kz_min = 1e-5
# kz_max = 1e3


# ----------------------------------------------------------
# LOS integration
# ----------------------------------------------------------

PIMAX = 100

# Pi = np.array([
#     -100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
#     0,
#     10,20,30,40,50,60,70,80,90,100
# ])

# pibins = list(zip(Pi[:-1], Pi[1:]))

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

outfilename = "flams_th.txt"

# ==========================================================
# USEFUL FUNCTIONS
# ==========================================================

def pgz(z, sigma=0.1):

    norm = 0.5

    return (
        (1 / norm)
        * (1 / (sigma * np.sqrt(2*np.pi)))
        * np.exp(-0.5 * (z / sigma)**2)
        * (z >= 0)
    )


def pvz(z, sigma=0.1):

    norm = 0.5

    return (
        (1 / norm)
        * (1 / (sigma * np.sqrt(2*np.pi)))
        * np.exp(-0.5 * (z / sigma)**2)
        * (z >= 0)
    )


def Plin(kz, kperp, z):

    k = np.sqrt(kz**2 + kperp**2)

    return cosmo_ccl.linear_power(k, 1 / (1 + z))


def Pnl(kz, kperp, z):

    k = np.sqrt(kz**2 + kperp**2)

    return cosmo_ccl.nonlin_power(k, 1 / (1 + z))


def dxdz(z):

    hh = 1e-5

    return (
        ccl.comoving_radial_distance(cosmo_ccl, 1/(1+z+hh))
        - ccl.comoving_radial_distance(cosmo_ccl, 1/(1+z))
    ) / hh


def int2(z):

    chi = ccl.comoving_radial_distance(cosmo_ccl, 1/(1+z))

    return pvz(z) * pgz(z) / (chi**2 * dxdz(z))


def W(z):

    return 1


def D(z):

    return ccl.growth_factor(cosmo_ccl, 1/(1+z))


def mean_density_h2Mpc(z, omega_m=om, h=h):

    cosmo = FlatLambdaCDM(H0=100, Om0=omega_m) # h=1 cosmology for density in h^2 M_sun/Mpc^3

    rho_matter = cosmo.critical_density(z) * cosmo.Om(z)

    return rho_matter.to(u.Msun / u.Mpc**3).value


# ==========================================================
# 2-VOID TERM
# ==========================================================

def integrand_k_2v(kz, kperp, rp):

    k = np.sqrt(kz**2 + kperp**2)

    if k == 0 or kz == 0:
        return 0

    term1 = kperp**3 / (k**2 * kz)

    term2 = Plin(kz, kperp, 0) # linear power spectrum
    # term2 = Pnl(kz, kperp, 0) # nonlinear power

    term3 = np.sin(kz * PIMAX)

    term4 = special.jv(2, kperp * rp)

    return term1 * term2 * term3 * term4


def integrate_k_2v(rp):

    kperp_limits_local = (kperp_min, kperp_max)
    kz_limits_local = (kz_min, kz_max)

    def integrand_2v(kperp, kz):

        return integrand_k_2v(kz, kperp, rp)

    integral_k, _ = integrate.nquad(
        integrand_2v,
        [kperp_limits_local, kz_limits_local],
        opts={
            'limit': 20,
            'epsabs': 1e-4,
            'epsrel': 1e-4
        }
    )

    return integral_k


def w_v_plus_2v(rp):

    prefactor = (
        A1
        * C1
        * mean_density_h2Mpc(0)
        / (np.pi**2)
    )

    

    integral_z = prefactor * integrate_k_2v(rp)

    return integral_z


# rp_debug = 10.7

# kperp_debug = np.logspace(np.log10(kperp_min), np.log10(kperp_max), 100)
# outer_pauline = []
# for kp in kperp_debug:
#     result, _ = integrate.quad(
#         lambda kz: integrand_k_2v(kz, kp, rp_debug) / special.jv(2, kp * rp_debug),
#         kz_min, kz_max,
#         limit=200, epsabs=1e-4, epsrel=1e-4
#     )
#     outer_pauline.append(result * special.jv(2, kp * rp_debug))

# np.save('/disks/shear16/herle/code/VoidIA/data_checks/debug_kperp_pauline.npy', kperp_debug)
# np.save('/disks/shear16/herle/code/VoidIA/data_checks/debug_integrand_kperp_pauline.npy', np.array(outer_pauline))

# ==========================================================
# THEORY COMPUTATION
# ==========================================================

if compute_th:


    if twovoids:

        print("COMPUTE THE 2 VOID TERM")

        w_values_2v = Parallel(n_jobs=60)(
            delayed(w_v_plus_2v)(rp)
            for rp in rp_values2v
        )

        np.savetxt(
            output_dir + 'th_2voids_' + outfilename,
            w_values_2v
        )
       

# ==========================================================
# PLOT THEORY
# ==========================================================

# plt.figure(figsize=(10, 6))


# if twovoids:

#     w_values_2v = np.loadtxt(
#         output_dir + 'th_2voids_' + outfilename
#     )

#     plt.plot(
#         rp_values2v,
#         w_values_2v,
#         color='navy',
#         label='2-void'
#     )

# plt.xlabel(r'$r_p$')
# plt.ylabel(r'$w_{v+}(r_p)$')

# plt.title(r'Theory $w_{v+}(r_p)$')

# plt.xscale('log')

# plt.grid(True)

# plt.legend()



# plt.savefig('/disks/shear16/herle/code/plots/2void_term_pauline.png')

void_finder = 'zobov'  # 'disperse' or 'zobov'

base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0_trunckrange/'


h = 0.681
# scaling_wvII = 1.0/(np.pi**2)
cosmo = ccl.Cosmology(Omega_c=0.306-0.0486, Omega_b=0.0486, h=0.681, sigma8 = 0.807, n_s=0.967)
C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
# scaling = (0.5*C1rhocrit*cosmo['Omega_m'])/((np.pi**2)*ccl.growth_factor(cosmo, scale_factor))
scaling = (C1rhocrit*cosmo['Omega_m'])/((np.pi**2)*ccl.growth_factor(cosmo, scale_factor))
# scaling = A1* C1* mean_density_h2Mpc(0)/ (np.pi**2)



print("my scaling:", scaling)
print("Pauline's scaling:", A1* C1* mean_density_h2Mpc(0)/ (np.pi**2))

wVII = np.load(base_folder+'/wvII.npy')
wVII *= scaling
#wVII /=h

# plot both theory models
model_rp = np.loadtxt(base_folder+'/w_gp_z0.0.txt')[:, 0] 
# model_rp = model_rp /h  # in Mpc

print("my rp:", model_rp)


fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 10),
    sharex='col',
    gridspec_kw={'height_ratios': [2, 1]}  
)

# Top row = main plot, bottom row = residuals
ax_main = axes[0]
ax_ratio  = axes[1]

ax_main.plot(rp_values2v, w_values_2v, color='navy', label='Pauline')
ax_main.plot(model_rp, wVII, color='red', label='Aniruddh')
ax_main.set_xlabel(r'$r_p [Mpc/h]$')
ax_main.set_ylabel(r'$w_{v+}(r_p)$')
ax_main.set_title(r'Theory $w_{v+}(r_p)$')
ax_main.set_xscale('log')
ax_main.legend(frameon=False)

# Compute ratio
#ratio = w_values_2v / np.interp(rp_values2v, model_rp, wVII) 
ratio = w_values_2v / wVII

print(ratio)
ax_ratio.plot(rp_values2v, ratio, color='purple')
ax_ratio.set_ylim(0.5, 5.0)
ax_ratio.set_xlabel(r'$r_p [Mpc/h]$')
ax_ratio.set_ylabel(r'Pauline / Aniruddh')
ax_ratio.set_xscale('log')

correct_tick_marks(ax_main)
correct_tick_marks(ax_ratio)

plt.savefig('/disks/shear16/herle/code/plots/2void_term_comp.png')

import numpy as np
import matplotlib.pyplot as plt

rp_debug = 10.7

kperp_m = np.load(f'/disks/shear16/herle/code/VoidIA/data_checks/debug_kperp_mine_rp{rp_debug:.1f}.npy')
outer_m  = np.load(f'/disks/shear16/herle/code/VoidIA/data_checks/debug_integrand_kperp_mine_rp{rp_debug:.1f}.npy')
kperp_p  = np.load('/disks/shear16/herle/code/VoidIA/data_checks/debug_kperp_pauline.npy')
outer_p  = np.load('/disks/shear16/herle/code/VoidIA/data_checks/debug_integrand_kperp_pauline.npy')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Multiply yours by kperp to get the true log-space outer integrand
axes[0].plot(kperp_m, outer_m * kperp_m, label='Mine', color='red')
axes[0].plot(kperp_p, outer_p * kperp_p, label='Pauline', color='navy', ls='--')
axes[0].set_xscale('log')
axes[0].set_xlabel(r'$k_\perp$ [h/Mpc]')
axes[0].set_ylabel('outer integrand')
axes[0].set_title(f'$r_p = {rp_debug}$ Mpc/h')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].legend(frameon=False)

# interpolate mine onto Pauline's kperp grid for ratio
outer_m_interp = np.interp(kperp_p, kperp_m, outer_m)
ratio = outer_m_interp / outer_p
print("outer integrand ratio:", ratio)
axes[1].plot(kperp_p, ratio, color='purple')
axes[1].set_xscale('log')
axes[1].set_xlabel(r'$k_\perp$ [h/Mpc]')
axes[1].set_ylabel('Mine / Pauline')
axes[1].set_title('Ratio of outer integrands')
axes[1].axhline(1, color='k', lw=0.5, ls='--')
axes[1].set_ylim(-10, 10)

plt.tight_layout()
plt.savefig(f'/disks/shear16/herle/code/plots/outer_integrand_comparison_rp{rp_debug:.1f}.png', dpi=150)

kz_arr = np.logspace(-6, 1, 1000)
kperp_fixed = 0.3   # near the peak of the outer integrand

inner = []
for kz in kz_arr:
    k = np.sqrt(kz**2 + kperp_fixed**2)
    pk = cosmo_ccl.linear_power(k, 1.0)
    inner.append((kperp_fixed**3 / (k**2 * kz)) * pk * np.sin(kz * PIMAX))

inner = np.array(inner)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(kz_arr, inner, color='black')
axes[0].axvline(1e-5, color='red',  ls='--', label='kz_min Pauline = 1e-5')
axes[0].axvline(1e-6, color='blue', ls='--', label='kz_min Mine = 1e-6')
axes[0].set_xscale('log')
axes[0].set_xlabel(r'$k_z$')
axes[0].set_title(rf'Inner integrand at $k_\perp={kperp_fixed}$')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].legend(frameon=False)

# Cumulative integral from low kz upward
from scipy.integrate import cumulative_trapezoid
cumint = cumulative_trapezoid(inner * kz_arr, np.log(kz_arr), initial=0)
axes[1].plot(kz_arr, cumint, color='black')
axes[1].axvline(1e-5, color='red',  ls='--', label='kz_min Pauline')
axes[1].axvline(1e-6, color='blue', ls='--', label='kz_min Mine')
axes[1].set_xscale('log')
axes[1].set_xlabel(r'$k_z$')
axes[1].set_ylabel('cumulative integral')
axes[1].set_title('How much does low-kz contribute?')
axes[1].legend(frameon=False)

plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/inner_integrand_kz_debug.png', dpi=150)

kperp_arr = np.logspace(-6, 1, 1000)
outer = []
for kp in kperp_arr:
    result, _ = integrate.quad(
        lambda kz: integrand_k_2v(kz, kp, rp_debug),
        1e-5, 10.0,
        limit=200, epsabs=1e-4, epsrel=1e-4
    )
    outer.append(result)

outer = np.array(outer)

from scipy.integrate import cumulative_trapezoid
cumint = cumulative_trapezoid(outer * kperp_arr, np.log(kperp_arr), initial=0)

plt.figure(figsize=(10, 5))
plt.plot(kperp_arr, cumint, color='black')
plt.axvline(1e-5, color='red',  ls='--', label='kperp_min Pauline = 1e-5')
plt.axvline(1e-6, color='blue', ls='--', label='kperp_min Mine = 1e-6')
plt.xscale('log')
plt.xlabel(r'$k_\perp$')
plt.ylabel('cumulative integral')
plt.legend(frameon=False)
plt.savefig('/disks/shear16/herle/code/plots/cumulative_kperp.png', dpi=150)

kz_arr = np.logspace(-6, -4, 1000)
kperp_fixed = 0.3

inner = []
for kz in kz_arr:
    k = np.sqrt(kz**2 + kperp_fixed**2)
    pk = cosmo_ccl.linear_power(k, 1.0)
    inner.append((kperp_fixed**3 / (k**2 * kz)) * pk * np.sin(kz * PIMAX))

inner = np.array(inner)
from scipy.integrate import cumulative_trapezoid
cumint = cumulative_trapezoid(inner * kz_arr, np.log(kz_arr), initial=0)

plt.figure()
plt.plot(kz_arr, cumint)
plt.xscale('log')
plt.axvline(1e-5, color='red', ls='--', label='kz_min = 1e-5')
plt.axvline(1e-6, color='blue', ls='--', label='kz_min = 1e-6')
plt.xlabel(r'$k_z$')
plt.ylabel('cumulative integral 1e-6 to 1e-4')
plt.legend(frameon=False)
plt.savefig('/disks/shear16/herle/code/plots/cumulative_kz_fine.png', dpi=150)

kz_arr = np.logspace(0, 1, 1000)
kperp_fixed = 0.3

inner = []
for kz in kz_arr:
    k = np.sqrt(kz**2 + kperp_fixed**2)
    pk = cosmo_ccl.linear_power(k, 1.0)
    inner.append((kperp_fixed**3 / (k**2 * kz)) * pk * np.sin(kz * PIMAX))

inner = np.array(inner)
from scipy.integrate import cumulative_trapezoid
cumint = cumulative_trapezoid(inner * kz_arr, np.log(kz_arr), initial=0)

plt.figure()
plt.plot(kz_arr, cumint)
plt.xscale('log')
plt.axvline(1.0, color='red',  ls='--', label='your kz stop ~ 1')
plt.axvline(10.,  color='blue', ls='--', label='Pauline kz_max = 10')
plt.xlabel(r'$k_z$')
plt.ylabel('cumulative integral from kz=1 onward')
plt.legend(frameon=False)
plt.savefig('/disks/shear16/herle/code/plots/cumulative_kz_1_to_10.png', dpi=150)

# checing integration limits only:
from scipy import integrate, special
import numpy as np
import pyccl as ccl

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, Omega_b=0.0486, h=0.681,
    A_s=2.099e-9, n_s=0.967, w0=-1.0, wa=0.0,
)

PIMAX = 100.0
kperp_fixed = 0.3

def inner_integrand(kz, kperp):
    k = np.sqrt(kz**2 + kperp**2)
    pk = cosmo_ccl.linear_power(k, 1.0)
    return (kperp**3 / (k**2 * kz)) * pk * np.sin(kz * PIMAX)

# Pauline: integrate from 1e-5 to 10
result_pauline, _ = integrate.quad(
    inner_integrand, 1e-5, 10.0,
    args=(kperp_fixed,),
    limit=500, epsabs=1e-6, epsrel=1e-6
)

# Yours: integrate from 1e-6 to 10
result_mine, _ = integrate.quad(
    inner_integrand, 1e-6, 10.0,
    args=(kperp_fixed,),
    limit=500, epsabs=1e-6, epsrel=1e-6
)

print(f"Pauline inner integral: {result_pauline:.6f}")
print(f"Mine inner integral:    {result_mine:.6f}")
print(f"Ratio mine/pauline:     {result_mine/result_pauline:.6f}")

# What does your integrator give for the same integral?
# Load your saved outer integrand at kperp~0.3
kperp_m = np.load('/disks/shear16/herle/code/VoidIA/data_checks/debug_kperp_mine_rp10.7.npy')
outer_m  = np.load('/disks/shear16/herle/code/VoidIA/data_checks/debug_integrand_kperp_mine_rp10.7.npy')

# Find index closest to kperp=0.3
idx = np.argmin(np.abs(kperp_m - 0.3))
rp_test = 10.7
kperp_val = kperp_m[idx]

# Your integrator's inner integral (strip J2 and kperp^3... wait,
# outer integrand = J2(kperp*rp) * inner_integral, so:
j2 = special.jv(2, kperp_val * rp_test)
your_inner = outer_m[idx] / j2

print(f"kperp = {kperp_val:.4f}")
print(f"quad inner integral:        622.9")
print(f"Your trapezoid inner:       {your_inner:.4f}")
print(f"Ratio your/quad:            {your_inner/622.9:.4f}")
from scipy import integrate, special
import numpy as np
import pyccl as ccl

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, Omega_b=0.0486, h=0.681,
    A_s=2.099e-9, n_s=0.967, w0=-1.0, wa=0.0,
)

PIMAX   = 100.0
rp_test = 10.7
kmin, kmax = 1e-5, 10.0

def full_integrand(kperp, kz, rp):
    k = np.sqrt(kz**2 + kperp**2)
    if k == 0 or kz == 0:
        return 0.0
    pk = cosmo_ccl.linear_power(k, 1.0)
    return (kperp**3 / (k**2 * kz)) * pk * np.sin(kz * PIMAX) * special.jv(2, kperp * rp)

def outer(kperp, rp):
    result, _ = integrate.quad(
        full_integrand, kmin, kmax,
        args=(kperp, rp),
        limit=500, epsabs=1e-6, epsrel=1e-6
    )
    return result

result, _ = integrate.quad(
    outer, kmin, kmax,
    args=(rp_test,),
    limit=500, epsabs=1e-6, epsrel=1e-6
)

print(f"Double quad result at rp={rp_test}: {result:.6f}")

# Compare with your saved value (strip prefactor)
wVII = np.load('/disks/shear16/herle/models/voids/pimax100/redshift0_trunckrange/wvII.npy')
model_rp = np.loadtxt('/disks/shear16/herle/models/voids/pimax100/redshift0_trunckrange/w_gp_z0.0.txt')[:, 0]
idx = np.argmin(np.abs(model_rp - rp_test))
your_raw = np.interp(rp_test, model_rp, wVII)
print(f"Your raw integral at rp={rp_test}: {your_raw:.6f}")
print(f"Ratio double-quad / yours: {result / your_raw:.6f}")

outer_quad = outer(0.3, rp_test)
print(f"quad outer at kperp=0.3: {outer_quad:.4f}")

# Compare with yours
idx = np.argmin(np.abs(kperp_m - 0.3))
print(f"your outer at kperp=0.3: {outer_m[idx]:.4f}")
print(f"ratio: {outer_m[idx]/outer_quad:.4f}")

# First zero of J2 is at x=5.136, so kperp_first_zero = 5.136/rp
kperp_first_zero = 5.136 / rp_test
print(f"First zero of J2 at kperp = {kperp_first_zero:.4f}")

# Evaluate outer integrand densely around the peak
kperp_dense = np.logspace(-2, 0, 500)
outer_dense = np.array([outer(kp, rp_test) for kp in kperp_dense])

plt.figure()
plt.plot(kperp_dense, outer_dense, color='navy', label='quad')
plt.plot(kperp_m, outer_m, color='red', label='yours', alpha=0.7)
plt.axvline(kperp_first_zero, color='k', ls='--', label=f'first J2 zero = {kperp_first_zero:.3f}')
plt.xscale('log')
plt.xlabel(r'$k_\perp$')
plt.legend(frameon=False)
plt.savefig('/disks/shear16/herle/code/plots/outer_integrand_dense.png', dpi=150)

def outer_strict(kperp, rp):
    result, err = integrate.quad(
        full_integrand, 1e-5, 10.0,
        args=(kperp, rp),
        limit=2000, epsabs=1e-8, epsrel=1e-8
    )
    return result, err

# Check at a few kperp values around the spike
for kp in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
    val, err = outer_strict(kp, rp_test)
    print(f"kperp={kp:.3f}: outer={val:.4f}, err={err:.4f}, rel_err={abs(err/val):.4e}")

print(f"Pauline raw at rp~10.7: {integrate_k_2v(10.7):.4f}")
print(f"Double quad reference:   88.9716")
print(f"Your integrator:         56.6700")