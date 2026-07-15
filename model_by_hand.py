import numpy as np
from scipy.interpolate import interp1d

# import fastpt
import fastpt as fpt
import fastpt.HT as HT

import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt

import pandas as pd
from nautilus import Sampler, Prior  # <<< moved Prior import here
from multiprocessing import Pool


import argparse

import os
os.environ["OMP_NUM_THREADS"] = "1"


void_finder = 'zobov'  # 'disperse' or 'zobov'

# base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0/'
base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0_modeltest/'

h = 0.681
cosmo = ccl.Cosmology(Omega_c=0.306-0.0486, Omega_b=0.0486, h=0.681, sigma8 = 0.807, n_s=0.967)
C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo, scale_factor)

wVII = np.load(base_folder+'/wvII.npy')
wVII *= scaling
# wVII /=h



wVI = np.load(base_folder+f'/zobov_fiducial_Rvbin0_wvI.npy')

wVI *= scaling
# wVI /= h

model_rp = np.loadtxt(base_folder+'/w_gp_z0.0.txt')[:, 0] 

# model_rp = model_rp /h  # in Mpc
print("Model rp range: ", model_rp.min(), model_rp.max())

Rv = 15.0
# masks
lower_rp_VI = 6.0
upper_rp_VI = 2.5*Rv

lower_rp_VII = 4.0*Rv
upper_rp_VII = 500.0

model_mask_VI = (model_rp >= lower_rp_VI) & (model_rp <= upper_rp_VI)
model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)


run = 'L1000N1800'
redshift = '0.0'

meas_file = (
    f'/disks/shear16/herle/correlations1/'
    f'correlations_revolver_10_20Mpc_' + run +
    f'_HYDRO_FIDUCIAL_' + redshift + '.h5'
)

df = pd.read_hdf(meas_file, key='a')

rp_meas = df['wgp_rp'].values
wvp_meas = df['wgp_xip'].values 
err_meas = df['wgp_std_from_cov'].values 

# rp_meas = df['wgp_rp'].values /h
# wvp_meas = df['wgp_xip'].values /h 
# err_meas = df['wgp_std_from_cov'].values /h

df = pd.read_hdf(meas_file, key='a')


mask_VI  = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
mask_VII = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)



def model_wVI(p):

    A1_v = p[0]

    wVI_model = wVI[model_mask_VI].copy()
    # wVI_model *= -1.0*A1_v
    wVI_model *= A1_v

    return model_rp[model_mask_VI], wVI_model

def model_wVII(p):

    A1_v = p[0]
    b_v = p[1]

    wVII_model = wVII[model_mask_VII].copy()
    wVII_model *= A1_v*b_v

    return model_rp[model_mask_VII], wVII_model

A, b = 5.0, 1.0

p = [A, b]

rp_model_VI, wVI_model   = model_wVI(p)      # returns only the VI part
rp_model_VII, wVII_model = model_wVII(p)     # returns only the VII part

print("-----------------------------------")

print("rp_model_VI range: ", rp_model_VI.min(), rp_model_VI.max())
print("rp_model_VII range: ", rp_model_VII.min(), rp_model_VII.max())

print("Measurement rp range: ", rp_meas.min(), rp_meas.max())

print("-----------------------------------")
print("A_v: ", A)
print("b_v: ", b)

print("model wv1 :" , wVI_model)
print("model wv2 :" , wVII_model)

# concatenate model pieces into one vector (matching data vector order)
rp_model_total = np.concatenate([rp_model_VI, rp_model_VII])
corr_model_total = np.concatenate([wVI_model, wVII_model])


import matplotlib.pyplot as plt
from mnras import *

plt.rcdefaults()
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 22

plt.figure(figsize=(10, 10))


plt.plot(rp_model_VII, rp_model_VII*wVII_model, alpha=0.2, ls='--')
plt.plot(rp_model_VI,  rp_model_VI*wVI_model,  alpha=0.2, ls='--')

plt.errorbar(
    rp_meas,
    rp_meas*wvp_meas,
    yerr=rp_meas*err_meas,
    fmt='o',
    capsize=3,
    label='Zobov'
)

plt.ylim(-7, 7)

plt.xscale('log')
plt.xlabel(r'$r_p$ [Mpc/h]')
plt.ylabel(r'$r_p w_{\mathrm{V}+} [\mathrm{Mpc}^2/h^2]$')

ax = plt.gca()
correct_tick_marks(ax)

mask_model_VI  = (model_rp >= lower_rp_VI)  & (model_rp <= upper_rp_VI)
mask_model_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)

line_VII, = plt.plot(
    rp_model_VII,
    rp_model_VII*wVII_model,
    label='2-void'
)

line_VI, = plt.plot(
    rp_model_VI,
    rp_model_VI*wVI_model,
    label='1-void'
)

plt.axvspan(lower_rp_VII, upper_rp_VII, color=line_VII.get_color(), alpha=0.08, zorder=0)
plt.axvspan(lower_rp_VI, upper_rp_VI,  color=line_VI.get_color(),  alpha=0.08, zorder=0)



plt.hlines(0, xmin=1, xmax=600, colors='k', linestyles='dotted')

plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/zobov_fiducial_fitby_hand.png')
