import numpy as np
from scipy.interpolate import interp1d

# import fastpt
import fastpt as fpt
import fastpt.HT as HT

import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt

import pandas as pd

import argparse
import h5py
import re
import os

os.environ["OMP_NUM_THREADS"] = "1"


cosmo = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
    n_s=0.967
)
parser = argparse.ArgumentParser()
parser.add_argument('-n_cores', type=int, default=128, help='Number of CPU cores to use for multiprocessing.')
args = parser.parse_args()

z_i = 0.0

h=0.681

# log10kmin = -5
# log10kmax = 1.7
# k=np.logspace(log10kmin, log10kmax, 1000001)

# Dictionary with the cosmological parameters, must be compatible as pyccl cosmology input
#cosmology_dict = {'Omega_c':0.306-0.0486, 'Omega_b':0.0486, 'h':0.681, 'A_s':2.099e-9, 'n_s':0.967}
cosmology_dict = {'Omega_c':0.306-0.0486, 
    'Omega_b':0.0486,  
    'h':0.681, 
    'sigma8':0.807, 
    'n_s':0.967}

integration_options={'kmin':1e-5, 'kmax':1e2, 'rel_eps':1e-5, 'n_spline':300}
redshift = [0.0]
interpolation_settings = {'r_min_over_h':0.1*h, 'r_max_over_h':600*h, 'number_of_points':20}
multipoles_options = {'mu_r_spline_number':50, 'perpendicular_scale_cut_over_h': 0.1}
simulation_settings = {'pi_max_over_h':100*h}
workdir = '.'
sim_outputdir = '/disks/shear16/herle/models/voids/pimax100/redshift0_trunckrange3/'

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from calculate_prediction_base import fitIntrinsicAlignmentAmplitude


def get_wVII_term():

    prediction_calculator = fitIntrinsicAlignmentAmplitude(
        cosmology_dict=cosmology_dict, 
        integration_options=integration_options, 
        redshift=redshift, 
        interpolation_settings=interpolation_settings, 
        k_user=None,
        multipoles_options=multipoles_options,
        simulation_settings=simulation_settings, 
        use_nonlinear_power_spectrum = 'linear',
        logger=logger,
        workdir=workdir,
        sim_outputdir=sim_outputdir,
        overwrite_settings=True)

    # Simulate the model for later use. Takes roughly 2 min per redshift, but only needs to be done once.
    # EXCEPT IF YOU CHANGE CLASS INPUT PARAMETERS AND THE CODE DOES NOT CHECK FOR THIS. If you do, then put resimulate=True
    resimulate=True
    prediction_calculator.simulate_projection(resimulate=resimulate, projection_type='gp', n_cores=args.n_cores)

    _, w_gp = prediction_calculator.load_projection(redshift=0.0, projection_type='gp')

    return np.asarray(w_gp)


def get_wVI_term(deltaV):

    prediction_calculator = fitIntrinsicAlignmentAmplitude(
        cosmology_dict=cosmology_dict, 
        integration_options=integration_options, 
        redshift=redshift, 
        interpolation_settings=interpolation_settings, 
        k_user=None,
        multipoles_options=multipoles_options,
        simulation_settings=simulation_settings, 
        use_nonlinear_power_spectrum = deltaV,
        logger=logger,
        workdir=workdir,
        sim_outputdir=sim_outputdir,
        overwrite_settings=True)

    # Simulate the model for later use. Takes roughly 2 min per redshift, but only needs to be done once.
    # EXCEPT IF YOU CHANGE CLASS INPUT PARAMETERS AND THE CODE DOES NOT CHECK FOR THIS. If you do, then put resimulate=True
    resimulate=True
    
    prediction_calculator.simulate_projection(resimulate=resimulate, projection_type='gp', n_cores=args.n_cores)

    _, w_gp = prediction_calculator.load_projection(redshift=0.0, projection_type='gp')

    return np.asarray(w_gp)




print('wVII')
wgg_1 = get_wVII_term()
np.save(sim_outputdir+'/wvII.npy', wgg_1)

# delta_v = np.load('/disks/shear16/herle/code/zobov_fiducial_void_delta_k_bin0_raw.npy')

# from scipy.signal import savgol_filter
# from scipy.interpolate import interp1d


# # ---------------------------------
# # work in log10(k) space
# # ---------------------------------


# logk = np.log10(k)


# # build uniform grid in log-space
# N_log = 20000   # resolution of smoothing grid (10–50k is good)

# logk_uniform = np.linspace(logk.min(), logk.max(), N_log)


# # interpolate raw signal onto uniform log grid
# interp_func = interp1d(logk, delta_v, kind="linear")

# delta_uniform = interp_func(logk_uniform)


# # ---------------------------------
# # Savitzky–Golay parameters (TUNE)
# # ---------------------------------
# window_length = 1001   # odd, measured in log-k samples
# polyorder = 3

# if window_length % 2 == 0:
#     window_length += 1


# delta_uniform_smooth = savgol_filter(
#     delta_uniform,
#     window_length,
#     polyorder
# )


# # ---------------------------------
# # interpolate back to original k
# # ---------------------------------
# interp_back = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")

# delta_v_smoothed = interp_back(logk)

# print('wVI')
# wgg_2 = get_wVI_term(delta_v_smoothed)
# np.save(sim_outputdir+'/zobov_fiducial_Rvbin0_wvI.npy', wgg_2)
