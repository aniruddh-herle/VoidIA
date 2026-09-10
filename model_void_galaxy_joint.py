import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt
import glob
import pandas as pd
from nautilus import Sampler, Prior  # <<< moved Prior import here
from multiprocessing import Pool
import h5py
import re

import argparse

import os
os.environ["OMP_NUM_THREADS"] = "1"


from getdist import plots, MCSamples

parser = argparse.ArgumentParser()
parser.add_argument('-box_size', type=float, default=1000, help='Size of the box.')
parser.add_argument('-n_cores', type=int, default=192, help='Number of CPU cores to use for multiprocessing.')
args = parser.parse_args()

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Create logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

import pyccl as ccl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import galaxy alignment predictor
# Add root directory to path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)
    

rootpath = Path('/disks/shear16/herle/code/GNAT')
if str(rootpath) not in os.sys.path:
    os.sys.path.append(str(rootpath))

from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum

import pandas as pd

# -------------------------------------------------
# Best-fit chi2
# -------------------------------------------------
def compute_chi2(logl, samples, corr):
    logL_max = np.max(logl)
    chi2_eff = -2 * logL_max
    Ndata = len(corr)
    Nparams_flat = samples.shape[1]
    Ndof = Ndata - Nparams_flat
    chi2_red = chi2_eff / Ndof
    return chi2_eff, chi2_red


def read_nautilus_samples(npz_filename):
    data = np.load(npz_filename)
    samples = data['samples']        # shape (Nsamples, Nparams)
    weights = data['weights']        # shape (Nsamples,)
    logl = data['logl']              # shape (Nsamples,)
    return samples, weights, logl


def read_flamingo_power_spectrum(file_path):
    with h5py.File(file_path, 'r') as f:
        z_keys = sorted(
            [key for key in f.keys() if re.match(r"z=\d+\.\d+", key)],
            key=lambda k: float(k.split('=')[1])
        )

        a_list, pk_list = [], []

        for zkey in z_keys:
            z = float(zkey.split('=')[1])
            a = 1.0 / (1.0 + z)
            k = f[f"{zkey}/k"][:]
            p = f[f"{zkey}/P(k)"][:]

            pk_list.append(p)
            a_list.append(a)

    a_array = np.array(a_list)
    P_array = np.array(pk_list).T

    sort_idx = np.argsort(a_array)
    a_array = a_array[sort_idx]
    P_array = P_array[:, sort_idx]

    return P_array, k, a_array

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
    n_s=0.967
)

names = ['bg','b_dummy']  
labels = ['b_1','b_d']

#############################################################
## get bg
#############################################################

# Define redshifts at which to calculate estimators
redshift_list = [0.0]

# Define scales at which to calculate estimators
rp_array_hMpc = np.geomspace(0.1, 200, 200)  # in Mpc/h
r_array_hMpc = np.geomspace(0.1, 200, 200)  # in Mpc/h

# Pi_max for projection integrals
pi_max_hMpc = 100.0  # in Mpc/h

h = 0.681

rp_array = rp_array_hMpc / h   # physical Mpc, for the predictor
r_array  = r_array_hMpc / h    # physical Mpc, for the predictor
pi_max   = pi_max_hMpc / h     # physical Mpc, for the predictor


# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(1e-5, 5e2, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 2, 1)).T
pk_input = np.zeros((len(k_array_single), 2, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

# Fill pk_input with g+ and gg power spectra
powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )

    # Initialise power spectra
    pkgp = powerSpectrumHandler.convert_pk_matter_to_pk_gplus(
        pk_matter=pkmm_ccl,
        cosmo=cosmo_ccl,
        redshift=z
    )
    pkgg = powerSpectrumHandler.convert_pk_matter_to_pk_gg(
        pk_matter=pkmm_ccl,
        cosmo=cosmo_ccl,
        redshift=z,
        )

    pk_input[:, 0, idx_z] = pkgp
    pk_input[:, 1, idx_z] = pkgg

# Then, we initialise the power spectra in the predictor
predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)
predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input,
    probetype_order = ['g+', 'gg'],
)

# The rest of the procedure is the same as above

predictor.calculate_estimators(
    rp_support=rp_array,
    beta_shape=0, # No RSD, for example in real space
    beta_density=0, # No RSD, for example in real space
    pi_max=pi_max,
)


# Retrieve projection estimators as arrays
w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
    rp_support=rp_array,
    r_support=r_array,
)

# Retrieve splines for projection estimators
# projection_splines is a list of lists of interp1d objects
# Outer list runs over redshifts, inner list over power spectrum types (g+, gg)
projection_splines = predictor.return_projection_splines()
# Example: evaluate w_gp and xi_gp for all redshifts
rp_grid_hMpc = np.geomspace(0.1, 200, 200) # in Mpc/h, from 0.1 to 200 Mpc/h
# r_grid = np.geomspace(0.1, 200, 200) # in Mpc/h, from 0.1 to 200 Mpc/h
rp_grid_Mpc  = rp_grid_hMpc / h  

for idx_z, z in enumerate(redshift_list):
    w_gg_spline = projection_splines[idx_z][1]  # gg projection spline
    w_gg_at_rp = w_gg_spline(rp_grid_Mpc) * h



redshift = '0.0'

# modes = ['FIDUCIAL', 'WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA']
modes = ['FIDUCIAL']


z = 0.0
kmin, kmax = 1e-5, 1e2

rp_values2v = np.logspace(-1, np.log10(900), 200)

# Define scales at which to calculate estimators

# rp_array_hMpc = rp_values2v          # rename for clarity — this IS the hMpc grid
# rp_array = rp_array_hMpc / h         # physical Mpc, for the predictor
# r_array = rp_array

# pi_max_hMpc = 100.0
# pi_max = pi_max_hMpc / h
pi_max = 100
rp_array = rp_values2v
r_array = rp_array

# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

# # Fill pk_input with g+ and gg power spectra
powerSpectrumHandler = powerSpectrum()

h = 0.681

ps_modes = ['L1_m9', 'fgas+2σ', 'fgas-2σ', 'fgas-4σ', 'fgas-8σ', 'M_-σ']


k_native = np.logspace(-5, 2, 1000001)   # h/Mpc

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo_ccl['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo_ccl, scale_factor)


# Radius bins
# r_bins = [(0, 10), (10, 20), (20, 30), (30, 50)]

# # Median void radii
# Rv_median = {
#     'FIDUCIAL':         {(0,10): 8.46, (10,20):14.72, (20,30):23.17, (30,50):32.50},
#     'WEAK_AGN':         {(0,10): 8.49, (10,20):14.65, (20,30):23.10, (30,50):32.15},
#     'STRONG_AGN':       {(0,10): 8.41, (10,20):14.70, (20,30):22.94, (30,50):32.04},
#     'STRONGER_AGN':     {(0,10): 8.44, (10,20):14.58, (20,30):23.10, (30,50):31.98},
#     'STRONGEST_AGN':    {(0,10): 8.37, (10,20):14.62, (20,30):22.94, (30,50):31.90},
#     'STRONG_SUPERNOVA': {(0,10): 8.59, (10,20):14.83, (20,30):23.19, (30,50):32.29},
# }


r_bins = [(10, 20)]

# Median void radii
Rv_median = {
    'FIDUCIAL':         {(10,20):14.72},
    'WEAK_AGN':         {(10,20):14.65},
    'STRONG_AGN':       {(10,20):14.70},
    'STRONGER_AGN':     {(10,20):14.58},
    'STRONGEST_AGN':    {(10,20):14.62},
    'STRONG_SUPERNOVA': {(10,20):14.83},
}

Rmin, Rmax = 10, 20



for mode, ps_mode in zip(modes, ps_modes):

    wgg_measured = pd.read_hdf(
        f'/disks/shear16/herle/correlations1/correlations_joint_revolver_{Rmin}_{Rmax}hMpc_L1000N1800_HYDRO_'+mode+'_'+redshift+'_joint3d.h5'
    )[['wgg_rp', 'wgg_xip', 'wgg_err']]
    cov_wgg = np.load(f'/disks/shear16/herle/correlations1/wgg_cov_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_L1000N1800_HYDRO_'+mode+'_'+redshift+'_joint3d.npy')
    


    min_rp_scale_wgg = 30.0
    max_rp_scale_wgg = 100.0

    wgg_rp = wgg_measured["wgg_rp"].values 

    mask_wgg = (wgg_rp >= min_rp_scale_wgg) & (wgg_rp <= max_rp_scale_wgg)

    wgg_measured = wgg_measured[mask_wgg].reset_index(drop=True)

    

    corr = wgg_measured["wgg_xip"].values
    rp = wgg_measured["wgg_rp"].values
    wgg_err = wgg_measured["wgg_err"].values

    # Apply the same cuts to the covariance matrix
    cov_wgg = cov_wgg[np.ix_(mask_wgg, mask_wgg)]
    inv_cov = np.linalg.pinv(cov_wgg)  # <<< cache inverse once



    def model_wgg_spec(p):

        # Galaxy biases   
        b_g = p[0]
        

        wgg = (b_g**2 * w_gg_at_rp) 
        return rp_grid_hMpc, wgg



    def log_likelihood(p):
        
        r_model, corr_model_wgg = model_wgg_spec(p)
        corr_model_wgg_interpol = np.interp(rp, r_model, corr_model_wgg)
        
        delta = (corr - corr_model_wgg_interpol)
        chisq = delta.dot(inv_cov.dot(delta.T))
        ll = -0.5 * chisq
        if np.isnan(ll):
            return -np.inf, chisq
        return ll, chisq

    prior_obj = Prior()  


    param_names = ['bg', 'b_dummy']
    prior_obj.add_parameter('bg', dist=(0, 3.0))
    prior_obj.add_parameter('b_dummy', dist=(0, 3.0))


    def loglikelihood_fn(p_dict):
        # p_dict keys follow prior_obj keys order
        p = np.array([p_dict[key] for key in param_names])
        logl, _ = log_likelihood(p)
        return logl


    n_cores = 80

    def run():
        n_live = 1000

        output_path = '/disks/shear16/herle/models/models/'

        filename = output_path + f'nautilus_chain_voids_hMpc_L1000N1800_HYDRO_'+mode+'_'+redshift+'_gg_joint3d.h5'
        
        sampler = Sampler(
            prior_obj,
            loglikelihood_fn,
            n_live=n_live,
            pool=n_cores if n_cores > 1 else None,
            filepath=filename,
            resume=False,
            n_networks=16,
        )
        
        sampler.run(verbose=True, discard_exploration=True)
        points, log_w, log_l = sampler.posterior()
        weights = np.exp(log_w)
        logz = sampler.log_z
        output_file = (output_path +
                    f'nautilus_chain_voids_hMpc_L1000N1800_HYDRO_'+mode+'_'+redshift+'_gg_joint3d.npz')

        np.savez(output_file,
                samples=points,
                logl=log_l,
                weights=weights,
                logz=logz)
        
        print(f"Saved results to {output_file}")



    run()

    filename = '/disks/shear16/herle/models/models/nautilus_chain_voids_hMpc_L1000N1800_HYDRO_'+mode+'_0.0_gg_joint3d.npz'
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

    rp_model, wgg_model = model_wgg_spec([bg])

    fig, axs = plt.subplots(figsize=(4, 4))

    axs.errorbar(rp, corr, yerr=wgg_err)
    axs.plot(rp_model, wgg_model)

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('rp')
    axs.set_ylabel('wgg')

    plt.tight_layout()
    plt.savefig('/disks/shear16/herle/code/plots/check_gnat_wgg_'+mode+'.png')

    #############################################################
    ## get bv
    #############################################################


    meas_file = (
        f'/disks/shear16/herle/correlations1/'
        f'correlations_joint_revolver_10_20hMpc_L1000N1800_HYDRO_{mode}_{redshift}_joint3d.h5'
    )

    df = pd.read_hdf(meas_file)

    # --------------------------
    # Measurements
    # --------------------------

    rp_xi_vv = df['xi_vv_r'].values
    print('rp_xi_vv: ', np.shape(rp_xi_vv))

    xi_vv = df['xi_vv'].values
    xi_vv_err = df['xi_vv_err'].values


    rp_xi_gg = df['xi_gg_r'].values
    xi_gg = df['xi_gg'].values
    xi_gg_err = df['xi_gg_err'].values

    mask = (xi_vv > 0) & (xi_gg > 0)

    rp = rp_xi_vv[mask]

    print('rp: ', np.shape(rp))
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


    mean_mask = (rp >= 30) & (rp <= 100)

    print(np.sum(mean_mask))

    bv_over_bg_mean = np.mean(ratio[mean_mask])
    N = np.sum(mean_mask)
    bv_over_bg_err = np.std(ratio[mean_mask], ddof=1) / np.sqrt(N)

    # Inputs
    bg_best = bg
    bg_err = bg_error

    ratio_best = bv_over_bg_mean
    ratio_err = bv_over_bg_err   # or use the standard error if that's what you want

    # Best-fit void bias
    bv_best = ratio_best * bg_best

    # Error propagation
    bv_err = bv_best * np.sqrt(
        (ratio_err / ratio_best)**2 +
        (bg_err / bg_best)**2
    )

    print(f"b_g = {bg_best:.3f} ± {bg_err:.3f}")
    print(f"b_v/b_g = {ratio_best:.3f} ± {ratio_err:.3f}")
    print(f"b_v = {bv_best:.3f} ± {bv_err:.3f}")

    #######################################################
    ## A1v and A2v
    #######################################################


    print("-----------------------------")
    print(mode)
    print("-----------------------------")

        
    flamingo_file_path = f'/disks/shear16/herle/power_spectra/{ps_mode}.hdf5'
    print(flamingo_file_path)
    P_flamingo, k_flamingo, a_flamingo = read_flamingo_power_spectrum(flamingo_file_path)

    k_switch = k_flamingo.min() 

    pk_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )

    pk = np.zeros_like(k_array_single)

    a_target = 1/(1+0.0)
    idx = np.argmin(np.abs(a_flamingo - a_target))

    mask_low = k_array_single < k_switch
    mask_high = ~mask_low

    pk[mask_low] = pk_ccl[mask_low]

    interp = interp1d(
        np.log(k_flamingo),
        np.log(P_flamingo[:, idx]),
        kind="linear",
        bounds_error=True
    )

    pk[mask_high] = np.exp(interp(np.log(k_array_single[mask_high])))



    # ============================================================
    # Smooth stitched power spectrum on uniform log-k grid
    # ============================================================

    logk = np.log(k_array_single)
    logpk = np.log(pk)

    # Create uniform log-k grid
    N_log = 20000
    logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

    # Interpolate log P(k) onto uniform log-k grid
    interp_func = interp1d(
        logk,
        logpk,
        kind="linear"
    )

    logpk_uniform = interp_func(logk_uniform)

    # Savitzky-Golay smoothing
    window_length = 1001
    polyorder = 3

    if window_length % 2 == 0:
        window_length += 1

    logpk_uniform_smooth = savgol_filter(
        logpk_uniform,
        window_length,
        polyorder
    )

    # Interpolate smoothed log P(k) back to original k grid
    interp_back = interp1d(
        logk_uniform,
        logpk_uniform_smooth,
        kind="linear"
    )

    logpk_smoothed = interp_back(logk)

    # Convert back
    pk_smoothed = np.exp(logpk_smoothed)
    # reshape to predictor format: (n_k, n_pk, n_redshifts)
    pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))

    for idx_z, z in enumerate(redshift_list):
        pk_input[:, 0, idx_z] = -np.pi**2*pk_smoothed

    # Then, we initialise the power spectra in the predictor
    predictor = galaxyAlignmentPredictor(
        cosmology=cosmo_ccl,
        redshift_list=redshift_list,
        logger=logger,
    )

    # 2 void term

    predictor.initialise_power_spectra(
        k_input=k_array,
        pk_input=pk_input,
        probetype_order = ['g+'],  # order 2 for g+, order 0 for gg
    )

    # The rest of the procedure is the same as above
    predictor.calculate_estimators(
        rp_support=rp_array,
        beta_shape=0, # No RSD, for example in real space
        beta_density=0, # No RSD, for example in real space
        pi_max=pi_max,
    )


    # Retrieve projection estimators as arrays
    w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
        rp_support=rp_array,
        r_support=r_array,
    )


    w_2v =  w_arr[:, 0, 0]

    bin_index_map = {(0,10): 0, (10,20): 1, (20,30): 2, (30,50): 3}

    for delta_bin, (Rmin, Rmax) in enumerate(r_bins):
        Rv = Rv_median[mode][(Rmin, Rmax)]

        lower_rp_VI = 6.0
        upper_rp_VI = 2.5 * Rv

        lower_rp_VII = 4.0 * Rv
        upper_rp_VII = 500.0

        # rest of your calculations

        # Map r_bin range to the correct file index
        
        bin_idx = bin_index_map[(Rmin, Rmax)]
        delta_v = np.load(f'/disks/shear16/herle/code/temp_data/zobov_{mode}_void_delta_k_bin{str(bin_idx)}_raw.npy')  # ← bin_idx

        # Reconstruct the k-grid delta_v was actually saved on
        # (must match the void script exactly: log10kmin=-5, log10kmax=2, N=1000001, then *h)
        
        k_void = k_native                    # Mpc^-1, matches delta_v's saved values

        assert delta_v.shape[0] == k_void.shape[0], \
            f"delta_v has {delta_v.shape[0]} points but k_void has {k_void.shape[0]}"

        # Interpolate delta_v(k) from its native grid onto k_array_single
        logk_void = np.log10(k_void)
        logk_target = np.log10(k_array_single)

        interp_to_target = interp1d(
            logk_void, delta_v, kind="linear",
            bounds_error=False, fill_value=0.0
        )
        delta_v_on_target = interp_to_target(logk_target)   # shape (1000,), matches k_array_single

        # --- now smooth on k_array_single's own log grid ---
        logk = logk_target

        N_log = 20000
        logk_uniform = np.linspace(logk.min(), logk.max(), N_log)

        interp_func = interp1d(logk, delta_v_on_target, kind="linear")
        delta_uniform = interp_func(logk_uniform)

        window_length = 1001
        polyorder = 3
        if window_length % 2 == 0:
            window_length += 1

        delta_uniform_smooth = savgol_filter(delta_uniform, window_length, polyorder)

        interp_back = interp1d(logk_uniform, delta_uniform_smooth, kind="linear")
        delta_v_smoothed = interp_back(logk)   # shape (1000,), exactly matches k_array_single

        # --- reshape to match pk_input convention: (n_k, n_pk, n_redshifts) ---
        pk_input_1v = np.zeros((len(k_array_single), 1, len(redshift_list)))
        for idx_z, z in enumerate(redshift_list):
            pk_input_1v[:, 0, idx_z] = -np.pi**2*delta_v_smoothed

        predictor.initialise_power_spectra(
            k_input=k_array,
            pk_input=pk_input_1v,
            probetype_order=['g+'],
        )

        predictor.calculate_estimators(
            rp_support=rp_array,
            beta_shape=0,
            beta_density=0,
            pi_max=pi_max,
        )

        w_arr, xi_arr = predictor.return_all_estimators_as_arrays(
            rp_support=rp_array,
            r_support=r_array,
        )

        w_1v = w_arr[:, 0, 0]


        wVII = w_2v.copy()
        wVII *= scaling
        #wVII *=h

        wVI = w_1v.copy()
        wVI *= scaling
        #wVI *= h

        # model_rp = rp_array * h
        model_rp = rp_array 


        model_mask_VI = (model_rp >= lower_rp_VI) & (model_rp <= upper_rp_VI)
        model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)


        redshift = '0.0'

        meas_file = (
            f'/disks/shear16/herle/correlations1/'
            f'correlations_joint_revolver_{Rmin}_{Rmax}hMpc_L1000N1800' +
            f'_HYDRO_{mode}_{redshift}_joint3d.h5'
        )

        wvp_JK_measured = np.load(
            f'/disks/shear16/herle/correlations1/'
            f'wvp_xip_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_L1000N1800_HYDRO_{mode}_{redshift}_joint3d.npy'
        )


        # -------------------------------------------------
        # read measurement
        # -------------------------------------------------
        df = pd.read_hdf(meas_file, key='a')

        rp_meas  = df['wvp_rp'].values
        wvp_meas = df['wvp_xip'].values


        # -------------------------------------------------
        # build masks (same logic as before)
        # -------------------------------------------------
        mask_VI  = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
        mask_VII = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)

        mask_total = mask_VI | mask_VII

        # -------------------------------------------------
        rp_cut = rp_meas[mask_total]
        corr   = wvp_meas[mask_total]

        wvp_JK_cut = wvp_JK_measured[:, mask_total]


        # -------------------------------------------------
        # JK mean + diffs  (same structure as wgg/wgp code)
        # -------------------------------------------------
        wvp_mean = np.mean(wvp_JK_cut, axis=0)

        wvp_diff = pd.DataFrame(
            wvp_JK_cut - wvp_mean,
            columns=rp_cut
        )
        # -------------------------------------------------
        # FULL jackknife covariance  (IDENTICAL formula)
        # -------------------------------------------------
        NPatches = wvp_JK_cut.shape[0]

        cov_mat = ((NPatches - 1) / NPatches) * np.sum(
            np.einsum('ij,ik->ijk', wvp_diff, wvp_diff),
            axis=0
        )

        cov_mat = pd.DataFrame(
            data=cov_mat,
            columns=rp_cut
        )

        cov_mat.set_index(rp_cut, inplace=True)

        cov_arr = cov_mat.values
        inv_cov = np.linalg.pinv(cov_arr)

        def model_wVI(p):

            A1_v = p[0]

            bg = np.random.normal(bg_best, bg_error)

            wVI_model = wVI[model_mask_VI].copy()
            wVI_model *= A1_v * bg
            return model_rp[model_mask_VI], wVI_model

        def model_wVII(p):

            A2_v = p[1]
            b_v = np.random.normal(bv_best, bv_err)

            wVII_model = wVII[model_mask_VII].copy()
            wVII_model *= A2_v*b_v

            return model_rp[model_mask_VII], wVII_model


        def log_likelihood(p):
            
            rp_model_VI, wVI_model   = model_wVI(p)      # returns only the VI part
            rp_model_VII, wVII_model = model_wVII(p)     # returns only the VII part

            # concatenate model pieces into one vector (matching data vector order)
            rp_model_total = np.concatenate([rp_model_VI, rp_model_VII])
            corr_model_total = np.concatenate([wVI_model, wVII_model])

            # interpolate model onto the measured rp vector
            corr_model_interpol = np.interp(rp_cut, rp_model_total, corr_model_total)

            delta = (corr - corr_model_interpol)
            chisq = delta.dot(inv_cov.dot(delta.T))
            ll = -0.5 * chisq
            if np.isnan(ll):
                return -np.inf, chisq
            return ll, chisq

        prior_obj = Prior()  

        param_names = ['A1_v', 'A2_v']
        prior_obj.add_parameter('A1_v', dist=(-10.0, 10.0))
        prior_obj.add_parameter('A2_v', dist=(-10.0, 10.0))



        def loglikelihood_fn(p_dict):
            # p_dict keys follow prior_obj keys order
            p = np.array([p_dict[key] for key in param_names])
            logl, _ = log_likelihood(p)
            return logl


        box_size = args.box_size
        print("box size: ", box_size)
        

        def run():
            n_live = 3000

            output_path = '/disks/shear16/herle/models/models/'
        
            filename = output_path + f"nautilus_chain_revolver_{mode}_voids_{box_size}_Rv{Rmin}_{Rmax}hMpc_pimax100_0.0_JK_flamingo_ps_joint3d_sampledb_v2.h5"
            
            sampler = Sampler(
                prior_obj,
                loglikelihood_fn,
                n_live=n_live,
                pool=n_cores if n_cores > 1 else None,
                filepath=filename,
                resume=False,
                n_networks=16,
            )
            
            sampler.run(verbose=True, discard_exploration=True)
            points, log_w, log_l = sampler.posterior()
            weights = np.exp(log_w)
            logz = sampler.log_z
            output_file = (output_path +
                        f"nautilus_chain_revolver_{mode}_voids_{box_size}_Rv{Rmin}_{Rmax}hMpc_pimax100_0.0_JK_flamingo_ps_joint3d_sampledb_v2.npz")

            np.savez(output_file,
                    samples=points,
                    logl=log_l,
                    weights=weights,
                    logz=logz)
            
            print(f"Saved results to {output_file}")


        run()

        names = ['A1_v','A2_v']  
        labels = ['A1_v','A2_v']

        filename = f'/disks/shear16/herle/models/models/nautilus_chain_revolver_{mode}_voids_{box_size}_Rv{Rmin}_{Rmax}hMpc_pimax100_0.0_JK_flamingo_ps_joint3d_sampledb_v2.npz'
        samples, weights, logl = read_nautilus_samples(filename)
        getdist = MCSamples(samples=samples, 
                                    weights=weights,
                                    names=names, 
                                    labels=labels, 
                                    ignore_rows=0)


        d = getdist.get1DDensity('A1_v')
        xyind = np.where(d.P == np.amax(d.P))
        A1v = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)

        lower_error = errors_density[0]
        upper_error = errors_density[1]

        A1v_error = (upper_error - lower_error) / 2.0

        d = getdist.get1DDensity('A2_v')
        xyind = np.where(d.P == np.amax(d.P))
        A2v = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)

        lower_error = errors_density[0]
        upper_error = errors_density[1]

        A2v_error = (upper_error - lower_error) / 2.0

        
        chi2_eff, chi2_red = compute_chi2(logl, samples, corr)
        print(f"[{mode}] Reduced Chi2: ", chi2_red)

        # -------------------------------------------------
        # Best-fit amplitudes (marginal getdist mode)
        # -------------------------------------------------
        A1_best = A1v
        A2_best = A2v

        print(f"[{mode}] A1_v best fit: ", A1_best)
        print(f"[{mode}] A2_v best fit: ", A2_best)

        wVI_bestfit  = wVI  * A1_best * bg
        wVII_bestfit = wVII * A2_best * bv_best

        # -------------------------------------------------
        # Posterior predictive draws (for uncertainty bands)
        # -------------------------------------------------
        Nsamp = 500

        weights_norm = weights / np.sum(weights)
        inds = np.random.choice(len(samples), size=Nsamp, p=weights_norm)
        posterior_draws = samples[inds]

        A1_draws = posterior_draws[:, 0]
        A2_draws = posterior_draws[:, 1]

        # bg/bv carry their own independent uncertainty (from the earlier stages),
        # so we fold that in here too for a fully-propagated band
        bg_draws = np.random.normal(bg, bg_error, size=Nsamp)
        bv_draws = np.random.normal(bv_best, bv_err, size=Nsamp)

        wVI_ens  = []
        wVII_ens = []

        for A1, A2, bg_d, bv_d in zip(A1_draws, A2_draws, bg_draws, bv_draws):
            wVI_ens.append(wVI * A1 * bg_d)
            wVII_ens.append(wVII * A2 * bv_d)

        wVI_ens  = np.array(wVI_ens)
        wVII_ens = np.array(wVII_ens)

        p16_VI, p50_VI, p84_VI    = np.percentile(wVI_ens, [16, 50, 84], axis=0)
        p16_VII, p50_VII, p84_VII = np.percentile(wVII_ens, [16, 50, 84], axis=0)

        # -------------------------------------------------
        # Data at the cut used in the fit (VI ∪ VII region)
        # -------------------------------------------------
        err_cut = np.sqrt(np.diag(cov_arr))

        # split the cut points back into their VI/VII sub-masks, for interpolation below
        mask_data_VI  = mask_VI[mask_total]
        mask_data_VII = mask_VII[mask_total]

        color = 'k'
        label = mode

        # -------------------------------------------------
        # Main panel
        # -------------------------------------------------
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1,
            figsize=(14, 14),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
        )

        ax_main.plot(
            model_rp[model_mask_VII],
            model_rp[model_mask_VII] * wVII_bestfit[model_mask_VII],
            color=color
        )
        ax_main.plot(
            model_rp[model_mask_VI],
            model_rp[model_mask_VI] * wVI_bestfit[model_mask_VI],
            color=color
        )

        ax_main.errorbar(
            rp_cut,
            rp_cut * corr,
            yerr=rp_cut * err_cut,
            fmt='o',
            capsize=3,
            color=color,
            label=label
        )

        ax_main.fill_between(
            model_rp[model_mask_VI],
            model_rp[model_mask_VI] * p16_VI[model_mask_VI],
            model_rp[model_mask_VI] * p84_VI[model_mask_VI],
            alpha=0.25,
            color=color
        )

        ax_main.fill_between(
            model_rp[model_mask_VII],
            model_rp[model_mask_VII] * p16_VII[model_mask_VII],
            model_rp[model_mask_VII] * p84_VII[model_mask_VII],
            alpha=0.25,
            color=color
        )

        # -------------------------------------------------
        # Residuals
        # -------------------------------------------------
        model_at_data = np.full_like(rp_cut, np.nan)

        model_at_data[mask_data_VI] = np.interp(
            rp_cut[mask_data_VI],
            model_rp[model_mask_VI],
            wVI_bestfit[model_mask_VI]
        )

        model_at_data[mask_data_VII] = np.interp(
            rp_cut[mask_data_VII],
            model_rp[model_mask_VII],
            wVII_bestfit[model_mask_VII]
        )

        residual = 100.0 * (corr - model_at_data) / corr
        frac_err = 100.0 * err_cut / np.abs(corr)

        ax_res.errorbar(
            rp_cut,
            residual,
            yerr=frac_err,
            fmt='o',
            capsize=3,
            color=color
        )

        # shaded 1-void/2-void regions
        ax_main.axvspan(lower_rp_VI, upper_rp_VI, color='grey', alpha=0.08, zorder=0)
        ax_main.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)

        ax_res.axvspan(lower_rp_VI, upper_rp_VI, color='grey', alpha=0.08, zorder=0)
        ax_res.axvspan(lower_rp_VII, upper_rp_VII, color='grey', alpha=0.08, zorder=0)

        ax_main.axhline(0, color='k', ls=':')
        ax_main.set_xscale('log')
        ax_main.set_xlim(1, 600)
        ax_main.set_ylabel(r'$r_p w_{\mathrm{V}+}\,[\mathrm{Mpc}^2/h^2]$')
        ax_main.legend(frameon=False)

        ax_res.axhline(0, color='k', ls=':')
        ax_res.set_xscale('log')
        ax_res.set_ylim(-200, 200)
        ax_res.set_xlabel(r'$r_p$ [Mpc/h]')
        ax_res.set_ylabel(r'$(\mathrm{Data}-\mathrm{Model})/\mathrm{Data}$ [%]')

        plt.tight_layout()
        plt.savefig('/disks/shear16/herle/code/plots/check_A1_A2_fits_v2.png')