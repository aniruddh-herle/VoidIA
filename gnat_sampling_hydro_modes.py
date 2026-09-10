import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt
import glob
import pandas as pd
from nautilus import Sampler, Prior  # <<< moved Prior import here
from multiprocessing import Pool


import argparse

import os
os.environ["OMP_NUM_THREADS"] = "1"


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

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306-0.0486, 
    Omega_b=0.0486,  
    h=0.681, 
    sigma8 = 0.807, 
    n_s=0.967
)

# Define redshifts at which to calculate estimators
redshift_list = np.array([0])


kmin, kmax = 1e-5, 1e2

rp_values2v = np.logspace(-1, np.log10(900), 200)

# Define scales at which to calculate estimators
rp_array = rp_values2v
r_array = rp_array

# Pi_max for projection integrals
pi_max = 100.0  # in Mpc


# We need to create k_array and pk_array as np.ndarrays in the format (n_k, n_pk, n_redshifts). 
# We use here the vanilla g+ and gg obtained from the 3d matter power spectrum
k_array_single = np.geomspace(kmin, kmax, 1000)  # shape (n_k,)
k_array = np.tile(k_array_single, (len(redshift_list), 1, 1)).T
pk_input = np.zeros((len(k_array_single), 1, len(redshift_list)))  # shape (n_k, n_pk, n_redshifts)

# Fill pk_input with g+ and gg power spectra
powerSpectrumHandler = powerSpectrum()
for idx_z, z in enumerate(redshift_list):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )

    pk_input[:, 0, idx_z] = -np.pi**2*pkmm_ccl

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

# 1 void term

h = 0.681

# modes = ['FIDUCIAL', 'WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA']
modes = ['FIDUCIAL']

k_native = np.logspace(-5, 2, 1000001)   # h/Mpc

C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo_ccl['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo_ccl, scale_factor)

Rv = 15.0
# masks
lower_rp_VI = 6.0
upper_rp_VI = 2.5*Rv

lower_rp_VII = 4.0*Rv
upper_rp_VII = 500.0


for i in range(len(modes)):

    print("-----------------------------")
    print(modes[i])
    print("-----------------------------")

    delta_v = np.load(f'/disks/shear16/herle/code/zobov_{modes[i]}_void_delta_k_bin0_raw.npy')

    # Reconstruct the k-grid delta_v was actually saved on
    # (must match the void script exactly: log10kmin=-5, log10kmax=2, N=1000001, then *h)
    
    k_void = k_native * h                     # Mpc^-1, matches delta_v's saved values

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

    window_length = 3001
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
    wVII *=h

    wVI = w_1v.copy()
    wVI *= scaling
    wVI *= h

    model_rp = rp_array * h

    model_mask_VI = (model_rp >= lower_rp_VI) & (model_rp <= upper_rp_VI)
    model_mask_VII = (model_rp >= lower_rp_VII) & (model_rp <= upper_rp_VII)


    run = 'L1000N1800'
    redshift = '0.0'

    # -------------------------------------------------
    # load measurement + JK realizations
    # -------------------------------------------------
    meas_file = (
        f'/disks/shear16/herle/correlations1/'
        f'correlations_revolver_10_20Mpc_' + run +
        f'_HYDRO_{modes[i]}_' + redshift + '_JK.h5'
    )


    wvp_JK_measured = np.load(
        f'/disks/shear16/herle/correlations1/'
        f'wgp_xip_jk_voids_revolver_10_20Mpc_{run}_HYDRO_{modes[i]}_{redshift}_JK.npy'
    )


    # -------------------------------------------------
    # read measurement
    # -------------------------------------------------
    df = pd.read_hdf(meas_file, key='a')

    rp_meas  = df['wgp_rp'].values
    wvp_meas = df['wgp_xip'].values


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

    # --- plots per bin (unchanged) ---
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(np.abs(cov_mat)), origin='lower')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'/disks/shear16/herle/code/plots/cov_void_corr_{modes[i]}.png')
    plt.close()

    # -------------------------------------------------
    # cache arrays for likelihood (same as old script)
    # -------------------------------------------------
    cov_arr = cov_mat.values
    inv_cov = np.linalg.pinv(cov_arr)



    # meas_file = (
    #     f'/disks/shear16/herle/correlations1/'
    #     f'correlations_revolver_10_20Mpc_{run}_HYDRO_{modes[i]}_{redshift}.h5'
    # )

    # df = pd.read_hdf(meas_file, key='a')

    # rp_meas  = df['wgp_rp'].values
    # wvp_meas = df['wgp_xip'].values

    # mask_VI  = (rp_meas >= lower_rp_VI)  & (rp_meas <= upper_rp_VI)
    # mask_VII = (rp_meas >= lower_rp_VII) & (rp_meas <= upper_rp_VII)

    # mask_total = mask_VI | mask_VII

    # rp_cut = rp_meas[mask_total]

    # corr = wvp_meas[mask_total]

    # # -------------------------------------------------
    # # Load bootstrap realizations
    # # -------------------------------------------------

    # boot_files = sorted(
    #     glob.glob(
    #         f'/disks/shear16/herle/correlations1/bootstrap1/'
    #         f'correlations_revolver_10_20Mpc_{run}_HYDRO_{modes[i]}_{redshift}_bootstrap*.h5'
    #     )
    # )

    # print(f"Found {len(boot_files)} bootstrap realizations.")

    # bootstrap_vectors = []

    # for f in boot_files:

    #     df_boot = pd.read_hdf(f, key='a')

    #     bootstrap_vectors.append(
    #         df_boot['wgp_xip'].values
    #     )

    # bootstrap_vectors = np.asarray(bootstrap_vectors)

    # print("Bootstrap array shape:", bootstrap_vectors.shape)


    # # -------------------------------------------------
    # # Apply same rp cut as measurement
    # # -------------------------------------------------

    # bootstrap_cut = bootstrap_vectors[:, mask_total]


    # # -------------------------------------------------
    # # Bootstrap covariance
    # # -------------------------------------------------

    # K = bootstrap_cut.shape[0]

    # bootstrap_mean = np.mean(
    #     bootstrap_cut,
    #     axis=0
    # )

    # delta_boot = bootstrap_cut - bootstrap_mean

    # cov_arr = (
    #     delta_boot.T @ delta_boot
    # ) / (K - 1)


    # # -------------------------------------------------
    # # Match previous variable names
    # # -------------------------------------------------

    # cov_mat = pd.DataFrame(
    #     data=cov_arr,
    #     columns=rp_cut,
    #     index=rp_cut
    # )


    # # -------------------------------------------------
    # # Plot covariance (optional)
    # # -------------------------------------------------

    # fig, ax = plt.subplots()

    # im = ax.imshow(
    #     np.log10(np.abs(cov_arr)),
    #     origin='lower'
    # )

    # fig.colorbar(im, ax=ax)

    # plt.tight_layout()

    # plt.savefig(
    #     f'/disks/shear16/herle/code/plots/cov_void_corr_bootstrap_{modes[i]}.png'
    # )

    # plt.close()


    # # -------------------------------------------------
    # # Inverse covariance (same variable names)
    # # -------------------------------------------------

    # inv_cov = np.linalg.pinv(cov_arr)

    def model_wVI(p):

        A1_v = p[0]

        wVI_model = wVI[model_mask_VI].copy()
        wVI_model *= A1_v * 1.530

        return model_rp[model_mask_VI], wVI_model

    def model_wVII(p):

        A1_v = p[0]
        b_v = p[1]

        wVII_model = wVII[model_mask_VII].copy()
        wVII_model *= A1_v*b_v

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

    param_names = ['A1_v', 'b_v']
    prior_obj.add_parameter('A1_v', dist=(-10.0, 10.0))
    prior_obj.add_parameter('b_v', dist=(-10.0, 10.0))



    def loglikelihood_fn(p_dict):
        # p_dict keys follow prior_obj keys order
        p = np.array([p_dict[key] for key in param_names])
        logl, _ = log_likelihood(p)
        return logl


    box_size = args.box_size
    n_cores = args.n_cores

    def run():
        n_live = 3000

        output_path = '/disks/shear16/herle/models/models/'
    
        filename = output_path + f"nautilus_chain_revolver_{modes[i]}_voids_{box_size}_Rv10_20Mpc_pimax100_0.0_JK125_bgtest.h5"
        
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
                    f"nautilus_chain_revolver_{modes[i]}_voids_{box_size}_Rv10_20Mpc_pimax100_0.0_JK125_bgtest.npz")

        np.savez(output_file,
                samples=points,
                logl=log_l,
                weights=weights,
                logz=logz)
        
        print(f"Saved results to {output_file}")


    if __name__ == '__main__':
        run()
