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



parser = argparse.ArgumentParser()
parser.add_argument('-box_size', type=float, default=1000, help='Size of the box.')
parser.add_argument('-ISR_lower_bound', type=float, default=6, help='Lower rp of ISR')
parser.add_argument('-ISR_upper_bound', type=float, default=35, help='Upper rp of ISR')
parser.add_argument('-LSR_lower_bound', type=float, default=60, help='Lower rp of LSR')
parser.add_argument('-LSR_upper_bound', type=float, default=500, help='Upper rp of LSR')
parser.add_argument('-n_cores', type=int, default=80, help='Number of CPU cores to use for multiprocessing.')
args = parser.parse_args()

void_finder = 'zobov'  # 'disperse' or 'zobov'

# base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0/'
base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0/'

h = 0.681
# scaling_wvII = 1.0/(np.pi**2)
# cosmo = ccl.Cosmology(Omega_c=0.306-0.0486, Omega_b=0.0486, h=0.681, sigma8 = 0.807, n_s=0.967)
# C1rhocrit = 0.0134
# z_i = 0.0
# scale_factor = 1.0/(1.0 + z_i)
# scaling = C1rhocrit*cosmo['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo, scale_factor)
cosmo = ccl.Cosmology(Omega_c=0.306-0.0486, Omega_b=0.0486, h=0.681, sigma8 = 0.807, n_s=0.967)
C1rhocrit = 0.0134
z_i = 0.0
scale_factor = 1.0/(1.0 + z_i)
scaling = C1rhocrit*cosmo['Omega_m']/(np.pi**2)/ccl.growth_factor(cosmo, scale_factor)

wVII = np.load(base_folder+'/wvII.npy')
wVII *= scaling
#wVII /=h



wVI = np.load(base_folder+f'/zobov_fiducial_Rvbin0_wvI.npy')

wVI *= scaling
#wVI /= h

model_rp = np.loadtxt(base_folder+'/w_gp_z0.0.txt')[:, 0] 
print("Model rp range: ", model_rp.min(), model_rp.max())
#model_rp = model_rp /h  # in Mpc

Rv = 15.0
# masks
lower_rp_VI = args.ISR_lower_bound
# upper_rp_VI = args.ISR_upper_bound
upper_rp_VI = 2.5*Rv


# lower_rp_VII = args.LSR_lower_bound
lower_rp_VII = 4.0*Rv
upper_rp_VII = args.LSR_upper_bound

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
    f'_HYDRO_FIDUCIAL_' + redshift + '.h5'
)

wvp_JK_measured = np.load(
    f'/disks/shear16/herle/correlations1/'
    f'wgp_xip_jk_voids_revolver_10_20Mpc_{run}_HYDRO_FIDUCIAL_{redshift}.npy'
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
# cut measurement  (THIS is your data vector)
# -------------------------------------------------
rp_cut = rp_meas[mask_total]
corr   = wvp_meas[mask_total]


# -------------------------------------------------
# cut JK realizations  (CRITICAL)
# shape -> (Npatches, Nbins_cut)
# -------------------------------------------------
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


# -------------------------------------------------
# cache arrays for likelihood (same as old script)
# -------------------------------------------------
cov_arr = cov_mat.values
inv_cov = np.linalg.pinv(cov_arr)



# # meas_file_voxel = '/disks/shear16/herle/correlations1/correlations_voids_voxel_10_30Mpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.h5'
# # df_voxel = pd.read_hdf(meas_file_voxel, key='a')
# # rp_meas_voxel = df_voxel['wgp_rp'].values
# # wvp_meas_voxel = df_voxel['wgp_xip'].values
# # err_meas_voxel = df_voxel['wgp_std_from_cov'].values

# # meas_file_zobov = '/disks/shear16/herle/correlations1/correlations_voids_zobov_10_30Mpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.h5'
# # df_zobov = pd.read_hdf(meas_file_zobov, key='a')
# # rp_meas_zobov = df_zobov['wgp_rp'].values
# # wvp_meas_zobov = df_zobov['wgp_xip'].values
# # err_meas_zobov = df_zobov['wgp_std_from_cov'].values

def model_wVI(p):

    A1_v = p[0]

    wVI_model = wVI[model_mask_VI].copy()
    wVI_model *= A1_v

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
   
    filename = output_path + f"nautilus_chain_{void_finder}_fiducial_voids_{box_size}_Rv10_20Mpc_pimax100_0.0.h5"
    
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
                   f"nautilus_chain_{void_finder}_fiducial_voids_{box_size}_Rv10_20Mpc_pimax100_0.0.npz")

    np.savez(output_file,
             samples=points,
             logl=log_l,
             weights=weights,
             logz=logz)
    
    print(f"Saved results to {output_file}")


if __name__ == '__main__':
    run()
