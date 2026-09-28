import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.ccllib as lib
import pyccl.nl_pt as pt
import glob
import pandas as pd
from nautilus import Sampler, Prior
from multiprocessing import Pool
from mcfit import P2xi

import argparse

import os
os.environ["OMP_NUM_THREADS"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument(
    '-box_size',
    type=float,
    default=1000,
    help='Size of the box.'
)
parser.add_argument(
    '-n_cores',
    type=int,
    default=192,
    help='Number of CPU cores to use for multiprocessing.'
)
args = parser.parse_args()


from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Create logger
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


import pyccl as ccl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


# ============================================================
# COSMOLOGY
# ============================================================

cosmo_ccl = ccl.Cosmology(
    Omega_c=0.306 - 0.0486,
    Omega_b=0.0486,
    h=0.681,
    sigma8=0.807,
    n_s=0.967
)


# ============================================================
# DEFINE REDSHIFT / k / r ARRAYS
# ============================================================

redshift_list = np.array([0])

kmin, kmax = 1e-5, 1e2

rp_values2v = np.logspace(
    -1,
    np.log10(900),
    200
)

# Define scales at which to calculate estimators
rp_array = rp_values2v
r_array = rp_array

# Pi_max for projection integrals
pi_max = 100.0  # Mpc/h


# ============================================================
# GNAT: 2-VOID TERM
# ============================================================

k_array_single = np.geomspace(
    kmin,
    kmax,
    1000
)

k_array = np.tile(
    k_array_single,
    (len(redshift_list), 1, 1)
).T

pk_input = np.zeros(
    (
        len(k_array_single),
        1,
        len(redshift_list)
    )
)


powerSpectrumHandler = powerSpectrum()

for idx_z, z in enumerate(redshift_list):

    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmo_ccl,
        k_input=k_array_single,
        redshift=z,
        pk_type='nonlinear_matter'
    )

    pk_input[:, 0, idx_z] = -np.pi**2 * pkmm_ccl


predictor = galaxyAlignmentPredictor(
    cosmology=cosmo_ccl,
    redshift_list=redshift_list,
    logger=logger,
)


# Initialise 2-void power spectrum
predictor.initialise_power_spectra(
    k_input=k_array,
    pk_input=pk_input,
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


w_2v = w_arr[:, 0, 0]


# ============================================================
# BASIC SETTINGS
# ============================================================

h = 0.681

modes = ['FIDUCIAL']


k_native = np.logspace(
    -5,
    2,
    1000001
)   # h/Mpc


C1rhocrit = 0.0134

z_i = 0.0

scale_factor = 1.0 / (1.0 + z_i)

scaling = (
    C1rhocrit
    * cosmo_ccl['Omega_m']
    / (np.pi**2)
    / ccl.growth_factor(
        cosmo_ccl,
        scale_factor
    )
)


# ============================================================
# VOID RADIUS BINS
# ============================================================

void_bins = [
    (0, 10),
    (10, 20),
    (20, 30)
]


Rv_median = {
    'FIDUCIAL': [
        8.46,
        14.72,
        23.17
    ]
}


bin_labels = [
    '0_10',
    '10_20',
    '20_30'
]


delta_v_bin_labels = [
    'bin0',
    'bin1',
    'bin2'
]

# void_bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
# Rv_median = {
#     'FIDUCIAL_2p8':    [8.28, 12.49, 17.17, 21.94, 26.76, 31.92]
# }

# bin_labels = ['0_10', '10_15', '15_20', '20_25', '25_30', '30_50']
# delta_v_bin_labels = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5']



# ============================================================
# GETDIST / HELPER FUNCTIONS
# ============================================================

from getdist import plots, MCSamples


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

    samples = data['samples']

    weights = data['weights']

    logl = data['logl']

    return samples, weights, logl


names = [
    'A1_v',
    'A2_v',
    'b_v'
]


labels = [
    'A1_v',
    'A2_v',
    'b_v'
]


# ============================================================
# PRECOMPUTE xi_mm ONCE
# ============================================================

z_target = 0.0

a_target = 1.0 / (1.0 + z_target)


k_hmpc = np.geomspace(
    1e-5,
    1e3,
    8192
)


pk_ccl_mm = ccl.nonlin_matter_power(
    cosmo_ccl,
    k_hmpc * h,
    a_target
)


pk_hmpc = pk_ccl_mm * h**3


r_mm, xi_mm = P2xi(
    k_hmpc,
    lowring=True
)(
    pk_hmpc,
    extrap=True
)


keep = (
    (r_mm >= 0.1)
    &
    (r_mm <= 300.0)
)


r_mm = r_mm[keep]

xi_mm = xi_mm[keep]


xi_mm_interp = interp1d(
    r_mm,
    xi_mm,
    kind='cubic',
    bounds_error=False,
    fill_value=0.0
)


# ============================================================
# MAIN LOOP OVER MODES
# ============================================================

for i in range(len(modes)):

    print()
    print("-----------------------------")
    print(modes[i])
    print("-----------------------------")


    # ========================================================
    # LOOP OVER VOID RADIUS BINS
    # ========================================================

    for bin_idx, (
        void_bin,
        bin_label,
        delta_bin_label
    ) in enumerate(
        zip(
            void_bins,
            bin_labels,
            delta_v_bin_labels
        )
    ):

        # ----------------------------------------------------
        # Void radius for this bin
        # ----------------------------------------------------

        Rv = Rv_median[modes[i]][bin_idx]


        print()
        print(
            f"Processing {modes[i]} "
            f"Rv bin {void_bin[0]}-{void_bin[1]} Mpc/h"
        )

        print(
            f"Median Rv = {Rv:.2f} Mpc/h"
        )


        # ----------------------------------------------------
        # Fitting ranges
        # ----------------------------------------------------

        lower_rp_VI = 6.0

        upper_rp_VI = 2.5 * Rv

        lower_rp_VII = 4.0 * Rv

        upper_rp_VII = 500.0


        print(
            f"VI fitting range  = "
            f"{lower_rp_VI:.2f} - {upper_rp_VI:.2f} Mpc/h"
        )

        print(
            f"VII fitting range = "
            f"{lower_rp_VII:.2f} - {upper_rp_VII:.2f} Mpc/h"
        )


        # ====================================================
        # 1-VOID TERM: LOAD AND SMOOTH delta_v
        # ====================================================

        # delta_v = np.load(
        #     f'/disks/shear16/herle/code/temp_data/'
        #     f'zobov_{modes[i]}_void_delta_k_{delta_bin_label}'
        #     f'_raw_particles.npy'
        # )

        delta_v = np.load(
            f'/disks/shear16/herle/code/temp_data/'
            f'zobov_{modes[i]}_hres_void_Rv{delta_bin_label}'
            f'_particles_rmax200.npy'
        )


        k_void = k_native


        assert delta_v.shape[0] == k_void.shape[0], (
            f"delta_v has {delta_v.shape[0]} points "
            f"but k_void has {k_void.shape[0]}"
        )


        logk_void = np.log10(k_void)

        logk_target = np.log10(k_array_single)


        interp_to_target = interp1d(
            logk_void,
            delta_v,
            kind="linear",
            bounds_error=False,
            fill_value=0.0
        )


        delta_v_on_target = interp_to_target(
            logk_target
        )


        logk = logk_target


        N_log = 20000


        logk_uniform = np.linspace(
            logk.min(),
            logk.max(),
            N_log
        )


        interp_func = interp1d(
            logk,
            delta_v_on_target,
            kind="linear"
        )


        delta_uniform = interp_func(
            logk_uniform
        )


        window_length = 3001

        polyorder = 3


        if window_length % 2 == 0:
            window_length += 1


        delta_uniform_smooth = savgol_filter(
            delta_uniform,
            window_length,
            polyorder
        )


        interp_back = interp1d(
            logk_uniform,
            delta_uniform_smooth,
            kind="linear"
        )


        delta_v_smoothed = interp_back(
            logk
        )


        # ----------------------------------------------------
        # Feed smoothed 1-void term into GNAT
        # ----------------------------------------------------

        pk_input_1v = np.zeros(
            (
                len(k_array_single),
                1,
                len(redshift_list)
            )
        )


        for idx_z, z in enumerate(redshift_list):

            pk_input_1v[:, 0, idx_z] = (
                -np.pi**2 * delta_v_smoothed
            )


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


        # ====================================================
        # MODEL TEMPLATES
        # ====================================================

        wVII = w_2v.copy() * scaling

        wVI = w_1v.copy() * scaling


        model_rp = rp_array


        model_mask_VI = (
            (model_rp >= lower_rp_VI)
            &
            (model_rp <= upper_rp_VI)
        )


        model_mask_VII = (
            (model_rp >= lower_rp_VII)
            &
            (model_rp <= upper_rp_VII)
        )


        # ====================================================
        # MEASUREMENT FILES
        # ====================================================

        run = 'L1000N1800'
        #run = 'L2800N5040'

        redshift = '0.0'


        # ----------------------------------------------------
        # Measured joint correlation
        # ----------------------------------------------------

        meas_file = (
            f'/disks/shear16/herle/correlations1/'
            f'correlations_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical.h5'
        )


        # ----------------------------------------------------
        # wvp JK samples
        # ----------------------------------------------------

        wvp_JK_measured = np.load(
            f'/disks/shear16/herle/correlations1/'
            f'wvp_xip_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical.npy'
        )


        # ----------------------------------------------------
        # xi_vm JK samples
        # ----------------------------------------------------

        xi_vm_JK_measured = np.load(
            f'/disks/shear16/herle/correlations1/'
            f'xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
            f'_HYDRO_{modes[i]}_{redshift}_env_dep_analytical.npy'
        )

        # meas_file = (
        #     f'/disks/shear16/herle/correlations1/'
        #     f'correlations_joint_revolver_{bin_label}hMpc_{run}'
        #     f'_HYDRO_FIDUCIAL_{redshift}_env_dep_analytical.h5'
        # )

        # wvp_JK_measured = np.load(
        #     f'/disks/shear16/herle/correlations1/'
        #     f'wvp_xip_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
        #     f'_HYDRO_FIDUCIAL_{redshift}_env_dep_analytical.npy'
        # )

        # xi_vm_JK_measured = np.load(
        #     f'/disks/shear16/herle/correlations1/'
        #     f'xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_{run}'
        #     f'_HYDRO_FIDUCIAL_{redshift}_env_dep_analytical.npy'
        # )


        df = pd.read_hdf(
            meas_file,
            key='a'
        )


        # ====================================================
        # READ MEASUREMENTS
        # ====================================================

        rp_meas = df['wvp_rp'].values

        wvp_meas = df['wvp_xip'].values


        rp_xi_vm = df['xi_vm_r'].values

        xi_vm = df['xi_vm'].values

        xi_vm_err = df['xi_vm_err'].values


        # ====================================================
        # BUILD WVP MASKS
        # ====================================================

        mask_VI = (
            (rp_meas >= lower_rp_VI)
            &
            (rp_meas <= upper_rp_VI)
        )


        mask_VII = (
            (rp_meas >= lower_rp_VII)
            &
            (rp_meas <= upper_rp_VII)
        )


        mask_total = (
            mask_VI
            |
            mask_VII
        )


        rp_cut = rp_meas[mask_total]

        corr = wvp_meas[mask_total]


        # ====================================================
        # xi_vm FITTING RANGE
        # ====================================================

        xi_rmin = 50.0

        xi_rmax = 200


        mask_xi = (
            (rp_xi_vm >= xi_rmin)
            &
            (rp_xi_vm <= xi_rmax)
        )


        rp_xi_cut = rp_xi_vm[mask_xi]

        xi_vm_cut = xi_vm[mask_xi]

        xi_vm_err_cut = xi_vm_err[mask_xi]


        # ====================================================
        # CUT JK REALIZATIONS
        # ====================================================

        wvp_JK_cut = (
            wvp_JK_measured[:, mask_total]
        )


        xi_vm_JK_cut = (
            xi_vm_JK_measured[:, mask_xi]
        )


        # ====================================================
        # JOINT COVARIANCE
        # ====================================================

        wvp_mean = np.mean(
            wvp_JK_cut,
            axis=0
        )


        xi_vm_mean = np.mean(
            xi_vm_JK_cut,
            axis=0
        )


        wvp_diff = pd.DataFrame(
            wvp_JK_cut - wvp_mean,
            columns=rp_cut
        )


        xi_vm_diff = pd.DataFrame(
            xi_vm_JK_cut - xi_vm_mean,
            columns=rp_xi_cut
        )


        corr_diff = pd.concat(
            [
                wvp_diff,
                xi_vm_diff
            ],
            axis=1
        )


        all_scales = np.concatenate(
            [
                rp_cut,
                rp_xi_cut
            ]
        )


        NPatches = wvp_JK_cut.shape[0]


        cov_mat = (
            (NPatches - 1) / NPatches
        ) * np.sum(
            np.einsum(
                'ij,ik->ijk',
                corr_diff.values,
                corr_diff.values
            ),
            axis=0
        )


        cov_mat = pd.DataFrame(
            data=cov_mat,
            index=all_scales,
            columns=all_scales
        )


        cov_arr = cov_mat.values


        inv_cov = np.linalg.pinv(
            cov_arr
        )


        # ----------------------------------------------------
        # Joint data vector
        # ----------------------------------------------------

        corr_joint = np.concatenate(
            [
                corr,
                xi_vm_cut
            ]
        )


        assert (
            wvp_JK_cut.shape[0]
            ==
            xi_vm_JK_cut.shape[0]
        ), (
            "wvp and xi_vm JK samples "
            "have different numbers of patches"
        )


        assert (
            wvp_JK_cut.shape[0]
            ==
            NPatches
        )


        # ====================================================
        # MODEL FUNCTIONS
        # ====================================================

        def model_wVI(p):

            A1_v = p[0]

            wVI_model = (
                wVI[model_mask_VI]
                .copy()
            )

            wVI_model *= A1_v

            return (
                model_rp[model_mask_VI],
                wVI_model
            )


        def model_wVII(p):

            A2_v = p[1]

            b_v = p[2]

            wVII_model = (
                wVII[model_mask_VII]
                .copy()
            )

            wVII_model *= A2_v * b_v

            return (
                model_rp[model_mask_VII],
                wVII_model
            )


        def model_xi_vm(p):

            b_v = p[2]

            return (
                rp_xi_cut,
                b_v * xi_mm_interp(rp_xi_cut)
            )


        # ====================================================
        # JOINT LOG-LIKELIHOOD
        # ====================================================

        def log_likelihood(p):

            rp_model_VI, wVI_model = model_wVI(p)

            rp_model_VII, wVII_model = model_wVII(p)


            rp_model_total = np.concatenate(
                [
                    rp_model_VI,
                    rp_model_VII
                ]
            )


            corr_model_total = np.concatenate(
                [
                    wVI_model,
                    wVII_model
                ]
            )


            wvp_model_interpol = np.interp(
                rp_cut,
                rp_model_total,
                corr_model_total
            )


            _, xi_vm_model = model_xi_vm(p)


            model_joint = np.concatenate(
                [
                    wvp_model_interpol,
                    xi_vm_model
                ]
            )


            delta = (
                corr_joint
                -
                model_joint
            )


            chisq = delta.dot(
                inv_cov.dot(delta)
            )


            ll = -0.5 * chisq


            if np.isnan(ll):

                return -np.inf, chisq


            return ll, chisq


        # ====================================================
        # PRIORS
        # ====================================================

        prior_obj = Prior()


        param_names = [
            'A1_v',
            'A2_v',
            'b_v'
        ]


        prior_obj.add_parameter(
            'A1_v',
            dist=(-10.0, 10.0)
        )


        prior_obj.add_parameter(
            'A2_v',
            dist=(-10.0, 10.0)
        )


        prior_obj.add_parameter(
            'b_v',
            dist=(-10.0, 0.0)
        )


        def loglikelihood_fn(p_dict):

            p = np.array(
                [
                    p_dict[key]
                    for key in param_names
                ]
            )


            logl, _ = log_likelihood(p)


            return logl


        box_size = args.box_size

        n_cores = args.n_cores


        # ====================================================
        # NAUTILUS SAMPLER
        # ====================================================

        def run_sampler():

            n_live = 3000


            output_path = (
                '/disks/shear16/herle/models/models/'
            )


            filename = (
                output_path
                +
                f"nautilus_chain_revolver_"
                f"{modes[i]}_voids_{box_size}_"
                f"Rv{bin_label}Mpc_"
                f"env_dep_analytical_hres.h5"
            )


            sampler = Sampler(
                prior_obj,
                loglikelihood_fn,
                n_live=n_live,
                pool=n_cores if n_cores > 1 else None,
                filepath=filename,
                resume=False,
                n_networks=16,
            )


            sampler.run(
                verbose=True,
                discard_exploration=True
            )


            points, log_w, log_l = (
                sampler.posterior()
            )


            weights = np.exp(
                log_w
            )


            logz = sampler.log_z


            output_file = (
                output_path
                +
                f"nautilus_chain_revolver_"
                f"{modes[i]}_voids_{box_size}_"
                f"Rv{bin_label}Mpc_"
                f"env_dep_analytical_hres.npz"
            )


            np.savez(
                output_file,
                samples=points,
                logl=log_l,
                weights=weights,
                logz=logz
            )


            print(
                f"Saved results to {output_file}"
            )


        # ====================================================
        # RUN SAMPLER
        # ====================================================

        if __name__ == '__main__':

            run_sampler()


        # ====================================================
        # LOAD POSTERIOR
        # ====================================================

        filename = (
            f'/disks/shear16/herle/models/models/'
            f'nautilus_chain_revolver_{modes[i]}_voids_'
            f'{box_size}_Rv{bin_label}Mpc_'
            f'env_dep_analytical_hres.npz'
        )


        samples, weights, logl = (
            read_nautilus_samples(
                filename
            )
        )


        getdist = MCSamples(
            samples=samples,
            weights=weights,
            names=names,
            labels=labels,
            ignore_rows=0
        )


        # ====================================================
        # CHI-SQUARED
        # ====================================================

        chi2_eff, chi2_red = compute_chi2(
            logl,
            samples,
            corr_joint
        )


        print(
            f"[{modes[i]} | {bin_label}] "
            f"Reduced Chi2: {chi2_red}"
        )


        # ====================================================
        # POSTERIOR DRAWS
        # ====================================================

        weights_norm = (
            weights
            /
            np.sum(weights)
        )


        inds = np.random.choice(
            len(samples),
            size=500,
            p=weights_norm
        )


        posterior_draws = samples[inds]


        # ====================================================
        # JOINT MAP
        # ====================================================
        #
        # IMPORTANT:
        # The best-fit values are taken from the SAME
        # posterior sample having the maximum JOINT
        # likelihood.
        #
        # These are NOT the marginalised GetDist modes.
        # ====================================================

        map_idx = np.argmax(logl)


        A1_best = samples[
            map_idx,
            0
        ]


        A2_best = samples[
            map_idx,
            1
        ]


        bv_best = samples[
            map_idx,
            2
        ]


        logl_best = logl[
            map_idx
        ]


        print()
        print(
            f"Joint MAP for {modes[i]} "
            f"Rv={bin_label}"
        )

        print(
            f"A1_v = {A1_best}"
        )

        print(
            f"A2_v = {A2_best}"
        )

        print(
            f"b_v  = {bv_best}"
        )

        print(
            f"logL = {logl_best}"
        )

        print(
            f"chi2 = {-2.0 * logl_best}"
        )


        # ====================================================
        # MARGINALISED PARAMETER SUMMARY
        # ====================================================
        #
        # These are retained for reporting, but are NOT used
        # as the best-fit model values.
        # ====================================================

        marginal_modes = {}

        marginal_errors = {}


        for j, name in enumerate(names):

            d = getdist.get1DDensity(
                name
            )


            xyind = np.where(
                d.P == np.amax(d.P)
            )


            marginal_mode = d.x[
                xyind[0]
            ][0]


            errors_density = d.getLimits(
                0.68
            )


            lower_error = (
                marginal_mode
                -
                errors_density[0]
            )


            upper_error = (
                errors_density[1]
                -
                marginal_mode
            )


            marginal_modes[name] = (
                marginal_mode
            )


            marginal_errors[name] = (
                lower_error,
                upper_error
            )


            print(
                f"{name}_marginal_mode "
                f"= {marginal_mode}"
            )


            print(
                f"{name}_err_plus "
                f"= {upper_error}"
            )


            print(
                f"{name}_err_minus "
                f"= {lower_error}"
            )


            print(
                f"{name}_joint_MAP "
                f"= {samples[map_idx, j]}"
            )


        # ====================================================
        # BEST-FIT MODEL VECTORS
        # ====================================================

        wVI_bestfit = (
            wVI
            *
            A1_best
        )


        wVII_bestfit = (
            wVII
            *
            A2_best
            *
            bv_best
        )


        # ====================================================
        # POSTERIOR DRAWS
        # ====================================================

        A1_draws = posterior_draws[:, 0]

        A2_draws = posterior_draws[:, 1]

        bv_draws = posterior_draws[:, 2]


        wVI_ens = []

        wVII_ens = []


        for A1, A2, bv_d in zip(
            A1_draws,
            A2_draws,
            bv_draws
        ):

            wVI_ens.append(
                wVI * A1
            )


            wVII_ens.append(
                wVII * A2 * bv_d
            )


        wVI_ens = np.array(
            wVI_ens
        )


        wVII_ens = np.array(
            wVII_ens
        )


        p16_VI, p50_VI, p84_VI = (
            np.percentile(
                wVI_ens,
                [16, 50, 84],
                axis=0
            )
        )


        p16_VII, p50_VII, p84_VII = (
            np.percentile(
                wVII_ens,
                [16, 50, 84],
                axis=0
            )
        )


        # ====================================================
        # ERRORS FROM JOINT COVARIANCE
        # ====================================================

        n_wvp = len(
            rp_cut
        )


        err_cut = np.sqrt(
            np.diag(cov_arr)
        )[:n_wvp]


        err_xi = np.sqrt(
            np.diag(cov_arr)
        )[n_wvp:]


        mask_data_VI = (
            mask_VI[mask_total]
        )


        mask_data_VII = (
            mask_VII[mask_total]
        )


        # ====================================================
        # BEST-FIT xi_vm + POSTERIOR BAND
        # ====================================================

        _, xi_vm_bestfit = model_xi_vm(
            [
                A1_best,
                A2_best,
                bv_best
            ]
        )


        xi_vm_ens = []


        for A1, A2, bv_d in zip(
            A1_draws,
            A2_draws,
            bv_draws
        ):

            _, xi_vm_draw = model_xi_vm(
                [
                    A1,
                    A2,
                    bv_d
                ]
            )


            xi_vm_ens.append(
                xi_vm_draw
            )


        xi_vm_ens = np.array(
            xi_vm_ens
        )


        p16_xi, p50_xi, p84_xi = (
            np.percentile(
                xi_vm_ens,
                [16, 50, 84],
                axis=0
            )
        )


        color = 'k'

        label = (
            f'{modes[i]} '
            f'$R_v={void_bin[0]}$--${void_bin[1]}$'
        )


        # ====================================================
        # FIGURE
        #
        # 3 rows:
        #   1. wvp main
        #   2. wvp residual
        #   3. xi_vm
        # ====================================================

        fig, (
            ax_wvp,
            ax_wvp_res,
            ax_xi
        ) = plt.subplots(
            3,
            1,
            figsize=(10, 14),
            sharex=False,
            gridspec_kw={
                "height_ratios": [3, 1, 2],
                "hspace": 0.35
            }
        )


        # ====================================================
        # WVP MAIN PANEL
        # ====================================================

        ax_wvp.plot(
            model_rp[model_mask_VII],
            model_rp[model_mask_VII]
            * wVII_bestfit[model_mask_VII],
            color=color,
            label='2-void'
        )


        ax_wvp.plot(
            model_rp[model_mask_VI],
            model_rp[model_mask_VI]
            * wVI_bestfit[model_mask_VI],
            color=color,
            ls='--',
            label='1-void'
        )


        ax_wvp.errorbar(
            rp_cut,
            rp_cut * corr,
            yerr=rp_cut * err_cut,
            fmt='o',
            capsize=3,
            color=color,
            label=label
        )


        ax_wvp.fill_between(
            model_rp[model_mask_VI],
            model_rp[model_mask_VI]
            * p16_VI[model_mask_VI],
            model_rp[model_mask_VI]
            * p84_VI[model_mask_VI],
            alpha=0.25,
            color=color
        )


        ax_wvp.fill_between(
            model_rp[model_mask_VII],
            model_rp[model_mask_VII]
            * p16_VII[model_mask_VII],
            model_rp[model_mask_VII]
            * p84_VII[model_mask_VII],
            alpha=0.25,
            color=color
        )


        ax_wvp.axvspan(
            lower_rp_VI,
            upper_rp_VI,
            color='grey',
            alpha=0.08,
            zorder=0
        )


        ax_wvp.axvspan(
            lower_rp_VII,
            upper_rp_VII,
            color='grey',
            alpha=0.08,
            zorder=0
        )


        ax_wvp.axhline(
            0,
            color='k',
            ls=':'
        )


        ax_wvp.set_xscale(
            'log'
        )


        ax_wvp.set_xlim(
            1,
            600
        )


        ax_wvp.set_ylabel(
            r'$r_p w_{\mathrm{V}+}\,[\mathrm{Mpc}^2/h^2]$'
        )


        ax_wvp.legend(
            frameon=False
        )


        # ====================================================
        # WVP RESIDUALS
        # ====================================================

        model_at_data = np.full_like(
            rp_cut,
            np.nan
        )


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


        residual = (
            100.0
            *
            (corr - model_at_data)
            /
            corr
        )


        frac_err = (
            100.0
            *
            err_cut
            /
            np.abs(corr)
        )


        ax_wvp_res.errorbar(
            rp_cut,
            residual,
            yerr=frac_err,
            fmt='o',
            capsize=3,
            color=color
        )


        ax_wvp_res.axvspan(
            lower_rp_VI,
            upper_rp_VI,
            color='grey',
            alpha=0.08,
            zorder=0
        )


        ax_wvp_res.axvspan(
            lower_rp_VII,
            upper_rp_VII,
            color='grey',
            alpha=0.08,
            zorder=0
        )


        ax_wvp_res.axhline(
            0,
            color='k',
            ls=':'
        )


        ax_wvp_res.set_xscale(
            'log'
        )


        ax_wvp_res.set_xlim(
            1,
            600
        )


        ax_wvp_res.set_ylim(
            -200,
            200
        )


        ax_wvp_res.set_xlabel(
            r'$r_p\,[\mathrm{Mpc}/h]$'
        )


        ax_wvp_res.set_ylabel(
            r'$\Delta/\mathrm{Data}\,[\%]$'
        )


        # ====================================================
        # xi_vm PANEL
        # ====================================================

        ax_xi.plot(
            rp_xi_cut,
            (rp_xi_cut**2) * xi_vm_bestfit,
            color=color,
            label=r'$b_v\,\xi_{mm} \times r^2$'
        )


        ax_xi.fill_between(
            rp_xi_cut,
            (rp_xi_cut**2) * p16_xi,
            (rp_xi_cut**2) * p84_xi,
            alpha=0.25,
            color=color
        )


        ax_xi.errorbar(
            rp_xi_cut,
            (rp_xi_cut**2) * xi_vm_cut,
            yerr=(rp_xi_cut**2) * err_xi,
            fmt='o',
            capsize=3,
            color=color,
            label=r'$\xi_{vm}$ data'
        )


        ax_xi.axhline(
            0,
            color='k',
            ls=':'
        )


        ax_xi.set_xscale(
            'log'
        )


        ax_xi.set_ylabel(
            r'$\xi_{vm}(r)\times r^2$'
        )


        ax_xi.legend(
            frameon=False
        )


        # ====================================================
        # xi_vm RESIDUALS AS INSET
        # ====================================================

        ax_xi_res = ax_xi.inset_axes(
            [
                0.0,
                -0.55,
                1.0,
                0.4
            ]
        )


        residual_xi = (
            100.0
            *
            (xi_vm_cut - xi_vm_bestfit)
            /
            xi_vm_cut
        )


        frac_err_xi = (
            100.0
            *
            err_xi
            /
            np.abs(xi_vm_cut)
        )


        ax_xi_res.errorbar(
            rp_xi_cut,
            residual_xi,
            yerr=frac_err_xi,
            fmt='o',
            capsize=3,
            color=color
        )


        ax_xi_res.axhline(
            0,
            color='k',
            ls=':'
        )


        ax_xi_res.set_xscale(
            'log'
        )


        ax_xi_res.set_xlabel(
            r'$r\,[\mathrm{Mpc}/h]$'
        )


        ax_xi_res.set_ylabel(
            r'$\Delta/\mathrm{Data}\,[\%]$'
        )


        ax_xi_res.set_ylim(
            -200,
            200
        )


        # ====================================================
        # SAVE FIT PLOT
        # ====================================================

        plt.savefig(
            f'/disks/shear16/herle/code/plots/'
            f'check_fits_env_dep_analytical_{bin_label}_hres.png',
            bbox_inches='tight'
        )


        plt.close()


        # ====================================================
        # TRIANGLE PLOT
        # ====================================================

        g = plots.get_subplot_plotter()


        g.settings.fig_width_inch = 12

        g.settings.legend_fontsize = 25

        g.settings.fontsize = 25

        g.settings.axes_labelsize = 30

        g.settings.axes_fontsize = 25


        g.triangle_plot(
            [getdist],
            [
                'A1_v',
                'A2_v',
                'b_v'
            ],
            filled=[True],
            marker_args={
                'lw': 1.5
            },
            contour_colors=['k'],
            legend_kwargs={
                'frameon': False,
                'ncol': 2
            }
        )


        # ----------------------------------------------------
        # Save triangle plot with bin label
        # ----------------------------------------------------

        plt.savefig(
            f'/disks/shear16/herle/code/plots/'
            f'check_contours_env_dep_analytical_{bin_label}_hres.png',
            bbox_inches='tight'
        )


        plt.close()


        # ====================================================
        # END OF VOID BIN
        # ====================================================

        print()
        print(
            f"Finished {modes[i]} "
            f"Rv bin {void_bin[0]}-{void_bin[1]}"
        )

        print(
            "--------------------------------------------------"
        )


# ============================================================
# END
# ============================================================