import os
os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import multiprocessing as mult
import logging
import numpy as np
import pyccl as ccl
from math import factorial
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from copy import deepcopy
import scipy as sp
from time import time
import warnings
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d

class fitIntrinsicAlignmentAmplitude:
    def __init__(
        self,
        cosmology_dict: dict,
        redshift: list,
        interpolation_settings: dict,
        simulation_settings: dict,
        use_nonlinear_power_spectrum: dict,
        k_user: dict,
        integration_options: dict = {'kmin':1e-5, 'kmax':500, 'rel_eps':1e-4, 'n_spline':200},
        multipoles_options: dict = {'mu_r_spline_number':50, 'perpendicular_scale_cut_over_h': 0.1},
        C1rhocrit: float = 0.0134,
        logger: logging.Logger = logging.getLogger(__name__),
        workdir: str = '.',
        sim_outputdir: str = 'w_gp_simulation',
        overwrite_settings: bool = False
       
    ):
        """class to simulate the w_g+ signal and do a subsequent fitting.

        Args:
            cosmology_dict (dict): dictionary with parameters defining a pyccl cosmology
            redshift (list): list of redshifts you want w_g+ evaluated at
            interpolation_settings (dict): distance scales at which to calculate the model w_g+ signal. Must contain "r_min_over_h, r_max_over_h, number_of_points"
            simulation_settings (dict): Settings for the simulation that produced the data_vector. Must contain "pi_max_over_h".
            integration_options (dict): Integration options for the double integral. Must contain "kmin, kmax, rel_eps, n_spline" Best leave this unchanged. kmin should be larger than 1e-6
            multipoles_options (dict): Options for the multipole integration. Must contain "mu_r_spline_number, perpendicular_scale_cut_over_h". Best leave this unchanged.
            C1rhocrit (float, optional): Product of alignment reaction strength C1 and critical density rhocrit. Defaults to 0.0134.
            use_nonlinear_power_spectrum (bool, optional): Use nonlinear power spectrum. Defaults to True.
            logger (logging.Logger, optional): Logger instance for logging (duh). Defaults to logging.getLogger(__name__).
            workdir (str, optional): Working directory. Defaults to '.'.
            sim_outputdir (str, optional): Directory where the numerically calculated w_g+ will be stored relative to workdir. Defaults to 'w_gp_simulation'.
        """
        
        self.logger = logger
        self.logger.info("Initialising intrinsic alignment amplitude simulation and fitter class")

        # Initialise cosmology
        self.cosmo = ccl.Cosmology(**cosmology_dict)
        self.redshift = redshift
        self.logger.debug(f"Initialised cosmology with parameters: {cosmology_dict}")
        
        # Initialise additional parameters
        self.C1rhocrit = C1rhocrit
        self.r_min = interpolation_settings['r_min_over_h']/self.cosmo['h']
        self.r_max = interpolation_settings['r_max_over_h']/self.cosmo['h']
        self.r = np.geomspace(self.r_min, self.r_max, interpolation_settings['number_of_points'])
        self.pi_max = simulation_settings['pi_max_over_h']/self.cosmo['h']
        self.integration_options = integration_options
        self.mu_r_spline_number = multipoles_options['mu_r_spline_number']
        self.perpendicular_scale_cut = multipoles_options['perpendicular_scale_cut_over_h']/self.cosmo['h']
        self.use_nonlinear_power_spectrum = use_nonlinear_power_spectrum

        self.workdir = workdir
        self.outputdir = os.path.join(workdir, sim_outputdir)
        self.k_user = k_user

        # Create folder to store results
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        # TODO: IS THIS INDEED A SMART THING TO DO???
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Interpolate power spectrum
        # ----------------------------------------
        # Note: This horrible, HORRIBLE abomination of a hack is necessary because apparently calling the power spectrum changes something
        # internally in pyccl.cosmology, leading to an unpickel error during multiprocessing 
        cosmo_copy = deepcopy(self.cosmo)

        # Going a bit beyond integration bounds to be safe
        self.k_int_spline = np.geomspace(1e-6, self.integration_options['kmax']*1.2, 5000)
        # self.k_int_spline = np.geomspace(self.integration_options['kmin'], self.integration_options['kmax'], 5000)
        if isinstance(self.use_nonlinear_power_spectrum, np.ndarray):
            # User provided P(k) on some k array
            k_user = self.k_user  # same length as use_nonlinear_power_spectrum
            pk_user = self.use_nonlinear_power_spectrum

            # Interpolate onto integration k grid
            pk_interp = interp1d(k_user, pk_user, kind='cubic', bounds_error=False, fill_value="extrapolate")
            self.pk_int = pk_interp(self.k_int_spline)
            
        
        elif self.use_nonlinear_power_spectrum == 'non-linear':
            self.pk_int = ccl.nonlin_matter_power(cosmo_copy, self.k_int_spline, a=1/(1+redshift[0]))
            
        elif self.use_nonlinear_power_spectrum == 'linear':
            self.pk_int = ccl.linear_matter_power(cosmo_copy, self.k_int_spline, a=1/(1+redshift[0]))

        

        # ----------------------------------------

        # Save a logfile for the simulation containing the input settings
        settings_dict = {
            'cosmology_dict': cosmology_dict,
            'redshift': redshift,
            'interpolation_settings': interpolation_settings,
            'simulation_settings': simulation_settings,
            'integration_options': integration_options,
            'multipoles_options': multipoles_options,
            'C1rhocrit': C1rhocrit,
            'use_nonlinear_power_spectrum': use_nonlinear_power_spectrum,
            'workdir': workdir,
            'sim_outputdir': sim_outputdir,
        }

        settingslog_filepath = os.path.join(self.outputdir, 'settings.txt')
        if os.path.exists(settingslog_filepath):
            self.logger.warning(f"Settings file {settingslog_filepath} already exists.")

            if overwrite_settings:
                self.logger.info(f"Overwriting settings file {settingslog_filepath}.")
                with open(f"{self.outputdir}/settings.txt", 'w') as f:
                    f.write('This logfiles contains the settings for the results in this folder.\n\n')
                    f.write('--------------------------------------------------------------\n')
                    for key, value in settings_dict.items():
                        f.write(key+': '+str(value)+'\n')
                    f.write('--------------------------------------------------------------\n')
            
            else:
                self.logger.info(f"Settings file will not be overwritten. If you want to, set overwrite_settings=True during initialisation")
        else:
            self.logger.info(f"Creating settings file {settingslog_filepath}.")
            with open(f"{self.outputdir}/settings.txt", 'w') as f:
                f.write('This logfiles contains the settings for the results in this folder.\n\n')
                f.write('--------------------------------------------------------------\n')
                for key, value in settings_dict.items():
                    f.write(key+': '+str(value)+'\n')
                f.write('--------------------------------------------------------------\n')

        

    # Define helper functions
    # ===========================================================================
    def __get_spline_for_period(
        self,
        period_idx: int,
        nth_root_of_oscillator: callable,
    ):
        
        if period_idx == 0:
            # if nth_root_of_oscillator(1) > self.integration_options['kmax']:
            #     x_spline = np.geomspace(self.integration_options['kmin'], self.integration_options['kmax'], self.integration_options['n_spline'])
            # else:
            #     x_spline = np.geomspace(self.integration_options['kmin'], nth_root_of_oscillator(1), self.integration_options['n_spline'])

            if nth_root_of_oscillator(1) > self.integration_options['kmax']:
                x_spline = np.geomspace(1e-6, self.integration_options['kmax'], self.integration_options['n_spline'])
            else:
                x_spline = np.geomspace(1e-6, nth_root_of_oscillator(1), self.integration_options['n_spline'])
        else:
            x_spline = np.geomspace(nth_root_of_oscillator(period_idx), nth_root_of_oscillator(period_idx+2), self.integration_options['n_spline'])
        
        return x_spline

    def __check_for_convergence(
        self,
        current_integral: float,
        period_contribution: float,
        period_idx: int,
        convergence_cnt: int,
        nth_root_of_oscillator: callable,
    ):
    
        if nth_root_of_oscillator(period_idx) > self.integration_options['kmax']:
            return True, convergence_cnt
        elif np.abs((current_integral + period_contribution)/current_integral - 1) < self.integration_options['rel_eps']:
            convergence_cnt += 1
            if convergence_cnt > 2:
                return True, convergence_cnt
            else:
                return False, convergence_cnt
            # return True
        else:
            return False, convergence_cnt

    def __integrator_parallel(
        self,
        oscillator: callable,
        nth_root_of_oscillator: callable,
        k_perpendicular: float,
        scale_factor: float,
        k_parallel_power: float,
        k_total_power: float,
    ):

        # Initialise parameters
        LOS_integral = 0
        period_idx_LOS = 0
        convergence_cnt_LOS = 0
        is_converged = False

        # Integrate until convergence
        while not is_converged:
            k_parallel_spline = self.__get_spline_for_period(period_idx_LOS, nth_root_of_oscillator)
            k_total_spline = np.sqrt(k_perpendicular**2 + k_parallel_spline**2)

            # Get the common inner integrand that is to be scaled with the oscillator and k powers
            period_integrand = np.interp(k_total_spline, self.k_int_spline, self.pk_int)
            
            # Filter out k_total < kmin
            period_integrand[k_total_spline < self.integration_options['kmin']] = 0

            # scale with the corresponding oscillator and k_powers
            period_integrand *= oscillator(k_parallel_spline) * k_parallel_spline**k_parallel_power * k_total_spline**k_total_power

            # Integrate and add contributions
            period_integral = np.trapz(period_integrand*k_parallel_spline, np.log(k_parallel_spline))
            LOS_integral += period_integral

            # Increase period index by number of roots. Do one for the initial integral until first root and then go by actual period (2 roots)
            if period_idx_LOS == 0:
                period_idx_LOS += 1
            else:
                period_idx_LOS += 2
                # We check for convergence only now to avoid divide by zero errors
                is_converged, convergence_cnt_LOS = self.__check_for_convergence(LOS_integral, period_integral, period_idx_LOS, convergence_cnt_LOS, nth_root_of_oscillator) 

        
        # At the end of __integrator_parallel, before return:
        # if abs(k_perpendicular - 0.3) < 0.01:
        #     print(f"kperp={k_perpendicular:.4f}, final kz={k_parallel_spline[-1]:.4f}, n_periods={period_idx_LOS}")
                
        return LOS_integral
    
    def __integrator_total(
        self,
        oscillator_parallel: callable,
        nth_root_of_oscillator_parallel: callable,
        oscillator_perpendicular: callable,
        nth_root_of_oscillator_perpendicular: callable,
        k_perpendicular_power: float,
        k_parallel_power: float,
        k_total_power: float,
        scale_factor: float,
        r_perpendicular=None,
    ):
        
        # Initialise parameters
        perpendicular_integral = 0
        period_idx_tot = 0
        convergence_cnt_tot = 0
        is_converged = False

        # ADD THESE to accumulate integrand evaluations
        # debug_kperp_all = []
        # debug_integrand_all = []


        # Integrate until convergence
        while not is_converged:
            k_perpendicular_spline = self.__get_spline_for_period(period_idx_tot, nth_root_of_oscillator_perpendicular)

            # Define integrand
            period_integrand = [self.__integrator_parallel(oscillator_parallel, nth_root_of_oscillator_parallel, k_perpendicular, scale_factor, k_parallel_power, k_total_power) for k_perpendicular in k_perpendicular_spline]
            period_integrand = np.array(period_integrand)
            period_integrand *= oscillator_perpendicular(k_perpendicular_spline) * k_perpendicular_spline**k_perpendicular_power

            # # ADD THESE two lines — no logic changed
            # debug_kperp_all.append(k_perpendicular_spline.copy())
            # debug_integrand_all.append(period_integrand.copy())

            period_integral = np.trapz(period_integrand*k_perpendicular_spline, np.log(k_perpendicular_spline))
            
            # if abs(r_perpendicular - 10.7) < 0.1:
            #     for kp, ig in zip(k_perpendicular_spline, period_integrand):
            #         print(f"  kperp={kp:.6f}, outer_integrand={ig:.6f}")
            
            perpendicular_integral += period_integral

            # Increase period index by number of roots. Do one for the initial integral until first root and then go by actual period (2 roots)
            if period_idx_tot == 0:
                period_idx_tot += 1
            else:
                period_idx_tot += 2
                # We check for convergence only now to avoid divide by zero errors
                is_converged, convergence_cnt_tot = self.__check_for_convergence(perpendicular_integral, period_integral, period_idx_tot, convergence_cnt_tot, nth_root_of_oscillator_perpendicular)

        # debug_kperp_all = np.concatenate(debug_kperp_all)
        # debug_integrand_all = np.concatenate(debug_integrand_all)
        # sort_idx = np.argsort(debug_kperp_all)
        # np.save(f'/disks/shear16/herle/code/VoidIA/data_checks/debug_kperp_mine_rp{r_perpendicular:.1f}.npy', debug_kperp_all[sort_idx])
        # np.save(f'/disks/shear16/herle/code/VoidIA/data_checks/debug_integrand_kperp_mine_rp{r_perpendicular:.1f}.npy', debug_integrand_all[sort_idx])

        self.logger.debug(f"Integral converged after {period_idx_tot//2} iterations.")
        self.logger.debug(f"Relative epsilon at break: {np.abs((perpendicular_integral + period_integral)/perpendicular_integral - 1)}.")
        self.logger.debug(f"Integral converged at k={nth_root_of_oscillator_perpendicular(period_idx_tot)} 1/Mpc.")

        return perpendicular_integral
    # ===========================================================================

    # Define projection integrals
    # ===========================================================================
    # position-position correlation function
    # ----------------------------------------------------------------------
    
    def w_gg_total_integral(
        self,
        r_perpendicular: float,
        scale_factor: float,
    ):
        self.logger.debug(f"Calculating projected position-position correlation function for r_perpendicular={r_perpendicular}, scale_factor={scale_factor}")
        t1 =time()

        # Define oscillators
        oscillator_parallel = lambda x: np.sin(x*self.pi_max)
        nth_root_of_oscillator_parallel = lambda n: n*np.pi/self.pi_max
        oscillator_perpendicular = lambda x: sp.special.jv(0, x*r_perpendicular)
        nth_root_of_oscillator_perpendicular = lambda n: sp.special.jn_zeros(0, n)[-1]/r_perpendicular

        # Specify k scaling power
        k_perpendicular_power = 1
        k_parallel_power = -1
        k_total_power = 0

        # Prefactor scaling
        #scaling = 1/(np.pi**2)
        #scaling = 0.5/np.pi
        scaling = 1.0
        # Initialise integral
        total_integral = self.__integrator_total(
            oscillator_parallel, 
            nth_root_of_oscillator_parallel, 
            oscillator_perpendicular, 
            nth_root_of_oscillator_perpendicular, 
            k_perpendicular_power, 
            k_parallel_power, 
            k_total_power,
            scale_factor
        )
        t2 = time()

        self.logger.info(f"Time for w_gg(r={r_perpendicular}): {t2-t1} seconds.")

        return scaling*total_integral
    # ----------------------------------------------------------------------

    # position-shape correlation function
    # ----------------------------------------------------------------------
    def w_gp_total_integral(
        self,
        r_perpendicular: float,
        scale_factor: float,
    ):
        self.logger.debug(f"Calculating projected position-shape correlation function for r_perpendicular={r_perpendicular}, scale_factor={scale_factor}")
        t1 =time()

        # Define oscillators
        oscillator_parallel = lambda x: np.sin(x*self.pi_max)
        nth_root_of_oscillator_parallel = lambda n: n*np.pi/self.pi_max
        oscillator_perpendicular = lambda x: sp.special.jv(2, x*r_perpendicular)
        nth_root_of_oscillator_perpendicular = lambda n: sp.special.jn_zeros(2, n)[-1]/r_perpendicular

        # Specify k scaling power
        k_perpendicular_power = 3
        k_parallel_power = -1
        k_total_power = -2

        # Prefactor scaling
        #scaling = self.C1rhocrit*self.cosmo['Omega_m']/(np.pi**2)/ccl.growth_factor(self.cosmo, scale_factor)
        scaling  = 1.0
        # Initialise integral
        total_integral = self.__integrator_total(
            oscillator_parallel, 
            nth_root_of_oscillator_parallel, 
            oscillator_perpendicular, 
            nth_root_of_oscillator_perpendicular, 
            k_perpendicular_power, 
            k_parallel_power, 
            k_total_power,
            scale_factor,
            r_perpendicular=r_perpendicular
        )
        t2 = time()

        self.logger.info(f"Time for w_gp(r={r_perpendicular}): {t2-t1} seconds.")

        return scaling*total_integral

    
    # ----------------------------------------------------------------------

    def get_projection_of_a(
        self,
        scale_factor: float,
        projection_type: str='gp',
        n_cores: int = 1,
        from_multipoles: bool = False,
    ):
        
        tasks = [(r_total, scale_factor) for r_total in self.r]
        with mult.Pool(processes=n_cores) as pool:
            if projection_type == 'gp':
                if from_multipoles:
                    projection_total = [self.w_gp_total_integral_from_multipoles(r_total, scale_factor, n_cores) for r_total in self.r]
                    projection_total = np.array(projection_total)*self.cosmo['h']
                else:
                    projection_total = pool.starmap(self.w_gp_total_integral, tasks)
                    projection_total = np.array(projection_total)*self.cosmo['h']
            elif projection_type == 'gg':
                projection_total = pool.starmap(self.w_gg_total_integral, tasks)
                projection_total = np.array(projection_total)*self.cosmo['h']
            else:
                raise ValueError("projection_type must be either 'gp' or 'gg'.")

        return projection_total, self.r*self.cosmo['h']
    
    def simulate_projection(
        self,
        resimulate: bool = False,
        projection_type: str='gp',
        n_cores: int = 1,
        from_multipoles: bool = False,
    ):
        """Simulates the w_gp signal with the NLA model for all input redshifts and stores it for later use. 
        It creates a file with two columns: r [Mpc/h] and w_gg [Mpc/h]^2 or w_gp [Mpc/h].

        Args:
            resimulate (bool, optional): Recalculate the w_gp signal, even if a corresponding file already exists. Defaults to False.
            projection_type (str, optional): Type of projection to simulate. Either 'gp' or 'gg'. Defaults to 'gp'.
            n_cores (int, optional): Number of cores to use for multiprocessing. Defaults to 1.
        """

        for redshift in self.redshift:
            filename = f"{self.outputdir}/w_{projection_type}_z{redshift}.txt"
            if os.path.exists(filename) and not resimulate:
                self.logger.info(f"File {filename} already exists. Skipping redshift {redshift}.")
                continue
            else:
                self.logger.info("-----------------------------")
                self.logger.info(f"Starting simulation of w_{projection_type} for redshift {redshift}")

                time1 = time()
                if projection_type == 'gp':
                    projection_array, r = self.get_projection_of_a(1/(1+redshift), projection_type=projection_type, n_cores=n_cores, from_multipoles=from_multipoles)
                elif projection_type == 'gg':
                    projection_array, r = self.get_projection_of_a(1/(1+redshift), projection_type=projection_type, n_cores=n_cores)
                else:
                    raise ValueError("projection_type must be either 'gp' or 'gg'.")

                time2 = time()
                self.logger.info(f"Simulation took {time2-time1} seconds.")
                self.logger.info(f"Saving results to {filename}")
                self.logger.info('-----------------------------')

                np.savetxt(filename, np.array([r, projection_array]).T)
    # -----------------------------------------------------------------------
    
    # ===========================================================================

    # Define multipole integrals
    # ===========================================================================
    def DEPR_xi_gp_LOS_integral(
        self,
        k_perpendicular: float,
        scale_factor: float,
        r_total: float,
        mu_r: float,
    ):
        # Find oscillatory behaviour
        r_parallel = r_total*mu_r
        oscillator_parallel = lambda x: np.cos(x*r_parallel)
        nth_root_of_oscillator_parallel = lambda n: n*np.pi/(r_parallel)

        # Initialise integral
        LOS_integral = 0
        period_idx_LOS = 0
        convergence_cnt_LOS = 0
        is_converged = False

        # Integrate until convergence
        while not is_converged:
            k_parallel_spline = self.__get_spline_for_period(period_idx_LOS, nth_root_of_oscillator_parallel)
            k_total_spline = np.sqrt(k_perpendicular**2 + k_parallel_spline**2)

            # Do simple trapzoidal integration
            period_integrand = np.interp(k_total_spline, self.k_int_spline, self.pk_int)
            period_integrand *= oscillator_parallel(k_parallel_spline)/(k_total_spline**2)

            # Filter out k_total < kmin
            period_integrand[k_total_spline < self.integration_options['kmin']] = 0

            # Integrate and add contributions
            period_integral = np.trapz(period_integrand*k_parallel_spline, np.log(k_parallel_spline))
            LOS_integral += period_integral

            # Increase period index by number of roots. Do one for the initial integral until first root and then go by actual period (2 roots)
            if period_idx_LOS == 0:
                period_idx_LOS += 1
            else:
                period_idx_LOS += 2
                # We check for convergence only now to avoid divide by zero errors
                is_converged, convergence_cnt_LOS = self.__check_for_convergence(LOS_integral, period_integral, period_idx_LOS, convergence_cnt_LOS, nth_root_of_oscillator_parallel)

        return LOS_integral
        # -----------------------

    def DEPR_xi_gp_total_integral(
        self,
        scale_factor: float,
        r_total: float,
        mu_r: float,
    ):
        
        self.logger.debug(f"Calculating projected galaxy-shape correlation function for r_total={r_total}, scale_factor={scale_factor}, mu_r={mu_r}")
        t1 = time()

        # Find oscillatory behaviour
        r_perpendicular = np.sqrt(r_total**2*(1-mu_r**2))
        oscillator_perpendicular = lambda x: sp.special.jv(2, x*r_perpendicular)
        nth_root_of_oscillator_perpendicular = lambda n: sp.special.jn_zeros(2, n)[-1]/r_perpendicular

        # Initialise parameters
        perpendicular_integral = 0
        period_idx_tot = 0
        convergence_cnt_tot = 0
        is_converged = False

        # Integrate until convergence
        while not is_converged:
            k_perpendicular_spline = self.__get_spline_for_period(period_idx_tot, nth_root_of_oscillator_perpendicular)

            # Define integrand
            period_integrand = [self.DEPR_xi_gp_LOS_integral(k_perpend, scale_factor, r_total, mu_r) for k_perpend in k_perpendicular_spline]
            period_integrand = np.array(period_integrand)
            period_integrand *= oscillator_perpendicular(k_perpendicular_spline)*k_perpendicular_spline**3

            # Integrate and add contributions
            period_integral = np.trapz(period_integrand*k_perpendicular_spline, np.log(k_perpendicular_spline))
            perpendicular_integral += period_integral

            # Increase period index by number of roots. Do one for the initial integral until first root and then go by actual period (2 roots)
            if period_idx_tot == 0:
                period_idx_tot += 1
            else:
                period_idx_tot += 2
                # We check for convergence only now to avoid divide by zero errors
                is_converged, convergence_cnt_tot = self.__check_for_convergence(perpendicular_integral, period_integral, period_idx_tot, convergence_cnt_tot, nth_root_of_oscillator_perpendicular)
        # -----------------------

        scaling = self.C1rhocrit*self.cosmo['Omega_m']/2/(np.pi**2)/ccl.growth_factor(self.cosmo, scale_factor)
        t2 = time()

        self.logger.debug(f"Time for r={r_total}: {t2-t1} seconds.")
        self.logger.debug(f"Integral converged after {period_idx_tot//2} iterations.")
        self.logger.debug(f"Relative epsilon at break: {np.abs((perpendicular_integral + period_integral)/perpendicular_integral - 1)}.")
        self.logger.debug(f"Integral converged at k={nth_root_of_oscillator_perpendicular(period_idx_tot+2)} 1/Mpc.")

        return scaling*perpendicular_integral

    def DEPR_xi_gp_22_multipole(
        self,
        r_total: float,
        scale_factor: float,
    ):

        self.logger.debug(f"Calculating xi_gp_22 for r_total={r_total}, scale_factor={scale_factor} ...")

        if self.perpendicular_scale_cut > r_total:
            self.logger.debug("...done")
            return 0

        else:
            # Define mu_r integration grid
            mu_r_grid = np.linspace(0.001, 0.999, self.mu_r_spline_number)

            # Calculate 3d correlation function
            xi_gp = []
            for mu_r in mu_r_grid:
                xi_gp.append(self.DEPR_xi_gp_total_integral(scale_factor, r_total, mu_r))
            xi_gp = np.array(xi_gp)

            # Calcualte legendre polynomial
            legendre = sp.special.assoc_legendre_p(2, 2, mu_r_grid)[0]

            # Calculate stepfunciton for wedged multipole
            r_perpendicular = np.sqrt(r_total**2*(1-mu_r_grid**2))
            stepfunction = np.zeros_like(mu_r_grid)
            stepfunction[r_perpendicular > self.perpendicular_scale_cut] = 1

            # Calculate integrand and spline integrand
            integrand = xi_gp*legendre*stepfunction
            integrand_spline = sp.interpolate.CubicSpline(mu_r_grid, integrand)

            # Calculate integral
            xi_gp_22 = 2*sp.integrate.quad(integrand_spline, a=mu_r_grid[0], b=mu_r_grid[-1], limit=200)[0]

            # Calculate multipole prefactor and add it
            ell=2
            s_ab=2
            prefactor = (2*ell+1)/2*factorial(ell-s_ab)/factorial(ell+s_ab)
            xi_gp_22 *= prefactor

            self.logger.debug("...done")
            return xi_gp_22

    def DEPR_get_xi_gp22_total_of_a(
        self,
        scale_factor: float,
        n_cores: int,
    ):

        tasks = [(r_total, scale_factor) for r_total in self.r]

        with mult.Pool(processes=n_cores) as pool:
            xi_gp_total = pool.starmap(self.DEPR_xi_gp_22_multipole, tasks)

        return np.array(xi_gp_total), self.r*self.cosmo['h']

    def simulate_xi_gp22(
        self,
        resimulate: bool = False,
        n_cores: int = 1,
        use_deprecated: bool = False,
    ):
        """Simulates the xi_gp_22 signal with the NLA model for all input redshifts and stores it for later use. 
        It creates a file with two columns: r [Mpc/h] and xi_gp_22 [/].

        Args:
            resimulate (bool, optional): Recalculate the signal, even if a corresponding file already exists. Defaults to False.
        """

        for redshift in self.redshift:
            filename = f"{self.outputdir}/xi_gp22_z{redshift}.txt"
            if os.path.exists(filename) and not resimulate:
                self.logger.info(f"File {filename} already exists. Skipping redshift {redshift}.")
                continue
            else:
                self.logger.info("-----------------------------")
                self.logger.info(f"Starting simulation of xi_gp22 for redshift {redshift}")

                time1 = time()
                if use_deprecated:
                    xi_gp22, r = self.DEPR_get_xi_gp22_total_of_a(scale_factor = 1/(1+redshift), n_cores=n_cores)
                else:
                    xi_gp22, r = self.get_xi_gp22_total_of_a(scale_factor = 1/(1+redshift), n_cores=n_cores)
                time2 = time()
                self.logger.info(f"Simulation took {time2-time1} seconds.")
                self.logger.info(f"Saving results to {filename}")
                self.logger.info('-----------------------------')

                np.savetxt(filename, np.array([r, xi_gp22]).T)    


    # Alternative calculation of the multipole with spherical bessel function
    # ---------------------------------------------------------------------------
    def xi_gp_total_integral(
        self,
        r_total: float,
        scale_factor: float,
        spin: int = 2,
    ):
        
        self.logger.debug(f"Calculating projected galaxy-shape correlation function for r_total={r_total}, scale_factor={scale_factor}")
        t1 = time()

        # Find oscillatory behaviour
        oscillator = lambda x: sp.special.spherical_jn(spin, x*r_total)

        # Shamelessly copied from Chatgpt
        # Tested that his indeed finds the correct root.
        # We use the non r-scaled function here since the oscillation is less extreme and easily generalisable.
        # ------------------------------
        def nth_root_of_oscillator(n):
            # Approximate the location of the nth root
            root_approx = (n + 1) * np.pi  # Approximation based on the asymptotic behavior
            
            # Define a function for root finding
            def bessel_function_at_root(x):
                return sp.special.spherical_jn(spin, x)
            
            # Attempt to find the root with a wider search interval
            try:
                # We expand the interval to make sure we bracket the root
                root = sp.optimize.brentq(bessel_function_at_root, root_approx - 2, root_approx + 2)
            except ValueError as e:
                print(f"Root-finding failed for n={n}. Error: {e}")
                root = np.nan  # Return NaN if the root-finding fails
            
            return root/r_total
        # ------------------------------


        # Initialise parameters
        perpendicular_integral = 0
        period_idx_tot = 0
        is_converged = False

        # Here we integrate until kmax since it is fast and the dropoff is slower than with double integration
        while not is_converged:
            k_total_spline = self.__get_spline_for_period(period_idx_tot, nth_root_of_oscillator)

            # Define integrand
            period_integrand = np.interp(k_total_spline, self.k_int_spline, self.pk_int)
            period_integrand *= oscillator(k_total_spline)*(k_total_spline**2)

            # Filter out k_total < kmin
            period_integrand[k_total_spline < self.integration_options['kmin']] = 0

            # Integrate and add contributions
            period_integral = np.trapz(period_integrand*k_total_spline, np.log(k_total_spline))
            perpendicular_integral += period_integral

            # Increase period index by number of roots. Do one for the initial integral until first root and then go by actual period (2 roots)
            if period_idx_tot == 0:
                period_idx_tot += 1
            else:
                period_idx_tot += 2
                # We check for convergence only now to avoid divide by zero errors
            if nth_root_of_oscillator(period_idx_tot) > self.integration_options['kmax']:
                is_converged = True
        # -----------------------

        scaling = self.C1rhocrit*self.cosmo['Omega_m']/2/(np.pi**2)/ccl.growth_factor(self.cosmo, scale_factor)/3
        t2 = time()

        self.logger.debug(f"Time for r={r_total}: {t2-t1} seconds.")
        self.logger.debug(f"Integral converged after {period_idx_tot//2} iterations.")
        self.logger.debug(f"Relative epsilon at break: {np.abs((perpendicular_integral + period_integral)/perpendicular_integral - 1)}.")
        self.logger.debug(f"Integral converged at k={nth_root_of_oscillator(period_idx_tot+2)} 1/Mpc.")

        return scaling*perpendicular_integral
    
    def get_xi_gp22_total_of_a(
        self,
        scale_factor: float,
        n_cores: int,
    ):
        """Soon to be deprecated in favour of get_xi_gpspin_total_of_a"""

        tasks = [(r_total, scale_factor, 2) for r_total in self.r]

        with mult.Pool(processes=n_cores) as pool:
            xi_gp_total = pool.starmap(self.xi_gp_total_integral, tasks)

        return np.array(xi_gp_total), self.r*self.cosmo['h']
    
    def get_xi_gpspin_total_of_a(
        self,
        scale_factor: float,
        n_cores: int,
        spin: int = 2,
    ):

        tasks = [(r_total, scale_factor, spin) for r_total in self.r]

        with mult.Pool(processes=n_cores) as pool:
            xi_gp_total = pool.starmap(self.xi_gp_total_integral, tasks)

        return np.array(xi_gp_total), self.r*self.cosmo['h']
    # ---------------------------------------------------------------------------
    # ===========================================================================

    # Calculate projections from multipoles
    # ===========================================================================
    def w_gp_total_integral_from_multipoles(
        self,
        r_perpendicular: float,
        scale_factor: float,
        n_cores: int = 1,
    ):
        """
        Calculate the projected galaxy-shape correlation function from the multipoles. 
        Currently serves as a consistency check but could become main method in the future due to speed?

        Args:
            r_perpendicular (float): Perpendicular distance
            scale_factor (float): Scale factor
            n_cores (int, optional): Number of cores to use for multiprocessing. Defaults to 1.
        Returns:
            float: Projected galaxy-shape correlation function at r_perpendicular
        """

        # Define Pi array (LOS distance in configuration space) that will be integrated over
        Pi_arr = np.geomspace(1e-5, self.pi_max, 100)
        r_total = np.sqrt(r_perpendicular**2 + Pi_arr**2)

        # Get multipoles
        # Without RSD, only ell=s=2 contributes. Otherwise ell=s=4 becomes important
        # See Singh et al 2023 for more information (https://arxiv.org/pdf/2307.02545v1)
        tasks = [(r_total, scale_factor, 2) for r_total in r_total]
        with mult.Pool(processes=n_cores) as pool:
            xi_gp22_total = pool.starmap(self.xi_gp_total_integral, tasks)

        # Scale with legendre function
        legendre_2 = sp.special.assoc_legendre_p(2, 2, Pi_arr/r_total)[0]
        integrand_2 = xi_gp22_total*legendre_2

        # Integrate over Pi
        contribution_22 = 2*np.trapz(integrand_2, Pi_arr)

        return contribution_22
        
    # ===========================================================================

    def load_projection(self, redshift: float, projection_type: str='gp'):
        filename = f"{self.outputdir}/w_{projection_type}_z{redshift}.txt"
        data = np.loadtxt(filename)
        r = data[:,0]
        projection = data[:,1]
        return r, projection
    
    def load_xi_gp22(self, redshift: float):
        filename = f"{self.outputdir}/xi_gp22_z{redshift}.txt"
        data = np.loadtxt(filename)
        r = data[:,0]
        xi_gp22 = data[:,1]
        return r, xi_gp22
        
    def project_model_to_data(self, r_data, model, r_model):
        """Simple interpolator to project the model to the data scales. Scaling needs to be manually added afterwards.
        Args:
            r_data (np.array): data scale array
            model (np.array): model prediction array
            r_model (np.array): model scale array

        Returns:
            np.array: model vector projected to data scales
        """
        return np.interp(r_data, r_model, model)

    def __scale_cut(self, r_min, r_max, r, data, cov, return_mask = False):
        mask = (r > r_min) & (r < r_max)
        r_cut = r[mask]
        data_cut = data[mask]
        cov_cut = cov[mask][:,mask]
        if return_mask:
            return r_cut, data_cut, cov_cut, mask
        else:
            return r_cut, data_cut, cov_cut
    
    def __renormalise_input(self, data, model, cov):
        
        # Renormalise covariance and data vector
        sigma = np.sqrt(np.diagonal(cov))
        data_renorm = data/sigma
        model_renorm = model/sigma
        covariance_renorm = cov/sigma[:,None]/sigma[None,:]
        
        return data_renorm, model_renorm, covariance_renorm

    def get_chi2(self, 
        A_IA, 
        b_g, 
        r_data,
        data,
        cov,
        fitting_range: list = [5, 16],
        projection_type: str = 'gp',
        redshift: float = None,    
    ):
        # Get the model for the data
        if redshift is None:
            redshift = self.redshift[0]
        
        try:
            r_model, model = self.load_projection(redshift, projection_type=projection_type)
        except FileNotFoundError:
            self.logger.critical(f"File for redshift {redshift} not found. Exiting script.")
            self.logger.critical("Did you run simulate_w_gp() before calling get_chi2()?")
            sys.exit
        
        # Cut data to relevant scales and then project model to data
        r_data, data, cov = self.__scale_cut(fitting_range[0], fitting_range[1], r_data, data, cov)

        model = self.project_model_to_data(r_data, model, r_model)

        # Renormalise input for more robust fitting
        # data_renorm, model_renorm, covariance_renorm = self.__renormalise_input(data, model, cov)
        data_renorm, model_renorm, covariance_renorm = data, model, cov

        # Scale model with A_IA and b_g
        if projection_type == 'gp':
            model_renorm *= A_IA*b_g
        elif projection_type == 'gg':
            model_renorm *= b_g**2

        # Calculate chi2
        chi2 = (data_renorm - model_renorm) @ np.linalg.inv(covariance_renorm) @ (data_renorm - model_renorm)

        return chi2


    def fit_Aia_bg_to_data(
            self, 
            r_data, 
            data, 
            cov,
            redshift: float = None, 
            fitting_range: list = [5, 16], 
            measurement: str = 'w_gp',
            use_SVD: bool = False,
            n_jk: int = None,
            return_chi2: bool = False,
            include_Hartlab_correction: bool = False,
        ):
        
        # THIS IS A HACKY SHITTY WAY TO DO THIS TODO: FIX THIS
        if redshift is None:
            redshift = self.redshift[0]

        try:
            if measurement == 'w_gp':
                r_model, model = self.load_projection(redshift, projection_type='gp')
            elif measurement == 'xi_gp22':
                r_model, model = self.load_xi_gp22(redshift)
            else:
                self.logger.critical("Invalid measurement. Use 'w_gp' or 'xi_gp22'.")

        except FileNotFoundError:
            self.logger.critical(f"File for redshift {redshift} not found. Exiting script.")
            self.logger.critical("Did you run simulate_w_gp() before calling fit_Aia_bg_to_data()?")
            sys.exit

        # Cut data to relevant scales and then project model to data
        r_data, data, cov = self.__scale_cut(fitting_range[0], fitting_range[1], r_data, data, cov)
        model = self.project_model_to_data(r_data, model, r_model)

        # Apply Hartlap correction if requested (https://arxiv.org/abs/astro-ph/0608064)
        # This is only valid for jackknife covariance matrices
        if include_Hartlab_correction:
            data_length = len(data)
            cov *= (n_jk -1)/(n_jk - data_length - 2)

        # Renormalise input for more robust fitting
        data_renorm, model_renorm, covariance_renorm = self.__renormalise_input(data, model, cov)

        if use_SVD:
            if n_jk is None:
                self.logger.critical("If using SVD, you must specify the number of jackknife samples.")
                sys.exit

            # SVD decomposition
            u, s2, vT = np.linalg.svd(covariance_renorm)

            # Rotate data and model into new basis
            data_rebase = u.T @ data_renorm
            model_rebase = u.T @ model_renorm

            # Filter out small singular values
            filtering_low_svd_bool = np.sqrt(2/n_jk) < s2
            data_rebase = data_rebase[filtering_low_svd_bool]
            model_rebase = model_rebase[filtering_low_svd_bool]
            s2 = s2[filtering_low_svd_bool]

            print(f"Points used for fitting: {len(data_rebase)}")

            # Defining fitting function
            data_func = lambda x, scaling: scaling*model_rebase

            fit_params, fit_cov = curve_fit(data_func, np.ones_like(data_rebase), data_rebase, p0=[0.5], sigma=np.sqrt(s2), absolute_sigma=True)

            # Calculate chi2
            chi2 = (data_rebase - model_rebase*fit_params[0]) @ np.linalg.inv(np.diag(s2)) @ (data_rebase - model_rebase*fit_params[0])
            reduced_chi2 = chi2/(len(data_rebase) - len(fit_params))

        else:
            # Fit with scipy
            data_func = lambda x, scaling: scaling*model_renorm
            fit_params, fit_cov = curve_fit(data_func, np.ones_like(data_renorm), data_renorm, p0=[1], sigma=covariance_renorm, absolute_sigma=True)
        
            # Calculate chi2
            chi2 = (data_renorm - model_renorm*fit_params[0]) @ np.linalg.inv(covariance_renorm) @ (data_renorm - model_renorm*fit_params[0])
            reduced_chi2 = chi2/(len(data_renorm) - len(fit_params))

        self.logger.info(f"Fitted scaling for redshift {redshift}: A_IA*b_g = {fit_params[0]}+-{np.sqrt(fit_cov[0][0])}")
        self.logger.info(f"Reduced chi2: {reduced_chi2}")

        if return_chi2:
            return fit_params[0], np.sqrt(fit_cov[0][0]), reduced_chi2
        else:
            return fit_params[0], np.sqrt(fit_cov[0][0])
    
    def fit_Aia_bg_to_data_with_SVD(self, r_data, data, cov, n_jk, fitting_range: list = [5, 16]):
        # METHOD INCLUDED IN fit_Aia_bg_to_data()
        # TODO: REMOVE THIS FUNCTION ONCE SCRIPTS NO LONGER DEPEND ON IT
        self.logger.warning("This function is deprecated. Use fit_Aia_bg_to_data() with use_SVD=True instead.")

        # Iterating over input redshifts
        for redshift in self.redshift:
            self.logger.info(f"Starting fit for redshift {redshift}")

            try:
                r_model, model = self.load_projection(redshift, projection_type='gp')
            except FileNotFoundError:
                self.logger.critical(f"File for redshift {redshift} not found. Exiting script.")
                self.logger.critical("Did you run simulate_w_gp() before calling fit_Aia_bg_to_data()?")
                sys.exit

            # Cut data to relevant scales and then project model to data and 
            r_data, data, cov = self.__scale_cut(fitting_range[0], fitting_range[1], r_data, data, cov)
            model = self.project_model_to_data(r_data, model, r_model)

            # Renormalise input for more robust fitting
            data_renorm, model_renorm, covariance_renorm = self.__renormalise_input(data, model, cov)

            # SVD decomposition
            u, s2, vT = np.linalg.svd(covariance_renorm)

            # Rotate data and model into new basis
            data_rebase = u.T @ data_renorm
            model_rebase = u.T @ model_renorm

            # Filter out small singular values
            filtering_low_svd_bool = np.sqrt(2/n_jk) < np.sqrt(s2)
            data_rebase = data_rebase[filtering_low_svd_bool]
            model_rebase = model_rebase[filtering_low_svd_bool]
            s2 = s2[filtering_low_svd_bool]

            # Defining fitting function
            data_func = lambda x, scaling: scaling*model_rebase

            fit_params, fit_cov = curve_fit(data_func, np.ones_like(data_rebase), data_rebase, p0=[0.5], sigma=np.sqrt(s2), absolute_sigma=True)
            
            self.logger.info(f"Fitted scaling for redshift {redshift}: A_IA*b_g = {fit_params[0]}+-{np.sqrt(fit_cov[0][0])}")

        return fit_params[0], np.sqrt(fit_cov[0][0])


    @staticmethod
    def return_input_dict_from_output_filepath(filepath):

        settings_dict = {}
        with open(f"{filepath}/settings.txt", 'r') as f:
            lines = f.readlines()
            for line in lines[3:-1]:  # Skip the first 3 lines and the last 2 lines
                key, value = line.strip().split(':', 1)  # Split by the first colon
                key = key.strip()
                value = value.strip()

                # Convert value to appropriate type
                if value.startswith('{') and value.endswith('}'):  # Handle dictionaries
                    settings_dict[key] = eval(value)
                elif value.startswith('[') and value.endswith(']'):  # Handle lists
                    settings_dict[key] = eval(value)
                elif value.isdigit():  # Handle integers
                    settings_dict[key] = int(value)
                else:
                    try:
                        settings_dict[key] = float(value)  # Handle floats
                    except ValueError:
                        settings_dict[key] = value  # Handle strings

        settings_dict['logger'] = logging.Logger(__file__)
        return settings_dict