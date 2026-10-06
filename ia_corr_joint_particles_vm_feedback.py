import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ia_correlation_measurements_3d.compute_IA_corr_from_data import make_randoms_box, compute_2p_corr

import gc; gc.collect()

# --- User parameters ---
path = '/disks/shear16/herle/shape_catalogues/catalogs/mass_cut_12/'
run = 'L2800N5040'  # change this to 'L1000N1800' for the smaller box
#run = 'L1000N1800'  
redshift = '0.0'

# --- Determine box size automatically ---
if 'L2800' in run:
    box_size = 2800
elif 'L1000' in run:
    box_size = 1000
else:
    raise ValueError("Unknown catalog. Please set box_size manually.")


h=0.681
# modes = ['FIDUCIAL', 'WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA', 'JETS_published', 'STRONG_JETS_published']

#modes = ['WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA']
modes = ['FIDUCIAL']

r_bins = [
   (0, 10)
    # (10, 15),
    # (15, 20),
    # (20, 25),
    # (25, 30),
    # (30, 50)
]

# r_bins = [
#     (0, 10),
#     (10, 20),
#     (20, 30)
# ]

print('particle data')

# particle_data = (
#     "/disks/shear16/herle/particle_data/L1000N1800_HYDRO_FIDUCIAL_77.npy"
# )

particle_data = (
    "/disks/shear16/herle/particle_data/L2800N5040_HYDRO_FIDUCIAL_78.npy"
)

pos_dm = np.load(particle_data)
pos_dm = pos_dm.astype(np.float32)  # 30GB → 15GB
pos_dm *= np.float32(h)

matter_positions = pd.DataFrame(
    pos_dm,
    columns=['halo_centre_x', 'halo_centre_y', 'halo_centre_z']
)
del pos_dm
gc.collect()

matter_positions['weight'] = np.float32(1.0)
matter_positions['e1'] = np.float32(1.0)
matter_positions['e2'] = np.float32(1.0)

print("x range: ", matter_positions['halo_centre_x'].min(), matter_positions['halo_centre_x'].max())
print("y range: ", matter_positions['halo_centre_y'].min(), matter_positions['halo_centre_y'].max())
print("z range: ", matter_positions['halo_centre_z'].min(), matter_positions['halo_centre_z'].max())


for Rmin, Rmax in r_bins:
    print("Rmin, Rmax = ", (Rmin, Rmax))

    for mode in modes:
        print(f"Processing {mode}")
        # --- Load data ---
        data_file = f'{path}{run}_HYDRO_{mode}_{redshift}_19aug.h5'
        data_shapes = pd.read_hdf(data_file)

        # --- Shape calculations ---
        data_shapes['q'] = data_shapes['b_stellar_it_2d'] / data_shapes['a_stellar_it_2d']
        data_shapes = data_shapes[data_shapes['N_stellar'] > 300]

        data_shapes['e'] = (1 - data_shapes.q) / (1 + data_shapes.q)
        theta = np.arctan2(data_shapes.e1y_stellar_it_2d, data_shapes.e1x_stellar_it_2d)
        data_shapes['e1'] = -data_shapes.e * np.cos(2 * theta)
        data_shapes['e2'] = -data_shapes.e * np.sin(2 * theta)

        a = 1.0/(1.0+float(redshift))
        for coord in ['x', 'y', 'z']:
            data_shapes[f'halo_centre_{coord}'] = (data_shapes[f'halo_centre_{coord}'] *h*(1/a)) / 1e3

        print("x range: ", data_shapes['halo_centre_x'].min(), data_shapes['halo_centre_x'].max())
        print("y range: ", data_shapes['halo_centre_y'].min(), data_shapes['halo_centre_y'].max())
        print("z range: ", data_shapes['halo_centre_z'].min(), data_shapes['halo_centre_z'].max())

        print(len(data_shapes), "galaxies with N_stellar > 300")
        data_shapes['weight'] = np.ones(len(data_shapes['halo_centre_x']))

        #voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_output' + mode + '_cat.txt')
        voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_outputFIDUCIAL_2gigabox_cat.txt')
        #h = 0.681
        #voids[:, 1:4] /= h
        print("Void x range: ", voids[:, 1].min(), voids[:, 1].max())
        print("Void y range: ", voids[:, 2].min(), voids[:, 2].max())
        print("Void z range: ", voids[:, 3].min(), voids[:, 3].max())

        cols = ["ID", "x", "y", "z", "Rv"]  # extend if more columns exist

        void_data = pd.DataFrame(voids[:, :5], columns=cols)

        void_data = void_data[(void_data['Rv'] >= Rmin) & 
                            (void_data['Rv'] < Rmax)]

        print("Number of voids: ", len(void_data))
        # rename coordinates
        void_data = void_data.rename(columns={
            'x': 'halo_centre_x',
            'y': 'halo_centre_y',
            'z': 'halo_centre_z'
        })

        # build positions table
        void_positions = void_data[['halo_centre_x', 'halo_centre_y', 'halo_centre_z']].copy()
        void_positions['weight'] = 1.0

        # --- Prepare for correlations ---
        col_names_box = ['halo_centre_x','halo_centre_y','halo_centre_z', 'e1', 'e2']
        col_names_box += ['weight']

        random_mock_box_void_positions = make_randoms_box(void_positions['halo_centre_x'], void_positions['halo_centre_y'], void_positions['halo_centre_z'], len(void_positions)*10, col_names=col_names_box)
        random_mock_box_shapes = make_randoms_box(data_shapes['halo_centre_x'], data_shapes['halo_centre_y'], data_shapes['halo_centre_z'], len(data_shapes)*10, col_names=col_names_box)

        # --- Correlation parameters ---
        config = {
            'col_names': col_names_box,
            'nbins': 100,
            'rmin': 1,
            'rmax': 600,
            'pi_max': 100,
            'npi': 1,
            'grid_resolution': 3,
            'mubins': 10,
            'NPatches': 125,
            'ncores': 192,
            'slop': 0.,
            'box': True,
            'box_size': box_size * h
        }

        # --- Compute correlations ---
        corr_voids = compute_2p_corr(void_positions, data_shapes, random_mock_box_void_positions, random_mock_box_shapes, config)
        
        corr_voids.compute_wgp()
        corr_voids.compute_xi3d_vv()

        #del random_mock_box_void_positions
        del random_mock_box_shapes
        gc.collect()

        
        # --- Prepare for correlations ---
        # col_names_box = ['particle_position_x','particle_position_y','particle_position_z', 'e1', 'e2']
        # col_names_box += ['weight']
        random_mock_box_matter_positions = make_randoms_box(matter_positions['halo_centre_x'], matter_positions['halo_centre_y'], matter_positions['halo_centre_z'], 500000, col_names=col_names_box)
        #random_mock_box_matter_positions = make_randoms_box(matter_positions['halo_centre_x'], matter_positions['halo_centre_y'], matter_positions['halo_centre_z'], len(matter_positions)*2, col_names=col_names_box)

        # --- Compute correlations ---
        corr_void_particles = compute_2p_corr(
            void_positions, matter_positions,
            random_mock_box_void_positions, random_mock_box_matter_positions,
            config
        )
        corr_void_particles.compute_xi3d_vm(
            corr_void_particles.scat,   # matter data — built from matter_positions
            corr_void_particles.rscat   # matter randoms — built from random_mock_box_matter_positions
        )

        # --- Stack results and save ---
        stacked_data = np.stack([
            
            corr_voids.wgp.rp, corr_voids.wgp.xip,
            corr_voids.wgp.std_from_cov,
            
            corr_voids.xi_vv.r, corr_voids.xi_vv.xi,
            corr_voids.xi_vv.std_from_cov,

            corr_void_particles.xi_vm.r, corr_void_particles.xi_vm.xi,
            corr_void_particles.xi_vm.std_from_cov,

        ], axis=1)

        df = pd.DataFrame(
            stacked_data,
            columns=[
                    'wvp_rp','wvp_xip',
                    'wvp_err',
                    
                    'xi_vv_r', 'xi_vv', 'xi_vv_err',
                    
                    'xi_vm_r', 'xi_vm', 'xi_vm_err'
                    
                ]
        )

        df.to_hdf(f'/disks/shear16/herle/correlations1/correlations_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.h5',
                key='a')

        np.save(f'/disks/shear16/herle/correlations1/wvp_cov_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.npy', corr_voids.wgp.cov_jk)
        np.save(f'/disks/shear16/herle/correlations1/wvp_xip_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.npy', corr_voids.wgp.xip_jk)

        np.save(f'/disks/shear16/herle/correlations1/xi_vv_cov_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.npy', corr_voids.xi_vv.cov_jk)
        np.save(f'/disks/shear16/herle/correlations1/xi_vv_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.npy', corr_voids.xi_vv.xi_jk)

        np.save(f'/disks/shear16/herle/correlations1/xi_vm_cov_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.npy', corr_void_particles.xi_vm.cov_jk)
        np.save(f'/disks/shear16/herle/correlations1/xi_vm_jk_voids_joint_revolver_{Rmin}_{Rmax}hMpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_env_dep_analytical_morerp.npy', corr_void_particles.xi_vm.xi_jk)

        # --- Cleanup before next iteration ---
        del random_mock_box_matter_positions
        del random_mock_box_void_positions
        del corr_voids
        del corr_void_particles
        del void_positions
        del data_shapes
        gc.collect()