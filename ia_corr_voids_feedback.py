import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ia_correlation_measurements.compute_IA_corr_from_data import make_randoms_box, compute_2p_corr


# --- User parameters ---
path = '/disks/shear16/herle/shape_catalogues/catalogs/mass_cut_12/'
#run = 'L2800N5040'  # change this to 'L1000N1800' for the smaller box
run = 'L1000N1800'  
redshift = '0.0'

# --- Determine box size automatically ---
if 'L2800' in run:
    box_size = 2800
elif 'L1000' in run:
    box_size = 1000
else:
    raise ValueError("Unknown catalog. Please set box_size manually.")


h=0.681
modes = ['FIDUCIAL', 'WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA', 'JETS_published', 'STRONG_JETS_published']

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

    voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_output' + mode + '_cat.txt')
    #h = 0.681
    #voids[:, 1:4] /= h
    print("Void x range: ", voids[:, 1].min(), voids[:, 1].max())
    print("Void y range: ", voids[:, 2].min(), voids[:, 2].max())
    print("Void z range: ", voids[:, 3].min(), voids[:, 3].max())

    cols = ["ID", "x", "y", "z", "Rv"]  # extend if more columns exist

    void_data = pd.DataFrame(voids[:, :5], columns=cols)

    void_data = void_data[(void_data['Rv'] >= 10) & 
                        (void_data['Rv'] <= 20)]

    print("Number of voids: ", len(void_data))
    # rename coordinates
    void_data = void_data.rename(columns={
        'x': 'halo_centre_x',
        'y': 'halo_centre_y',
        'z': 'halo_centre_z'
    })

    # build positions table
    data_positions = void_data[['halo_centre_x', 'halo_centre_y', 'halo_centre_z']].copy()
    data_positions['weight'] = 1.0

    # --- Prepare for correlations ---
    col_names_box = ['halo_centre_x','halo_centre_y','halo_centre_z', 'e1', 'e2']
    col_names_box += ['weight']

    random_mock_box_positions = make_randoms_box(data_positions['halo_centre_x'], data_positions['halo_centre_y'], data_positions['halo_centre_z'], len(data_positions)*10, col_names=col_names_box)
    random_mock_box_shapes = make_randoms_box(data_shapes['halo_centre_x'], data_shapes['halo_centre_y'], data_shapes['halo_centre_z'], len(data_shapes)*10, col_names=col_names_box)

    # --- Correlation parameters ---
    config = {
        'col_names': col_names_box,
        'nbins': 20,
        'rmin': 1,
        'rmax': 600,
        'pi_max': 100,
        'npi': 1,
        'grid_resolution': 3,
        'mubins': 10,
        'NPatches': 125,
        'delete_n': 1,
        'ncores': 128,
        'slop': 0.,
        'box': True,
        'box_size': box_size * h
    }

    # --- Compute correlations ---
    corr = compute_2p_corr(data_positions, data_shapes, random_mock_box_positions, random_mock_box_shapes, config)

    corr.compute_wgp()

    # --- Stack results and save ---
    stacked_data = np.stack([
        corr.wgp.rp, corr.wgp.xip,
        corr.wgp.std_from_cov
        
    ], axis=1)

    df = pd.DataFrame(
        stacked_data,
        columns=['wgp_rp','wgp_xip',
                'wgp_std_from_cov']
    )

    df.to_hdf('/disks/shear16/herle/correlations1/correlations_revolver_10_20Mpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_JK.h5',
            key='a')

    np.save('/disks/shear16/herle/correlations1/wgp_cov_jk_voids_revolver_10_20Mpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_JK.npy', corr.wgp.cov_jk)
    np.save('/disks/shear16/herle/correlations1/wgp_xip_jk_voids_revolver_10_20Mpc_'+run+'_HYDRO_'+mode+'_'+redshift+'_JK.npy', corr.wgp.xip_jk)
