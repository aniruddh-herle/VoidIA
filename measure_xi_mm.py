import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ia_correlation_measurements_3d.compute_IA_corr_from_data import make_randoms_box, compute_2p_corr

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
# modes = ['FIDUCIAL', 'WEAK_AGN', 'STRONG_AGN', 'STRONGER_AGN', 'STRONGEST_AGN', 'STRONG_SUPERNOVA', 'JETS_published', 'STRONG_JETS_published']

modes = ['FIDUCIAL']

for mode in modes:
    print(f"Processing {mode}")

    particle_data = (
        "/disks/shear16/herle/particle_data/L1000N1800_HYDRO_FIDUCIAL_77.npy"
    )

    pos_dm = np.load(particle_data)
    pos_dm *= h

    # build positions table
    data_positions = pd.DataFrame(
        pos_dm,
        columns=['particle_position_x', 'particle_position_y', 'particle_position_z']
    )
    data_positions['weight'] = 1.0
    data_positions['e1'] = 1.0
    data_positions['e2'] = 1.0

    # --- Prepare for correlations ---
    col_names_box = ['particle_position_x','particle_position_y','particle_position_z', 'e1', 'e2']
    col_names_box += ['weight']

    random_mock_box_positions = make_randoms_box(data_positions['particle_position_x'], data_positions['particle_position_y'], data_positions['particle_position_z'], len(data_positions)*10, col_names=col_names_box)

    # --- Correlation parameters ---
    config = {
        'col_names': col_names_box,
        'nbins': 25,
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
    corr_particles = compute_2p_corr(data_positions, data_positions, random_mock_box_positions, random_mock_box_positions, config)
    corr_particles.compute_xi3d_gg()

    # --- Stack results and save ---
    stacked_data = np.stack([
        
        
        corr_particles.xi_gg.r, corr_particles.xi_gg.xi,
        corr_particles.xi_gg.std_from_cov,

        ], axis=1)

    df = pd.DataFrame(
        stacked_data,
        columns=[
                
                'xi_mm_r', 'xi_mm', 'xi_mm_err',
                
               ]
        )

    df.to_hdf(f'/disks/shear16/herle/correlations1/correlations_mm'+run+'_HYDRO_'+mode+'_'+redshift+'.h5',
            key='a')