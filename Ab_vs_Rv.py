import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from getdist import MCSamples
from mnras import *

# -------------------------
# Load void catalog
# -------------------------
voids = np.loadtxt('/disks/shear16/herle/voids/zobov-Flamingo_output_cat.txt')
h = 0.681
voids[:, 1:4] /= h  # convert coordinates to Mpc
cols = ["ID", "x", "y", "z", "Rv"]  # first 5 columns
void_data = pd.DataFrame(voids[:, :5], columns=cols)

# -------------------------
# Define void bins
# -------------------------
rv_bins = [(0, 10), (10,20), (20, 30), (30, 50)]  # in Mpc

# -------------------------
# Store best fits
# -------------------------
A1_best = []
b_best  = []
A1_err_lower = []
A1_err_upper = []
b_err_lower = []
b_err_upper = []
Rv_median = []

for Rmin, Rmax in rv_bins:
    # filter voids in bin
    bin_voids = void_data[(void_data['Rv'] >= Rmin) & (void_data['Rv'] < Rmax)]
    median_rv = bin_voids['Rv'].median()
    Rv_median.append(median_rv)
    
    # load Nautilus chain
    tag = f"{int(Rmin)}_{int(Rmax)}Mpc"
    chain_file = f'/disks/shear16/herle/models/models/nautilus_chain_zobov_voids_1000_Rv{tag}_pimax100_0.0.npz'
    data = np.load(chain_file)
    samples = data['samples']
    weights = data['weights']
    
    # get best-fit and 68% limits using getdist
    gd = MCSamples(samples=samples, weights=weights, names=['A1_v','b_v'], labels=['A1_v','b_v'])
    
    best_fit_results = {}
    for name in ['A1_v','b_v']:
        d = gd.get1DDensity(name)
        xyind = np.where(d.P == np.amax(d.P))
        max_density = d.x[xyind[0]][0]
        errors_density = d.getLimits(0.68)

        lower_error = errors_density[0]
        upper_error = errors_density[1]
        
        best_fit_results[name] = {
            'best_fit': float(max_density),
            'lower_68': float(lower_error),
            'upper_68': float(upper_error)
        }
    
    # store values
    A1_best.append(best_fit_results['A1_v']['best_fit'])
    A1_err_lower.append(best_fit_results['A1_v']['best_fit'] - best_fit_results['A1_v']['lower_68'])
    A1_err_upper.append(best_fit_results['A1_v']['upper_68'] - best_fit_results['A1_v']['best_fit'])
    
    b_best.append(best_fit_results['b_v']['best_fit'])
    b_err_lower.append(best_fit_results['b_v']['best_fit'] - best_fit_results['b_v']['lower_68'])
    b_err_upper.append(best_fit_results['b_v']['upper_68'] - best_fit_results['b_v']['best_fit'])

# -------------------------
# Make figure with asymmetric error bars
# -------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

# A1_v
ax1.errorbar(
    Rv_median, A1_best,
    yerr=[A1_err_lower, A1_err_upper],
    fmt='o-', capsize=4, color='tab:blue'
)
ax1.set_ylabel(r'$A_1^{\mathrm{V}}$')
correct_tick_marks(ax1)

# b_v
ax2.errorbar(
    Rv_median, b_best,
    yerr=[b_err_lower, b_err_upper],
    fmt='s-', capsize=4, color='tab:orange'
)
ax2.set_xlabel('Median $R_v$ [Mpc]')
ax2.set_ylabel(r'$b_v$')
correct_tick_marks(ax2)

plt.tight_layout()

plt.savefig('/disks/shear16/herle/code/plots/bestfit_A_b_vs_Rv.png')