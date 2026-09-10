import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from getdist import plots, MCSamples

plt.rcdefaults()
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 22

def read_nautilus_samples(npz_filename):
    data = np.load(npz_filename)
    samples = data['samples']        # shape (Nsamples, Nparams)
    weights = data['weights']        # shape (Nsamples,)
    logl = data['logl']              # shape (Nsamples,)
    return samples, weights, logl


ranges = {
    'A1_v': [1, 10],
    'A2_v': [1, 10]
}

names = ['A1_v', 'A2_v']  
labels = ['A1_v', 'A2_v']

# filename = '/disks/shear16/herle/models/models/nautilus_chain_revolver_FIDUCIAL_voids_1000_Rv10_20hMpc_pimax100_0.0_JK_flamingo_ps_joint3d_sampledb.npz'
# samples, weights, logl = read_nautilus_samples(filename)
# getdist = MCSamples(samples=samples, 
#                             weights=weights,
#                             names=names, 
#                             labels=labels, 
#                             ignore_rows=0)


# # results = {"bin_id": bin_id}
# for i, name in enumerate(names):
#     d = getdist.get1DDensity(name)
#     xyind = np.where(d.P == np.amax(d.P))
#     max_density = d.x[xyind[0]][0]
#     errors_density = d.getLimits(0.68)

#     lower_error = errors_density[0]
#     upper_error = errors_density[1]

#     print(f"{name}_map", max_density)
#     print(f"{name}_err_plus", upper_error - max_density)
#     print(f"{name}_err_minus", max_density - lower_error)    

# g = plots.get_subplot_plotter()
# g.settings.fig_width_inch = 12
# g.settings.legend_fontsize = 25
# g.settings.fontsize = 25
# g.settings.axes_labelsize = 30
# g.settings.axes_fontsize = 25


# g.triangle_plot(
#     [getdist],
#     ['A1_v', 'A2_v'],
#     filled=[True],
#     marker_args={'lw':1.5},
#     contour_colors=['red'],
#     legend_kwargs={'frameon': False, 'ncol': 2}
#     )


# plt.savefig('/disks/shear16/herle/code/plots/check_contours_env_dependance.png')
# plt.close()

ranges = {
    'A1_v': [1, 10],
    'b_v': [1, 10]
}

names = ['A1_v', 'b_v']  
labels = ['A1_v', 'b_v']

filename = '/disks/shear16/herle/models/models/nautilus_chain_revolver_FIDUCIAL_voids_1000_Rv10_20Mpc_pimax100_0.0_JK125_bgtest.npz'
samples, weights, logl = read_nautilus_samples(filename)
getdist = MCSamples(samples=samples, 
                            weights=weights,
                            names=names, 
                            labels=labels, 
                            ignore_rows=0)


# results = {"bin_id": bin_id}
for i, name in enumerate(names):
    d = getdist.get1DDensity(name)
    xyind = np.where(d.P == np.amax(d.P))
    max_density = d.x[xyind[0]][0]
    errors_density = d.getLimits(0.68)

    lower_error = errors_density[0]
    upper_error = errors_density[1]

    print(f"{name}_map", max_density)
    print(f"{name}_err_plus", upper_error - max_density)
    print(f"{name}_err_minus", max_density - lower_error)    

g = plots.get_subplot_plotter()
g.settings.fig_width_inch = 12
g.settings.legend_fontsize = 25
g.settings.fontsize = 25
g.settings.axes_labelsize = 30
g.settings.axes_fontsize = 25


g.triangle_plot(
    [getdist],
    ['A1_v', 'b_v'],
    filled=[True],
    marker_args={'lw':1.5},
    contour_colors=['red'],
    legend_kwargs={'frameon': False, 'ncol': 2}
    )


plt.savefig('/disks/shear16/herle/code/plots/check_contours_bgtest.png')
plt.close()