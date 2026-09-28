import numpy as np
import matplotlib.pyplot as plt
import os

from getdist import plots, MCSamples


# =====================================================
# Settings
# =====================================================

os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

plt.rcdefaults()
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 22


# =====================================================
# Functions
# =====================================================

def read_nautilus_samples(npz_filename):

    data = np.load(npz_filename)

    samples = data['samples']
    weights = data['weights']
    logl = data['logl']

    return samples, weights, logl



# =====================================================
# Parameters
# =====================================================

names = ['A1_v', 'b_v']
labels = ['A1_v', 'b_v']


modes = [
    'FIDUCIAL',
    'WEAK_AGN',
    'STRONG_AGN',
    'STRONGER_AGN',
    'STRONGEST_AGN',
    'STRONG_SUPERNOVA',
    'FIDUCIAL_2p8'
]


FLAMINGO_colors = [
    '#117733',
    '#ABD0E6',
    '#6AAED6',
    '#3787C0',
    '#105BA4',
    '#FF8C40',
    '#332288'
]

FLAMINGO_labels = [
    r"L1_m9",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    r"M_*$-\sigma$",
    r"L2p8$_m9$"
]

# Standard binning for all models except FIDUCIAL_2p8
r_bins_standard = [
    (0, 10),
    (10, 20),
    (20, 30)
]

# FIDUCIAL_2p8 binning
r_bins_2p8 = [
    (0, 10),
    (10, 15),
    (15, 20),
    (20, 25),
    (25, 30),
    (30, 50)
]

# Rv_median = {

#     'FIDUCIAL':
#     {
#         (0,10): 8.46,
#         (10,20): 14.72,
#         (20,30): 23.17,
#         (30,50): 32.50
#     },

#     'WEAK_AGN':
#     {
#         (0,10): 8.49,
#         (10,20): 14.65,
#         (20,30): 23.10,
#         (30,50): 32.15
#     },

#     'STRONG_AGN':
#     {
#         (0,10): 8.41,
#         (10,20): 14.70,
#         (20,30): 22.94,
#         (30,50): 32.04
#     },

#     'STRONGER_AGN':
#     {
#         (0,10): 8.44,
#         (10,20): 14.58,
#         (20,30): 23.10,
#         (30,50): 31.98
#     },

#     'STRONGEST_AGN':
#     {
#         (0,10): 8.37,
#         (10,20): 14.62,
#         (20,30): 22.94,
#         (30,50): 31.90
#     },

#     'STRONG_SUPERNOVA':
#     {
#         (0,10): 8.59,
#         (10,20): 14.83,
#         (20,30): 23.19,
#         (30,50): 32.29
#     },
#     'FIDUCIAL_2p8':
#     {   
#         (0,10): 8.28,
#         (10,20): 14.37,
#         (20,30): 22.82,
#         (30,50): 31.92
#     }
# }
Rv_mean = {

    'FIDUCIAL':
    {
        (0,10): 8.23,
        (10,20): 14.82,
        (20,30): 23.65,
        (30,50): 33.17
    },

    'WEAK_AGN':
    {
        (0,10): 8.27,
        (10,20): 14.74,
        (20,30): 23.59,
        (30,50): 32.96
    },

    'STRONG_AGN':
    {
        (0,10): 8.18,
        (10,20): 14.78,
        (20,30): 23.47,
        (30,50): 32.91
    },

    'STRONGER_AGN':
    {
        (0,10): 8.25,
        (10,20): 14.70,
        (20,30): 23.59,
        (30,50): 32.98
    },

    'STRONGEST_AGN':
    {
        (0,10): 8.15,
        (10,20): 14.73,
        (20,30): 23.48,
        (30,50): 32.73
    },

    'STRONG_SUPERNOVA':
    {
        (0,10): 8.36,
        (10,20): 14.88,
        (20,30): 23.71,
        (30,50): 33.10
    },

    'FIDUCIAL_2p8':
    {
        (0,10): 8.04,
        (10,15): 12.49,
        (15,20): 17.28,
        (20,25): 22.12,
        (25,30): 26.98,
        (30,50): 32.56
    }
}

bin_labels = {
    (0, 10):  '0_10',
    (10, 15): '10_15',
    (10, 20): '10_20',
    (15, 20): '15_20',
    (20, 25): '20_25',
    (20, 30): '20_30',
    (25, 30): '25_30',
    (30, 50): '30_50',
}


# =====================================================
# Storage for Av,bv evolution
# =====================================================

parameter_results = {

    mode:
    {
        'Rv': [],

        'A1_v': [],
        'A1_v_low': [],
        'A1_v_high': [],

        'b_v': [],
        'b_v_low': [],
        'b_v_high': []
    }

    for mode in modes
}



# =====================================================
# Main loop over radius bins
# =====================================================

for i, mode in enumerate(modes):

    print("-----------------------------------")
    print(f"Processing {mode}")
    print("-----------------------------------")

    # Select appropriate Rv bins
    if mode == 'FIDUCIAL_2p8':
        r_bins = r_bins_2p8
    else:
        r_bins = r_bins_standard

    # Box size
    box_size = 2800 if mode == 'FIDUCIAL_2p8' else 1000

    # ----------------------------------------------
    # Loop over radius bins
    # ----------------------------------------------

    for Rmin, Rmax in r_bins:

        print(f"Processing Rv bin {Rmin}-{Rmax}")

        # Store posterior samples for this bin
        getdist_samples_all = []

        filename = (
            '/disks/shear16/herle/models/models/'
            f'nautilus_chain_revolver_{mode}_voids_{box_size}'
            f'_Rv{bin_labels[(Rmin,Rmax)]}Mpc_seperate_env_dep_analytical.npz'
        )

        print(filename)


        samples, weights, logl = read_nautilus_samples(filename)


        gd_samples = MCSamples(
            samples=samples,
            weights=weights,
            names=names,
            labels=labels,
            ignore_rows=0,
            label=FLAMINGO_labels[i]
        )


        getdist_samples_all.append(gd_samples)



        # ------------------------------------------
        # Extract best fit and 68% intervals
        # ------------------------------------------

        best_fit_results = {}


        for name in names:


            density = gd_samples.get1DDensity(name)


            max_index = np.argmax(density.P)

            best = density.x[max_index]


            limits = density.getLimits(0.68)


            best_fit_results[name] = {

                'best_fit': float(best),

                'lower_68': float(limits[0]),

                'upper_68': float(limits[1])

            }


        # ------------------------------------------
        # Store vs Rv
        # ------------------------------------------

        #Rv = Rv_median[mode][(Rmin,Rmax)]
        Rv = Rv_mean[mode][(Rmin,Rmax)]

        parameter_results[mode]['Rv'].append(Rv)


        for p in ['A1_v','b_v']:

            best = best_fit_results[p]['best_fit']
            low = best_fit_results[p]['lower_68']
            high = best_fit_results[p]['upper_68']


            parameter_results[mode][p].append(best)
            parameter_results[mode][p+'_low'].append(best-low)
            parameter_results[mode][p+'_high'].append(high-best)



    # =====================================================
    # Triangle plot AFTER all modes loaded
    # =====================================================

    g = plots.get_subplot_plotter()


    g.settings.fig_width_inch = 12
    g.settings.legend_fontsize = 25
    g.settings.fontsize = 25
    g.settings.axes_labelsize = 30
    g.settings.axes_fontsize = 25


    g.triangle_plot(

        getdist_samples_all,

        ['A1_v','b_v'],

        filled=[True]*len(modes),

        marker_args={'lw':1.5},

        contour_colors=FLAMINGO_colors,

        legend_labels=FLAMINGO_labels,

        legend_kwargs={
            'frameon':False,
            'ncol':2
        }
    )


    outfile = (
        '/disks/shear16/herle/code/plots/'
        f'zobov_all_modes_void_model_Rv{Rmin}_{Rmax}hMpc.png'
    )


    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


    print(f"Saved {outfile}")





# =====================================================
# A1_v and b_v versus median void radius
# =====================================================


fig, axes = plt.subplots(
    2,
    1,
    figsize=(8,12),
    sharex=True
)



for i, mode in enumerate(modes):


    Rv = np.array(
        parameter_results[mode]['Rv']
    )


    axes[0].errorbar(

        Rv,

        parameter_results[mode]['A1_v'],

        yerr=[
            parameter_results[mode]['A1_v_low'],
            parameter_results[mode]['A1_v_high']
        ],

        marker='o',

        markersize=8,

        linewidth=2,

        capsize=4,

        color=FLAMINGO_colors[i],

        label=FLAMINGO_labels[i]

    )


    axes[1].errorbar(

        Rv,

        parameter_results[mode]['b_v'],

        yerr=[
            parameter_results[mode]['b_v_low'],
            parameter_results[mode]['b_v_high']
        ],

        marker='o',

        markersize=8,

        linewidth=2,

        capsize=4,

        color=FLAMINGO_colors[i]

    )


hamaus2014a_x = np.array([5.059159748485247, 6.867251016986972, 8.12390934941441, 9.246456994753434,
    10.366029488331112, 11.580463551530201, 12.990799915322604, 14.852329486614677, 17.71762949061969,
    28.296811438314233])
hamaus2014a_y = np.array([1.4622723552281562, 1.200046040515654, 1.0503376304481278, 0.8337681604256191,
    0.8149938612645795, 0.712476979742173, 0.5347350112543483, 0.34676181706568454, 0.12955289543687343,
    -0.6734960098219767])

axes[1].plot(hamaus2014a_x, hamaus2014a_y, linestyle='-', color='k', label='Hamaus2014a')

axes[0].set_ylabel(r'$A_{v}$')

axes[1].set_ylabel(r'$b_v$')

axes[1].set_xlabel(
    r'$R_{\rm v,mean}\,[h^{-1}{\rm Mpc}]$'
)


axes[0].axhline(
    0,
    ls='--',
    lw=1
)

axes[1].axhline(
    0,
    ls='--',
    lw=1
)

axes[0].legend(
    fontsize=16,
    frameon=False,
    loc='upper right'
)

# axes[1].legend(
#     fontsize=16,
#     frameon=False,
#     loc='upper right'
# )




plt.tight_layout()


plt.savefig(

    '/disks/shear16/herle/code/plots/'
    'void_parameters_vs_Rv.png',

    dpi=300,

    bbox_inches='tight'

)


plt.close()


print("Finished all plots")