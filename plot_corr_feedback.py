import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

run = 'L1000N1800'
redshift = '0.0'

modes = [
    'FIDUCIAL',
    'WEAK_AGN',
    'STRONG_AGN',
    'STRONGER_AGN',
    'STRONGEST_AGN',
    'STRONG_SUPERNOVA',
    'JETS_published',
    'STRONG_JETS_published'
]


FLAMINGO_colors = [
    '#117733', '#ABD0E6', '#6AAED6', '#3787C0',
    '#105BA4', '#FF8C40', '#7EFF4B', '#55E18E'
]

FLAMINGO_labels = [
    r"L1$\_$m9",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    r"M*$-\sigma$",
    "JETS",
    r"JETS fgas$-4\sigma$"
]

# -----------------------
# Load fiducial first
# -----------------------
fid_file = (
    f'/disks/shear16/herle/correlations1/'
    f'correlations_revolver_10_30Mpc_{run}_HYDRO_FIDUCIAL_{redshift}.h5'
)

fid = pd.read_hdf(fid_file)

rp_fid = fid['wgp_rp'].values
w_fid = fid['wgp_xip'].values
err_fid = fid['wgp_std_from_cov'].values

# -----------------------
# Figure
# -----------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(6, 7),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# -----------------------
# Main measurement panel
# -----------------------
for mode, color, label in zip(modes, FLAMINGO_colors, FLAMINGO_labels):

    print(f"Processing {mode}")

    data_file = (
        f'/disks/shear16/herle/correlations1/'
        f'correlations_revolver_10_30Mpc_{run}_HYDRO_{mode}_{redshift}.h5'
    )

    data = pd.read_hdf(data_file)

    rp = data['wgp_rp'].values
    w = data['wgp_xip'].values
    err = data['wgp_std_from_cov'].values

    ax1.errorbar(
        rp,
        rp * w,
        yerr=rp * err,
        fmt='o',
        capsize=3,
        color=color,
        label=label
    )

    # -----------------------
    # Ratio panel
    # -----------------------
    delta = w - w_fid

    delta_err = np.sqrt(
        err**2 +
        err_fid**2
    )

    ax2.errorbar(
        rp,
        rp * delta,
        yerr=rp * delta_err,
        fmt='o',
        capsize=3,
        color=color
    )

ax2.axhline(0, color='k', ls='dotted')
ax2.set_ylabel(r'$r_p \Delta w_{\rm V+}$')

# -----------------------
# Cosmetics
# -----------------------
ax1.set_ylim(-5, 5)
ax1.set_xscale('log')

ax1.set_ylabel(
    r'$r_p w_{\mathrm{V}+} \,[\mathrm{Mpc^2/h^2}]$'
)

ax1.axhline(
    0,
    color='k',
    ls='dotted'
)

ax1.legend(frameon=False, ncol=2, loc='upper left')

ax2.set_xscale('log')

ax2.axhline(
    0,
    color='k',
    ls='dotted'
)

ax2.set_ylabel(r'$r_p \Delta w_{\rm V+}$')
ax2.set_xlabel(r'$r_p$ [Mpc/h]')
ax2.set_ylim(-2, 2)

plt.tight_layout()

plt.savefig(
    '/disks/shear16/herle/code/plots/revolver_10_20Mpc_feedback.png'
)