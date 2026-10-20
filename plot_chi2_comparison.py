import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── config ────────────────────────────────────────────────────────────────────

base_dir    = '/disks/shear16/herle/code/temp_data/'
output_fig  = '/disks/shear16/herle/code/plots/chi2red_comp_1gpc.png'
output_fig2 = '/disks/shear16/herle/code/plots/chi2red_comp_1gpc_wl.png'

modes = ['FIDUCIAL']

# ── mean Rv x-axis values ─────────────────────────────────────────────────────
# standard: 3-bin (0-10, 10-20, 20-30)
# 2p8:      6-bin (0-10, 10-15, 15-20, 20-25, 25-30, 30-50)

rv_mean_x = {
    'standard': np.array([8.28, 14.37, 22.82]),
    '2p8':      np.array([8.23, 12.49, 17.17, 21.94, 26.76, 31.92]),
}

def x_key(variant):
    return '2p8' if '2p8' in variant else 'standard'

# ── style maps ────────────────────────────────────────────────────────────────

color_map = {
    100: '#1f77b4',
    200: '#ff7f0e',
    300: '#2ca02c',
    400: '#d62728',
}
ls_map     = {'': '-', '_xi_vm': '--', '_hres': ':', '_2p8': '-.', '_xi_vm_2p8': '--'}
lw_map     = {'': 1.8, '_xi_vm': 1.8, '_hres': 2.2, '_2p8': 1.8, '_xi_vm_2p8': 1.8}
marker_map = {'': 'o', '_xi_vm': 's', '_hres': '^', '_2p8': 'D', '_xi_vm_2p8': 'D'}

label_variant = {
    '':           'base',
    '_xi_vm':     r'$\xi_{vm}$',
    '_hres':      'hres',
    '_2p8':       '2p8',
    '_xi_vm_2p8': r'$\xi_{vm}$ 2p8',
}

# ── files grouped by rv_type ──────────────────────────────────────────────────

groups = {
    'Rv_mean': [
        (100, ''), (100, '_xi_vm'),
        (200, ''), (200, '_xi_vm'),
        (300, ''), (300, '_xi_vm'), (300, '_hres'),
        (400, ''), (400, '_xi_vm'), (400, '_hres'),
        (400, '_2p8'), (400, '_xi_vm_2p8'),
    ],
    'Rv_median': [
        (100, ''), (100, '_xi_vm'), (100, '_hres'),
        (200, ''), (200, '_xi_vm'), (200, '_hres'),
        (300, ''), (300, '_xi_vm'), (300, '_hres'),
        (400, ''), (400, '_xi_vm'), (400, '_hres'),
    ],
}

panel_titles = {
    'Rv_mean':   r'$R_v$ mean',
    'Rv_median': r'$R_v$ median',
}

# ── plot 1 (original) ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, (rv_type, entries) in zip(axes, groups.items()):

    for rmax, variant in entries:

        suffix = f'{rv_type}_rmax{rmax}{variant}'
        fname  = f'{base_dir}chi2red_results_env_dep_analytical_mean_{suffix}.npz'

        try:
            data = np.load(fname, allow_pickle=True)
        except FileNotFoundError:
            print(f"Missing: {fname} — skipping")
            continue

        xvals = rv_mean_x[x_key(variant)]

        for mode in modes:
            try:
                chi2_red = data[f'{mode}_chi2_red']
            except KeyError:
                print(f"Mode '{mode}' not found in {fname} — skipping")
                continue

            ax.plot(
                xvals, chi2_red,
                color=color_map[rmax],
                ls=ls_map[variant],
                lw=lw_map[variant],
                marker=marker_map[variant],
                ms=5,
            )

    ax.axhline(1, ls=':', lw=1, color='k')
    ax.set_xlabel(r'Mean $R_v$ [Mpc/$h$]')
    ax.set_title(panel_titles[rv_type])

axes[0].set_ylabel(r'$\chi^2_{\rm red}$')

# ── shared legend ─────────────────────────────────────────────────────────────

color_handles = [
    Line2D([0], [0], color=c, lw=2, label=f'$r_{{\\rm max}}={r}$')
    for r, c in color_map.items()
]
ls_handles = [
    Line2D([0], [0], color='k', ls=ls_map[v], lw=2,
           marker=marker_map[v], ms=5, label=label_variant[v])
    for v in ['', '_xi_vm', '_hres', '_2p8', '_xi_vm_2p8']
]

fig.legend(
    handles=color_handles + [Line2D([0],[0], lw=0)] + ls_handles,
    loc='lower center',
    ncol=len(color_handles) + 1 + len(ls_handles),
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.08),
)

plt.tight_layout()
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_fig}")
plt.show()

# ── plot 2: wl comparison ─────────────────────────────────────────────────────
# All files share: Rv_mean, rmax400, xi_vm_2p8 — only wl varies.

wl_values = [1401, 1501, 1601, 1701, 1801]
wl_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(wl_values)))
xvals_2p8 = rv_mean_x['2p8']

fig2, ax2 = plt.subplots(figsize=(7, 5))

for wl, color in zip(wl_values, wl_colors):
    fname = (f'{base_dir}chi2red_results_env_dep_analytical_mean_'
             f'Rv_mean_rmax400_xi_vm_2p8_wl{wl}_fullt.npz')
    try:
        data = np.load(fname, allow_pickle=True)
    except FileNotFoundError:
        print(f"Missing: {fname} — skipping")
        continue

    for mode in modes:
        try:
            chi2_red = data[f'{mode}_chi2_red']
        except KeyError:
            print(f"Mode '{mode}' not found in {fname} — skipping")
            continue

        ax2.plot(
            xvals_2p8, chi2_red,
            color=color,
            ls='-',
            lw=1.8,
            marker='D',
            ms=5,
            label=f'$w_l = {wl}$',
        )

ax2.set_ylim(0.0, 3.0)

ax2.axhline(1, ls=':', lw=1, color='k')
ax2.set_xlabel(r'Mean $R_v$ [Mpc/$h$]')
ax2.set_ylabel(r'$\chi^2_{\rm red}$')
ax2.set_title(r'$R_v$ mean, $r_{\rm max}=400$, $\xi_{vm}$ 2p8 — $WL$ comparison')
ax2.legend(frameon=False, fontsize=9, loc='best')

plt.tight_layout()
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_fig2}")
plt.show()