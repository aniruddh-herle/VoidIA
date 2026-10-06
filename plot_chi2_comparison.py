import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── config ────────────────────────────────────────────────────────────────────

base_dir   = '/disks/shear16/herle/code/temp_data/'
output_fig = '/disks/shear16/herle/code/plots/chi2red_comp_1gpc.png'

modes = ['FIDUCIAL']

bin_display = {
    '0_10':  r'$0$–$10$',
    '10_20': r'$10$–$20$',
    '20_30': r'$20$–$30$',
}

# ── style maps ────────────────────────────────────────────────────────────────
# colour   → rmax  (well-separated)
# linestyle → variant

color_map = {
    100: '#1f77b4',   # blue
    200: '#ff7f0e',   # orange
    300: '#2ca02c',   # green
    400: '#d62728',   # red
}
ls_map     = {'': '-', '_xi_vm': '--', '_hres': ':'}
lw_map     = {'': 1.8, '_xi_vm': 1.8, '_hres': 2.2}
marker_map = {'': 'o', '_xi_vm': 's', '_hres': '^'}

label_variant = {'': 'base', '_xi_vm': r'$\xi_{vm}$', '_hres': 'hres'}

# ── files grouped by rv_type ──────────────────────────────────────────────────

groups = {
    'Rv_mean': [
        (100, ''), (100, '_xi_vm'),
        (200, ''), (200, '_xi_vm'),
        (300, ''), (300, '_xi_vm'), (300, '_hres'),
        (400, ''), (400, '_xi_vm'), (400, '_hres'),
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

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, (rv_type, entries) in zip(axes, groups.items()):

    x_pos    = None
    x_labels = None

    for rmax, variant in entries:

        suffix = f'{rv_type}_rmax{rmax}{variant}'
        fname  = f'{base_dir}chi2red_results_env_dep_analytical_mean_{suffix}.npz'

        try:
            data = np.load(fname, allow_pickle=True)
        except FileNotFoundError:
            print(f"Missing: {fname} — skipping")
            continue

        for mode in modes:
            try:
                bin_labels = data[f'{mode}_bin_labels']
                chi2_red   = data[f'{mode}_chi2_red']
            except KeyError:
                print(f"Mode '{mode}' not found in {fname} — skipping")
                continue

            if x_pos is None:
                x_pos    = np.arange(len(bin_labels))
                x_labels = [bin_display.get(bl, bl) for bl in bin_labels]

            ax.plot(
                x_pos, chi2_red,
                color=color_map[rmax],
                ls=ls_map[variant],
                lw=lw_map[variant],
                marker=marker_map[variant],
                ms=5,
            )

    if x_pos is not None:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)

    ax.axhline(1, ls=':', lw=1, color='k')
    ax.set_xlabel(r'$R_v$ bin [Mpc/$h$]')
    ax.set_title(panel_titles[rv_type])

axes[0].set_ylabel(r'$\chi^2_{\rm red}$')

# ── shared legend ─────────────────────────────────────────────────────────────

# colour legend (rmax)
color_handles = [
    Line2D([0], [0], color=c, lw=2, label=f'$r_{{\\rm max}}={r}$')
    for r, c in color_map.items()
]
# linestyle legend (variant)
ls_handles = [
    Line2D([0], [0], color='k', ls=ls_map[v], lw=2,
           marker=marker_map[v], ms=5, label=label_variant[v])
    for v in ['', '_xi_vm', '_hres']
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