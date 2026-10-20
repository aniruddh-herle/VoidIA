import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import glob
import os
import pandas as pd
from mnras import *
from scipy.interpolate import interp1d

h = 0.681

# ── font sizes ────────────────────────────────────────────────────────────────
FS_LABEL  = 15    # axis labels  (xlabel / ylabel)
FS_TITLE  = 15    # panel titles
FS_TICK   = 15    # tick-mark labels
FS_LEGEND = 15    # legend text

plt.rcParams.update({
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def get_radius_from_name(fname):
    base = os.path.basename(fname)
    return float(base.split("Rv_")[1].split("_")[0])

def stack_stats_sem(arr):
    """Mean profile with SEM error bars."""
    arr   = np.array(arr)
    Nv    = arr.shape[0]
    mean  = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0, ddof=1)
    sem   = sigma / np.sqrt(Nv)
    return mean, mean - sem, mean + sem

def j0(x):
    out = np.ones_like(x)
    mask = x != 0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def hankel_transform(r, delta_r, k_vals):
    dr        = np.gradient(r)
    kr        = np.outer(k_vals, r)
    integrand = j0(kr) * (r**2 * dr * delta_r)
    return 4 * np.pi * integrand.sum(axis=1)

# ── settings ──────────────────────────────────────────────────────────────────

run           = 'L2800N5040'
mode          = 'FIDUCIAL'
redshift      = '0.0'
mean_density  = 1.846582e-01

corr_dir    = '/disks/shear16/herle/correlations1/'
temp_dir    = '/disks/shear16/herle/code/temp_data/'
plots_dir   = '/disks/shear16/herle/code/plots/'
profile_dir = f"/disks/shear16/herle/void_profiles/zobov_hydro_particles/{mode}/"

bin_labels = ['0_10', '10_15', '15_20', '20_25', '25_30', '30_50']
bin_edges  = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
bin_titles = ["0–10 Mpc/h", "10–15 Mpc/h", "15–20 Mpc/h",
              "20–25 Mpc/h", "25–30 Mpc/h", "30–50 Mpc/h"]
shell_keys = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5']

n_bins = len(bin_labels)

xi_suffix  = '_env_dep_analytical_morerp'

log10kmin  = -5
log10kmax  = 2
k_native   = np.logspace(log10kmin, log10kmax, 1000001)
k          = k_native * h
rmax_trunc = 400.0

col_shell = 'C0'
col_xi    = 'C2'

# ── cache control ─────────────────────────────────────────────────────────────
# Set RECOMPUTE = True to force a fresh computation from raw files, e.g. when
# bin_edges, mean_density, xi_suffix, or rmax_trunc change.
# Otherwise the pre-computed cache in temp_dir is reloaded instantly.

cache_file = os.path.join(temp_dir, f'zobov_{mode}_combined_plot_cache.npz')
RECOMPUTE  = False

# ── data I/O ──────────────────────────────────────────────────────────────────

def _load_xi_vm(bin_label):
    meas_file = (
        f'{corr_dir}correlations_joint_revolver_{bin_label}hMpc_'
        f'{run}_HYDRO_{mode}_{redshift}{xi_suffix}_fullt.h5'
    )
    jk_file = (
        f'{corr_dir}xi_vm_jk_voids_joint_revolver_{bin_label}hMpc_'
        f'{run}_HYDRO_{mode}_{redshift}{xi_suffix}_fullt.npy'
    )
    jk       = np.load(jk_file)
    NPatches = jk.shape[0]
    xi       = np.mean(jk, axis=0)
    diff     = jk - xi
    cov      = ((NPatches - 1) / NPatches) * (diff.T @ diff)
    err      = np.sqrt(np.diag(cov))
    df = pd.read_hdf(meas_file, key='a')
    if 'xi_vm_r' in df.columns:
        r = df['xi_vm_r'].values
    else:
        r_cols = [c for c in df.columns if 'xi_vm' in c and '_r' in c]
        if r_cols:
            r = df[r_cols[0]].values
        else:
            raise KeyError(f"Cannot find r-column in {meas_file}. "
                           f"Available: {list(df.columns)}")
    return r, xi, err


def compute_all():
    """Load raw profiles and compute Hankel transforms; return all plot data."""
    print("Computing from raw files …")

    # ── shell profiles ────────────────────────────────────────────────────────
    files  = sorted(glob.glob(os.path.join(profile_dir, "*.npz")))
    r_phys = None
    bins   = [[] for _ in range(n_bins)]
    for f in files:
        d        = np.load(f)
        r_phys   = d["r_phys"]
        rho_phys = d["density_phys"]
        delta    = rho_phys / mean_density - 1
        Rv       = get_radius_from_name(f)
        for b, (lo, hi) in enumerate(bin_edges):
            if lo <= Rv < hi:
                bins[b].append(delta)

    bin_stats = [stack_stats_sem(b) for b in bins]
    bin_means = [s[0] for s in bin_stats]
    bin_los   = [s[1] for s in bin_stats]
    bin_his   = [s[2] for s in bin_stats]
    bin_Nvs   = [len(b) for b in bins]

    # ── xi_vm correlations + Hankel transforms ────────────────────────────────
    xi_rs, xi_means, xi_errs, dk_xis, xi_ok = [], [], [], [], []
    for bl in bin_labels:
        try:
            print(f"  Hankel transform for bin {bl} …")
            r, xi, err = _load_xi_vm(bl)
            r_fine  = np.logspace(np.log10(r.min()), np.log10(r.max()), 2000)
            xi_fine = interp1d(r, xi, kind='cubic')(r_fine)
            mask    = r_fine <= rmax_trunc
            dk_xi   = hankel_transform(r_fine[mask], xi_fine[mask], k_native) / h**3
            xi_rs.append(r);       xi_means.append(xi);  xi_errs.append(err)
            dk_xis.append(dk_xi); xi_ok.append(True)
        except Exception as e:
            print(f"  Could not load xi_vm for bin {bl}: {e}")
            xi_rs.append(None); xi_means.append(None)
            xi_errs.append(None); dk_xis.append(None); xi_ok.append(False)

    # ── write cache ───────────────────────────────────────────────────────────
    os.makedirs(temp_dir, exist_ok=True)
    save_dict = {'r_phys': r_phys}
    for b in range(n_bins):
        save_dict[f'bin_mean_{b}'] = bin_means[b]
        save_dict[f'bin_lo_{b}']   = bin_los[b]
        save_dict[f'bin_hi_{b}']   = bin_his[b]
        save_dict[f'bin_Nv_{b}']   = bin_Nvs[b]
        save_dict[f'xi_ok_{b}']    = xi_ok[b]
        if xi_ok[b]:
            save_dict[f'xi_r_{b}']    = xi_rs[b]
            save_dict[f'xi_mean_{b}'] = xi_means[b]
            save_dict[f'xi_err_{b}']  = xi_errs[b]
            save_dict[f'dk_xi_{b}']   = dk_xis[b]
    np.savez_compressed(cache_file, **save_dict)
    print(f"Cache saved → {cache_file}")
    return r_phys, bin_means, bin_los, bin_his, bin_Nvs, xi_rs, xi_means, xi_errs, dk_xis, xi_ok


def load_cache():
    """Reload plot data from the pre-computed cache."""
    print(f"Loading from cache: {cache_file}")
    cache = np.load(cache_file, allow_pickle=True)
    r_phys = cache['r_phys']
    bin_means, bin_los, bin_his, bin_Nvs = [], [], [], []
    xi_rs, xi_means, xi_errs, dk_xis, xi_ok = [], [], [], [], []
    for b in range(n_bins):
        bin_means.append(cache[f'bin_mean_{b}'])
        bin_los.append(cache[f'bin_lo_{b}'])
        bin_his.append(cache[f'bin_hi_{b}'])
        bin_Nvs.append(int(cache[f'bin_Nv_{b}']))
        ok = bool(cache[f'xi_ok_{b}'])
        xi_ok.append(ok)
        if ok:
            xi_rs.append(cache[f'xi_r_{b}'])
            xi_means.append(cache[f'xi_mean_{b}'])
            xi_errs.append(cache[f'xi_err_{b}'])
            dk_xis.append(cache[f'dk_xi_{b}'])
        else:
            xi_rs.append(None); xi_means.append(None)
            xi_errs.append(None); dk_xis.append(None)
    return r_phys, bin_means, bin_los, bin_his, bin_Nvs, xi_rs, xi_means, xi_errs, dk_xis, xi_ok


# ── dispatch ──────────────────────────────────────────────────────────────────
if not RECOMPUTE and os.path.exists(cache_file):
    (r_phys, bin_means, bin_los, bin_his, bin_Nvs,
     xi_rs, xi_means, xi_errs, dk_xis, xi_ok) = load_cache()
else:
    (r_phys, bin_means, bin_los, bin_his, bin_Nvs,
     xi_rs, xi_means, xi_errs, dk_xis, xi_ok) = compute_all()


def load_hankel_shell(b):
    """Load the pre-saved shell Hankel transform from temp_dir."""
    path = os.path.join(
        temp_dir,
        f'zobov_FIDUCIAL_2p8_void_delta_k_{shell_keys[b]}'
        f'_raw_particles_rmax400_mean.npy'
    )
    return np.load(path)


# ── combined full-page figure ─────────────────────────────────────────────────
#
#  Layout (2 columns, n_pairs outer rows):
#
#  ┌─────────────────────┬─────────────────────┐
#  │  bin 0  Δ_v(r)      │  bin 1  Δ_v(r)      │  ← inner row 0 (profile)
#  ├─────────────────────┼─────────────────────┤
#  │  bin 0  Δ_v(k)      │  bin 1  Δ_v(k)      │  ← inner row 1 (Hankel)
#  ╠═════════════════════╪═════════════════════╡  ← gap between outer rows
#  │  bin 2  Δ_v(r)      │  bin 3  Δ_v(r)      │
#  ├─────────────────────┼─────────────────────┤
#  │  bin 2  Δ_v(k)      │  bin 3  Δ_v(k)      │
#  ╠═════════════════════╪═════════════════════╡
#  │  bin 4  Δ_v(r)      │  bin 5  Δ_v(r)      │
#  ├─────────────────────┼─────────────────────┤
#  │  bin 4  Δ_v(k)      │  bin 5  Δ_v(k)      │
#  └─────────────────────┴─────────────────────┘

ncols   = 2
n_pairs = int(np.ceil(n_bins / ncols))    # 3 for 6 bins

fig = plt.figure(figsize=(15, 20))

# Outer GridSpec: manual margins so tight_layout never overrides them.
outer_gs = GridSpec(
    n_pairs, 1, figure=fig,
    hspace=0.08,
    left=0.10, right=0.97, top=0.97, bottom=0.03
)

ax_prof = {}   # (pair, col) → Axes
ax_hank = {}

for pair in range(n_pairs):
    # hspace=0.0 → profile and Hankel panels are flush.
    # The profile x-label is moved to the TOP of the profile panel so no
    # text hangs below it into the gap.
    inner_gs = GridSpecFromSubplotSpec(
        2, ncols,
        subplot_spec=outer_gs[pair],
        height_ratios=[1.1, 1.0],
        hspace=0.0,
        wspace=0.38
    )
    for col in range(ncols):
        ax_prof[(pair, col)] = fig.add_subplot(inner_gs[0, col])
        ax_hank[(pair, col)] = fig.add_subplot(inner_gs[1, col])


for b in range(n_bins):
    pair = b // ncols
    col  = b % ncols
    ap   = ax_prof[(pair, col)]
    ak   = ax_hank[(pair, col)]

    # ── real-space profile ────────────────────────────────────────────────────
    Nv = bin_Nvs[b]
    ap.plot(r_phys, bin_means[b], color=col_shell, lw=1.5,
            label=rf"$\Delta_v(r)$, $N_v={Nv}$")
    ap.fill_between(r_phys, bin_los[b], bin_his[b],
                    alpha=0.25, color=col_shell)
    if xi_ok[b]:
        ap.errorbar(xi_rs[b], xi_means[b], yerr=xi_errs[b],
                    fmt='s', ms=3, capsize=2, alpha=0.8,
                    color=col_xi, label=r"$\xi_{vm}$")

    ap.axhline(0, ls='--', color='k', lw=0.8)
    ap.set_xscale('log')
    ap.set_ylabel(r"$\xi_{vm}(r)$", fontsize=FS_LABEL)
    ap.set_title(bin_titles[b], fontsize=FS_TITLE, pad=3)
    ap.legend(frameon=False, fontsize=FS_LEGEND)
    correct_tick_marks(ap)
    # Move x-axis to the top of the profile panel so the bottom edge is clean
    # and the profile+Hankel pair can sit flush (hspace=0).
    ap.xaxis.set_label_position('top')
    ap.xaxis.tick_top()
    ap.set_xlabel(r"$r\;[\mathrm{Mpc}/h]$", fontsize=FS_LABEL, labelpad=4)

    # ── Hankel / k-space panel ────────────────────────────────────────────────
    try:
        dk_shell = load_hankel_shell(b)
        ak.plot(k, dk_shell / 1e3, color=col_shell, lw=1.5, ls='--',
                label=r"$\Delta_v(k)$ shells")
    except Exception as e:
        print(f"No shell Hankel for bin {b}: {e}")

    if xi_ok[b] and dk_xis[b] is not None:
        ak.plot(k, dk_xis[b] / 1e3, color=col_xi, lw=1.5,
                label=r"$\xi_{vm}$")

    ak.axhline(0, ls=':', color='k', lw=0.8)
    ak.set_xscale('log')
    ak.set_xlim(1e-5, 1e3)
    ak.set_xlabel(r"$k\;[\mathrm{Mpc}^{-1}]$", fontsize=FS_LABEL)
    ak.set_ylabel(r"$\Delta_v(k)\times 10^3$", fontsize=FS_LABEL)
    ak.legend(frameon=False, fontsize=FS_LEGEND)
    correct_tick_marks(ak)
    # Remove top ticks on Hankel panel — the flush border with the profile
    # panel above would otherwise show a double row of tick marks.
    ak.tick_params(top=False, which='both')


# hide any unused panels (odd n_bins)
for b in range(n_bins, n_pairs * ncols):
    pair = b // ncols
    col  = b % ncols
    ax_prof[(pair, col)].set_visible(False)
    ax_hank[(pair, col)].set_visible(False)


# ── save ──────────────────────────────────────────────────────────────────────
# No tight_layout — it would override the manual GridSpec margins and hspace
# values set above, re-introducing the excessive whitespace.
os.makedirs(plots_dir, exist_ok=True)
stem    = f'zobov_{mode}_profiles_hankel_combined'
out_pdf = os.path.join(plots_dir, stem + '.pdf')
out_png = os.path.join(plots_dir, stem + '.png')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved:\n  {out_pdf}\n  {out_png}")