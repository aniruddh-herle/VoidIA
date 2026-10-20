from scipy.spatial import KDTree
import numpy as np
import os

# =============================================================
# Mean inter-tracer distance via KD-tree nearest neighbour
# =============================================================

tracer_files = [
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_FIDUCIAL_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_JETS_published_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_STRONG_AGN_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_STRONGER_AGN_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_STRONGER_AGN_STRONG_SUPERNOVA_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_STRONGEST_AGN_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_STRONG_JETS_published_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_STRONG_SUPERNOVA_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L1000N1800_HYDRO_WEAK_AGN_0.0_tracer_positions.npy',
    '/disks/shear16/herle/shape_catalogues/HBT/new/mass_cut_10/L2800N5040_HYDRO_FIDUCIAL_0.0_tracer_positions.npy',
]

def box_size_from_filename(fname):
    base = os.path.basename(fname)
    if   base.startswith('L1000'): return 1000
    elif base.startswith('L2800'): return 2800
    else: raise ValueError(f"Cannot infer box size from: {fname}")

def short_label(fname):
    """Strip path and suffix to get a readable sim name."""
    return (os.path.basename(fname)
            .replace('_0.0_tracer_positions.npy', '')
            .replace('_HYDRO', '')
            .replace('_published', ''))

# For very large catalogs (L2800 has ~tens of millions of points)
# querying every point is slow; subsample for the NN statistics.
NN_SUBSAMPLE = 200_000   # set to None to query all points

h = 0.681

print("=" * 70)
print("MEAN INTER-TRACER DISTANCE  (KD-tree nearest neighbour)")
print("=" * 70)

for fpath in tracer_files:
    label = short_label(fpath)
    L     = box_size_from_filename(fpath)

    pos = np.load(fpath)          # shape (N, 3), units Mpc/h
    N   = len(pos)
    V = (float(L) * h) ** 3  # (Mpc/h)^3

    # --- analytic estimate (assumes uniform Poisson field) ---
    n      = N / V
    d_anal = n ** (-1.0 / 3.0)   # Mpc/h

    # --- KD-tree: query a subsample for speed on huge catalogs ---
    tree = KDTree(pos)

    if NN_SUBSAMPLE is not None and N > NN_SUBSAMPLE:
        rng     = np.random.default_rng(42)
        idx     = rng.choice(N, size=NN_SUBSAMPLE, replace=False)
        query_pos = pos[idx]
        note    = f"(NN stats from {NN_SUBSAMPLE:,} random tracers)"
    else:
        query_pos = pos
        note    = "(all tracers)"

    # k=2: index 0 is the point itself (dist=0), index 1 is the NN
    dd, _   = tree.query(query_pos, k=2, workers=-1)
    nn_dist = dd[:, 1]            # Mpc/h

    print(f"\n  {label}")
    print(f"    Box  L = {L} Mpc/h   |   N tracers = {N:,}")
    print(f"    Analytic  d̄  (n^-1/3)  = {d_anal:.4f}  Mpc/h")
    print(f"    NN mean                 = {nn_dist.mean():.4f}  Mpc/h  {note}")
    print(f"    NN median               = {np.median(nn_dist):.4f}  Mpc/h")
    print(f"    NN std                  = {nn_dist.std():.4f}  Mpc/h")

print("\nDone.")