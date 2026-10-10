"""
Falk-Langer D2min 
  1. freud for neighbor search (C++, ~O(N), triclinic box handles the xy tilt)
  2. Fully vectorized D2min: per-particle Y/X accumulated with np.add.at, the
     whole stack of 3x3s pseudo-inverted at once, residuals via one einsum.
     No python per-particle loop.
  3. One sequential GSD pass into arrays up front, then intervals (which are
     independent) run in parallel across cores with joblib.
"""

import numpy as np
import pandas as pd
import gsd.hoomd
import freud
from joblib import Parallel, delayed

# ------------------------- config -------------------------
GSD = "../sim-scripts/bi-colloid_50-50/potential-morse-bysize/12kT/kappa10/shear-const-0.5-allinit-bi-DPD/Shear-const-SR0.5-1strains-vinit0.5-bi-DPD.gsd"
COLLOID_TYPEIDS = (1, 2)
kappa = 10
NEIGH_CUTOFF = 3/kappa
DFRAME = 3 #50 #5
STEP = 5
N_JOBS = -1 # all cores 
OUT_PARTICLE = "d2min_per_particle.parquet"
OUT_SUMMARY = "d2min_summary.csv"
# ----------------------------------------------------------


def min_image_dr(ri, rj, Lbox, m_xy):
    """Vectorized Lees-Edwards min image for exact dr / gap on candidate pairs."""
    inv = 1.0 / Lbox
    dr = ri - rj
    dr[:, 2] -= Lbox[2] * np.round(dr[:, 2] * inv[2])
    img = Lbox[1] * np.round(dr[:, 1] * inv[1])
    dr[:, 1] -= img
    dr[:, 0] -= img * m_xy
    dr[:, 0] -= Lbox[0] * np.round(dr[:, 0] * inv[0])
    return dr


def d2min_interval(pos0, pos1, box0_arr, box1_arr, radius, cutoff):
    """D2min for every particle over one interval. Fully vectorized."""
    N = len(pos0)
    L0 = np.array(box0_arr[:3]); mxy0 = box0_arr[3]
    L1 = np.array(box1_arr[:3]); mxy1 = box1_arr[3]

    # --- freud neighbor search at t0 (triclinic box handles the tilt) ---
    r_max = cutoff + 2.0 * radius.max()
    box = freud.box.Box.from_box(box0_arr)
    aq = freud.locality.AABBQuery(box, pos0.astype(np.float32))
    nl = aq.query(pos0.astype(np.float32),
                  dict(r_max=r_max, exclude_ii=True)).toNeighborList()
    i_idx = np.asarray(nl.query_point_indices)
    j_idx = np.asarray(nl.point_indices)

    # --- exact dr / gap filter to the surface-gap cutoff ---
    d0 = min_image_dr(pos0[i_idx].astype(float), pos0[j_idx].astype(float), L0, mxy0)
    r0 = np.sqrt(np.einsum("pi,pi->p", d0, d0))
    gap0 = r0 - radius[i_idx] - radius[j_idx]
    keep = gap0 <= cutoff
    i_idx, j_idx, d0 = i_idx[keep], j_idx[keep], d0[keep]
    d1 = min_image_dr(pos1[i_idx].astype(float), pos1[j_idx].astype(float), L1, mxy1)

    # --- accumulate per-particle Y = sum d0 d0^T, X = sum d1 d0^T ---
    Y = np.zeros((N, 3, 3)); X = np.zeros((N, 3, 3))
    np.add.at(Y, i_idx, d0[:, :, None] * d0[:, None, :])
    np.add.at(X, i_idx, d1[:, :, None] * d0[:, None, :])
    count = np.bincount(i_idx, minlength=N)
    valid = count >= 3

    # --- batched best-fit affine J = X Y^{-1} (pinv is robust to coplanar) ---
    J = np.zeros((N, 3, 3))
    J[valid] = X[valid] @ np.linalg.pinv(Y[valid])

    # --- residuals per pair, accumulated per particle ---
    Jd0 = np.einsum("pij,pj->pi", J[i_idx], d0)
    sq = np.einsum("pi,pi->p", d1 - Jd0, d1 - Jd0)
    d2sum = np.bincount(i_idx, weights=sq, minlength=N)
    d2 = np.where(valid, d2sum / np.maximum(count, 1), np.nan)
    return d2


def main():
    traj = gsd.hoomd.open(GSD, "r")
    nframe = len(traj)
    ref = traj[0]
    index = np.where(np.isin(ref.particles.typeid, COLLOID_TYPEIDS))[0]
    radius = 0.5 * ref.particles.diameter[index]
    typeid = ref.particles.typeid[index]

    # one sequential GSD pass into arrays
    positions = np.empty((nframe, len(index), 3), dtype=np.float32)
    boxes = np.empty((nframe, 6), dtype=float)
    for f in range(nframe):
        fr = traj[f]
        positions[f] = fr.particles.position[index]
        boxes[f] = fr.configuration.box
    gamma = boxes[:, 3].copy()
    if np.allclose(gamma, 0):
        print("box tilt ~0 for all frames; using frame index as strain axis.")
        gamma = np.arange(nframe, dtype=float)

    starts = list(range(0, nframe - DFRAME, STEP))

    def one(t0):
        d2 = d2min_interval(positions[t0], positions[t0 + DFRAME],
                            boxes[t0], boxes[t0 + DFRAME], radius, NEIGH_CUTOFF)
        mid_g = 0.5 * (gamma[t0] + gamma[t0 + DFRAME])
        return t0, mid_g, d2

    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(t0) for t0 in starts)

    part_frames, summary = [], []
    for t0, mid_g, d2 in sorted(results):
        valid = ~np.isnan(d2)
        summary.append({"t0": t0, "gamma": mid_g,
                        "mean_d2min": np.nanmean(d2),
                        "p90_d2min": np.nanpercentile(d2, 90),
                        "n_valid": int(valid.sum())})
        part_frames.append(pd.DataFrame({
            "t0": t0, "gamma": mid_g, "particle": index[valid],
            "typeid": typeid[valid], "d2min": d2[valid],
            "x": positions[t0][valid, 0], "y": positions[t0][valid, 1]}))

    pd.concat(part_frames, ignore_index=True).to_parquet(OUT_PARTICLE, index=False)
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_PARTICLE} and {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
