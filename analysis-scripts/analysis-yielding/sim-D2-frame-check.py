"""
Measure the thermal D2min floor and compare it to the yield-peak D2min, to
choose a valid value of DFRAME by signal-to-noise rather than by frame count.
----
D2min removes the local affine deformation, so in a window with little genuine
rearrangement (quiescent / pre-yield, gamma << overshoot) the D2min that remains
is the thermal contribution -- any per-particle jiggle an affine fit cannot absorb.
This the system's noise floor.

For each candidate DFRAME we compute median D2min in a QUIESCENT window (floor)
and in the YIELD window (signal), per size class, and check for peak/floor >~ 2-3. 
If a DFRAME gives ~1 it is noise-limited and should be increased
"""

import numpy as np
import pandas as pd
import gsd.hoomd
import freud

GSD = "../Shear-const-SR0.5-1strains-vinit0.5-bi-DPD.gsd"
COLLOID_TYPEIDS = (1, 2)
kappa = 10
NEIGH_CUTOFF = 3 / kappa
FRAMES_PER_STRAIN = 10000
CANDIDATE_DFRAME = [5, 20, 50, 100, 200]
# windows given as (start_frame, end_frame); set from the gamma/overshoot.
# quiescent: earliest frames, well below the strain peak
# yield:     brackets the overshoot peak
QUIESCENT_WINDOW = (0, 50)
YIELD_WINDOW = (150, 200) # ex: (frame_at_overshoot-100, frame_at_overshoot+100)
N_SAMPLE = 5              # number of intervals to sample per window
# --------------------------------------------------------------------


def min_image_dr(ri, rj, Lbox, m_xy):
    inv = 1.0 / Lbox
    dr = ri - rj
    dr[:, 2] -= Lbox[2] * np.round(dr[:, 2] * inv[2])
    img = Lbox[1] * np.round(dr[:, 1] * inv[1])
    dr[:, 1] -= img
    dr[:, 0] -= img * m_xy
    dr[:, 0] -= Lbox[0] * np.round(dr[:, 0] * inv[0])
    return dr


def d2min_interval(pos0, pos1, box0, box1, radius, cutoff):
    N = len(pos0)
    L0 = np.array(box0[:3]); mxy0 = box0[3]
    L1 = np.array(box1[:3]); mxy1 = box1[3]
    r_max = cutoff + 2.0 * radius.max()
    box = freud.box.Box.from_box(box0)
    nl = freud.locality.AABBQuery(box, pos0.astype(np.float32)).query(
        pos0.astype(np.float32), dict(r_max=r_max, exclude_ii=True)).toNeighborList()
    i_idx = np.asarray(nl.query_point_indices); j_idx = np.asarray(nl.point_indices)
    d0 = min_image_dr(pos0[i_idx].astype(float), pos0[j_idx].astype(float), L0, mxy0)
    gap0 = np.sqrt(np.einsum("pi,pi->p", d0, d0)) - radius[i_idx] - radius[j_idx]
    keep = gap0 <= cutoff
    i_idx, j_idx, d0 = i_idx[keep], j_idx[keep], d0[keep]
    d1 = min_image_dr(pos1[i_idx].astype(float), pos1[j_idx].astype(float), L1, mxy1)
    Y = np.zeros((N, 3, 3)); X = np.zeros((N, 3, 3))
    np.add.at(Y, i_idx, d0[:, :, None] * d0[:, None, :])
    np.add.at(X, i_idx, d1[:, :, None] * d0[:, None, :])
    count = np.bincount(i_idx, minlength=N); valid = count >= 3
    J = np.zeros((N, 3, 3)); J[valid] = X[valid] @ np.linalg.pinv(Y[valid])
    Jd0 = np.einsum("pij,pj->pi", J[i_idx], d0)
    sq = np.einsum("pi,pi->p", d1 - Jd0, d1 - Jd0)
    d2sum = np.bincount(i_idx, weights=sq, minlength=N)
    return np.where(valid, d2sum / np.maximum(count, 1), np.nan)


def median_d2_in_window(positions, boxes, radius, typeid, window, dframe, n_sample):
    lo, hi = window
    starts = np.linspace(lo, hi - dframe, n_sample).astype(int)
    starts = starts[starts >= 0]
    per_type = {t: [] for t in np.unique(typeid)}
    allv = []
    for t0 in starts:
        d2 = d2min_interval(positions[t0], positions[t0 + dframe],
                            boxes[t0], boxes[t0 + dframe], radius, NEIGH_CUTOFF)
        for t in per_type:
            per_type[t].append(np.nanmedian(d2[typeid == t]))
        allv.append(np.nanmedian(d2))
    return np.nanmedian(allv), {t: np.nanmedian(v) for t, v in per_type.items()}


def main():
    traj = gsd.hoomd.open(GSD, "r")
    nframe = len(traj)
    ref = traj[0]
    index = np.where(np.isin(ref.particles.typeid, COLLOID_TYPEIDS))[0]
    radius = 0.5 * ref.particles.diameter[index]
    typeid = ref.particles.typeid[index]

    max_d = max(CANDIDATE_DFRAME)
    need = max(QUIESCENT_WINDOW[1], (YIELD_WINDOW[1] if YIELD_WINDOW else 0)) + max_d + 1
    positions = np.empty((min(need, nframe), len(index), 3), dtype=np.float32)
    boxes = np.empty((min(need, nframe), 6), dtype=float)
    for f in range(len(positions)):
        fr = traj[f]
        positions[f] = fr.particles.position[index]
        boxes[f] = fr.configuration.box

    print(f"{'DFRAME':>7} {'d_gamma':>9} {'floor':>11} {'peak':>11} {'peak/floor':>11}")
    for dframe in CANDIDATE_DFRAME:
        dgamma = dframe / FRAMES_PER_STRAIN
        floor, floor_by = median_d2_in_window(positions, boxes, radius, typeid,
                                              QUIESCENT_WINDOW, dframe, N_SAMPLE)
        if YIELD_WINDOW is not None:
            peak, peak_by = median_d2_in_window(positions, boxes, radius, typeid,
                                               YIELD_WINDOW, dframe, N_SAMPLE)
            ratio = peak / floor
            print(f"{dframe:>7} {dgamma:>9.2e} {floor:>11.3e} {peak:>11.3e} {ratio:>11.2f}")
        else:
            print(f"{dframe:>7} {dgamma:>9.2e} {floor:>11.3e} {'(set YIELD_WINDOW)':>23}")
        # per-size floor, to see the size dependence of the noise itself
        print("         floor by typeid:",
              {int(t): float(f"{v:.2e}") for t, v in floor_by.items()})


if __name__ == "__main__":
    main()
