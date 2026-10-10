"""
Per-bond fabric-tensor / stretch analysis for a sheared particle network.

Background
----------
The "fabric tensor" R_ab = (1/N) * sum_contacts (n_a * n_b), where
where n = dr/|dr| the unit separation vector of each contact, is
a bulk, ensemble-averaged order parameter describing how contact orientations
are distributed:
  - isotropic network       -> off-diagonal terms (R_xy, R_xz, R_yz) ~ 0
  - shear-induced alignment -> R_xy grows

For an example of R_xy as a single scalar parameter for tracking microstructural 
reorganization under shear, correlated with the macroscopic stress, see:
   Jamali, McKinley & Armstrong (PRL 2017) 
   DOI: https://doi.org/10.1103/PhysRevLett.118.048003


Here we calucalte the per-particle affine elastic elongation of particle pairs
bonded at the reference frame (quiescent gel) over the whole shear trajectory
  - r_ij(t):                          current bond (center-center) length
  - strain_ij(t):                     (r_ij(t) - r_ij(t_ref)) / r_ij(t_ref)
  - fab_xx, fab_yy, fab_zz, fab_xy:   a bond's own contribution to the
                                        fabric tensor (its orientation bin)
  - typeid_i, typeid_j, tag_i, tag_j: track the specific particles involved

NOTE: formation and breaking of bonds is handled separately
"""



import numpy as np
import pandas as pd
import gsd.hoomd


def _min_image_dr(pos_i, pos_j, Lbox, m_xy):
    """Vectorized Lees-Edwards-aware minimum image displacement"""
    inv_Lbox = 1.0 / Lbox
    dr = pos_i - pos_j
    dr[:, 2] -= Lbox[2] * np.round(dr[:, 2] * inv_Lbox[2])
    img = Lbox[1] * np.round(dr[:, 1] * inv_Lbox[1])
    dr[:, 1] -= img
    dr[:, 0] -= img * m_xy
    dr[:, 0] -= Lbox[0] * np.round(dr[:, 0] * inv_Lbox[0])
    return dr


def find_contact_pairs(frame, index, radius, cut_off):
    """
    Build the (i, j) contact list from the reference frame 
    """
    pos = frame.particles.position[index]
    Lbox = frame.configuration.box[:3]
    m_xy = frame.configuration.box[3]

    n = len(index)
    ii, jj = np.triu_indices(n, k=1)
    dr = _min_image_dr(pos[ii].copy(), pos[jj].copy(), Lbox, m_xy)
    r = np.linalg.norm(dr, axis=1)
    gap = r - radius[ii] - radius[jj]
    mask = gap <= cut_off
    return ii[mask], jj[mask]


def bond_frame_metrics(frame, index, ii, jj, radius, r0=None, cut_off=None):
    """Vectorized per-bond metrics for one frame, for the fixed pair list ii, jj"""
    pos = frame.particles.position[index]
    Lbox = frame.configuration.box[:3]
    m_xy = frame.configuration.box[3]

    dr = _min_image_dr(pos[ii].copy(), pos[jj].copy(), Lbox, m_xy)
    r2 = np.einsum('ij,ij->i', dr, dr)
    r = np.sqrt(r2)

    n_hat = dr / r[:, None]
    fab_xy = n_hat[:, 0] * n_hat[:, 1]
    fab_xx = n_hat[:, 0] * n_hat[:, 0]
    fab_yy = n_hat[:, 1] * n_hat[:, 1]
    fab_zz = n_hat[:, 2] * n_hat[:, 2]

    gap = r - radius[ii] - radius[jj]

    out = {
        "r": r,
        "gap": gap,
        "fab_xx": fab_xx,
        "fab_yy": fab_yy,
        "fab_zz": fab_zz,
        "fab_xy": fab_xy,
    }
    if r0 is not None:
        out["strain"] = (r - r0) / r0
    if cut_off is not None:
        # IMPORTANT: 
        # this script tracks a FIXED pair list over their strain history
        # even through moments they may lose contact; therefore, summing
        # fab_xy over all rows will NOT reproduce the original Rxy_avg 
        # -- you would need to filter to in_contact rows first using this flag
        # lets you do that filtering.
        out["in_contact"] = gap <= cut_off
    return out


def fabric_tensor_per_bond(filename, colloid1_typeid, colloid2_typeid,
                            cut_off=0.3, ref_frame=0,
                            out_path="fabric_per_bond.parquet"):
    """
    Compute per-bond fabric-tensor components + strain for every frame,
    for every pair "in contact" (by cut_off) in the reference frame.
    Saves one row per (bond, frame) with typeid and tag labels to track
    populations and individual bonds across frames
    """
    traj = gsd.hoomd.open(filename, "r")
    nframe = len(traj)

    ref = traj[ref_frame]
    index = np.where((ref.particles.typeid == colloid1_typeid) |
                      (ref.particles.typeid == colloid2_typeid))[0]
    radius = 0.5 * ref.particles.diameter[index]
    typeid = ref.particles.typeid[index]
    tag = ref.particles.tag[index] if hasattr(ref.particles, "tag") else index.copy()

    # fixed contact list, defined once from the reference frame
    ii, jj = find_contact_pairs(ref, index, radius, cut_off)
    nbonds = len(ii)
    print(f"Tracking {nbonds} bonds across {nframe} frames.")

    bond_id = np.arange(nbonds)
    tag_i, tag_j = tag[ii], tag[jj]
    typeid_i, typeid_j = typeid[ii], typeid[jj]

    # reference bond length, used as the strain baseline
    ref_metrics = bond_frame_metrics(ref, index, ii, jj, radius, r0=None, cut_off=cut_off)
    r0 = ref_metrics["r"]

    rows = []
    for f in range(nframe):
        frame = traj[f]
        m = bond_frame_metrics(frame, index, ii, jj, radius, r0=r0, cut_off=cut_off)
        rows.append(pd.DataFrame({
            "frame": f,
            "bond_id": bond_id,
            "tag_i": tag_i,
            "tag_j": tag_j,
            "typeid_i": typeid_i,
            "typeid_j": typeid_j,
            "r": m["r"],
            "gap": m["gap"],
            "strain": m["strain"],
            "in_contact": m["in_contact"],
            "fab_xx": m["fab_xx"],
            "fab_yy": m["fab_yy"],
            "fab_zz": m["fab_zz"],
            "fab_xy": m["fab_xy"],
        }))

    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(out_path, index=False)
    print(f"Saved per-bond fabric/strain data to {out_path}")

    # Sanity check: sum fab_xy over bonds that are still in_contact;
    # this should be close to Rxy_avg from the current fabric tensor, tho
    # it is missing any contacts that formed since the reference frame
    active = result[result["in_contact"]]
    check = active.groupby("frame")["fab_xy"].sum() / len(index)
    check.to_csv(out_path.replace(".parquet", "_Rxy_avg_check.csv"))

    return result


if __name__ == "__main__":
    kappa = 10
    fabric_tensor_per_bond(
        filename="../sim-scripts/bi-colloid_50-50/potential-morse-bysize/12kT/kappa10/shear-const-0.5-allinit-bi-DPD/Shear-const-SR0.5-1strains-vinit0.5-bi-DPD.gsd",
        colloid1_typeid=1, 
        colloid2_typeid=2,
        cut_off=3/kappa,
        ref_frame=0,
        out_path="fabric_per_bond.parquet",
    )
