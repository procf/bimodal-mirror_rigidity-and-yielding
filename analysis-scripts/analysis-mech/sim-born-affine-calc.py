"""
Affine bond-by-bond G' calculation for HOOMD-blue GSD trajectories.

Computes the Born (affine) contribution to the shear modulus G_xy by summing
over every bond in the gel network. The Born term is an upper bound on the 
true G' because it ignores nonaffine relaxations, but the bond-level 
decomposition estimates relative contributions to elasticity.

For a central-force pair potential U(r), the Born contribution of bond ij to
the elastic constant C_xyxy under affine xy-shear is:

    C^born_ij = (1/V) * [k(r_ij) - F(r_ij)/r_ij] * (n_x * n_y)^2 * r_ij^2

where:
    k(r) = U''(r) is the radial stiffness
    F(r) = -U'(r) is the central force attraction (positive = repulsive)
    n_x, n_y are components of the bond unit vector

For Morse potential U(r) = U0 * [(1 - exp(-kappa*(r-r0)))^2 - 1]:
    U'(r)  = 2*U0*kappa * exp(-kappa*(r-r0)) * (1 - exp(-kappa*(r-r0)))
    U''(r) = 2*U0*kappa^2 * exp(-kappa*(r-r0)) * (2*exp(-kappa*(r-r0)) - 1)
"""

import numpy as np
import pandas as pd
import gsd.hoomd
import sys
import os


# create "born-Gprime" subfolder if it doesn't exit
data_outpath = "born-Gprime"
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)



def load_bonds_from_psf(psf_path):
    """
    Load bond connectivity from a CHARMM PSF file.
    
    PSF files have sections marked with header lines like:
        !NATOM
        !NBOND: bonds
        !NTHETA: angles
        ...
    
    The !NBOND section lists bonds as pairs of atom indices (1-indexed in PSF).
    Bond pairs are typically grouped 4 per line (8 numbers per line), but
    formatting can vary.
    
    Returns:
        bonds: (M, 2) numpy array of 0-indexed particle indices
        n_particles: number of particles (from !NATOM line)
    """
    with open(psf_path, 'r') as f:
        lines = f.readlines()
    
    bonds = []
    n_particles = 0
    
    # Parse line by line, looking for section headers
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # NATOM section header
        if '!NATOM' in line:
            # The number of atoms is typically the first integer on this line
            parts = line.split()
            for part in parts:
                try:
                    n_particles = int(part)
                    break
                except ValueError:
                    continue
            # Skip past the atom records (one per particle)
            i += 1 + n_particles
            continue
        
        # NBOND section header
        if '!NBOND' in line:
            # The number of bonds is on this line
            parts = line.split()
            n_bonds_expected = 0
            for part in parts:
                try:
                    n_bonds_expected = int(part)
                    break
                except ValueError:
                    continue
            
            # Parse bond pairs from subsequent lines
            i += 1
            collected = 0
            while collected < n_bonds_expected and i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith('!'):
                    i += 1
                    continue
                
                # Each line has integers; bonds come in pairs
                values = line.split()
                try:
                    ints = [int(v) for v in values]
                except ValueError:
                    # Not a bond line, must be another section
                    break
                
                # Group into pairs
                for k in range(0, len(ints) - 1, 2):
                    bond_i = ints[k] - 1      # PSF uses 1-indexed
                    bond_j = ints[k + 1] - 1
                    bonds.append([bond_i, bond_j])
                    collected += 1
                    if collected >= n_bonds_expected:
                        break
                i += 1
            continue
        
        i += 1
    
    return np.array(bonds, dtype=int), n_particles


def morse_force_and_stiffness(r, U0, kappa, r0):
    """
    Compute the radial force F(r) and stiffness k(r) for a Morse pair potential.
    
    U(r) = U0 * [(1 - exp(-kappa*(r - r0)))^2 - 1]
    F(r) = -dU/dr
    k(r) = d2U/dr2
    
    Returns:
        F_r: force magnitude (positive = repulsive, negative = attractive)
        k_r: radial stiffness (d2U/dr2)
    """
    exp_term = np.exp(-kappa * (r - r0))
    # First derivative: dU/dr = 2*U0*kappa * exp(-k(r-r0)) * (1 - exp(-k(r-r0)))
    dU_dr = 2.0 * U0 * kappa * exp_term * (1.0 - exp_term)
    # Second derivative
    d2U_dr2 = 2.0 * U0 * kappa**2 * exp_term * (2.0 * exp_term - 1.0)
    
    F_r = -dU_dr ## FORCE TERM
    k_r = d2U_dr2
    return F_r, k_r


def get_bond_potential_params(type_i, type_j, U0_dict, kappa, r0_dict):
    """
    Given bond endpoint particle types, return the Morse parameters for that bond.
    
    Args:
        type_i, type_j: integer or string particle types (e.g. 0=small, 1=large
                        or 'S'=small, 'L'=large depending on your convention)
        U0_dict: dict like {'SS': 12, 'SL': 18, 'LL': 24} (kT units)
        kappa: scalar, same for all pairs
        r0_dict: dict like {'SS': 2.0, 'SL': 3.0, 'LL': 4.0} (contact distances)
    
    Returns:
        U0, kappa, r0 for this bond
    """
    # Convert numeric types to size label
    type_map = {1: 'S', 2: 'L'} 
    
    if isinstance(type_i, (int, np.integer)):
        ti = type_map[int(type_i)]
    else:
        ti = type_i
    if isinstance(type_j, (int, np.integer)):
        tj = type_map[int(type_j)]
    else:
        tj = type_j
    
    # Sort to get canonical bond type
    pair_sorted = ''.join(sorted([ti, tj]))
    
    return U0_dict[pair_sorted], kappa, r0_dict[pair_sorted]


def compute_affine_modulus_from_frame(positions, types, bonds, box_lengths,
                                        U0_dict, kappa, r0_dict, kT=1.0):
    """
    Compute affine Born modulus contributions for every bond in a frame.
    
    Args:
        positions: (N, 3) array of particle positions
        types: (N,) array of particle types
        bonds: (M, 2) array of bond endpoint indices
        box_lengths: (3,) array of box edge lengths
        U0_dict: bond-type-keyed dict of attraction depths (in kT units)
        kappa: range parameter (typical: 60 for κa=60 with r0=1)
        r0_dict: bond-type-keyed dict of equilibrium distances (~contact)
        kT: thermal energy unit (default 1.0)
    
    Returns:
        DataFrame with one row per bond, columns:
            i, j: particle indices
            type: bond type (SS, SL, LL)
            r: bond length
            F: radial force at this length
            k_radial: radial stiffness
            G_xy_contribution: affine contribution to G_xy
            G_xy_normalized: contribution per unit volume
    """
    V = float(np.prod(box_lengths))
    rows = []
    
    for bidx in range(len(bonds)):
        i, j = int(bonds[bidx, 0]), int(bonds[bidx, 1])
        
        # Vector with PBC
        dr = positions[j] - positions[i]
        dr -= box_lengths * np.round(dr / box_lengths)
        r = np.linalg.norm(dr)
        
        if r == 0:
            continue
        
        # Unit vector components
        nx, ny, nz = dr / r
        
        # Get Morse parameters for this bond type
        U0, kap, r0 = get_bond_potential_params(
            types[i], types[j], U0_dict, kappa, r0_dict
        )
        U0_phys = U0 * kT  # convert kT units to energy units
        
        # Compute force and stiffness
        F_r, k_r = morse_force_and_stiffness(r, U0_phys, kap, r0)
        
        # Born contribution to G_xy
        # C_xyxy_ij = (1/V) * [k(r) - F(r)/r] * (nx*ny)^2 * r^2
        # = (1/V) * [k(r) - F(r)/r] * (nx*ny*r)^2
        # = (1/V) * [k(r) - F(r)/r] * (dr_x * dr_y)^2 / r^2 * r^2
        # = (1/V) * [k(r) - F(r)/r] * dr_x^2 * dr_y^2 / r^2
        # which simplifies to: (1/V) * k(r) * (dr_x*dr_y)^2/r^2 - (1/V)*F(r)*(dr_x*dr_y)^2/r^3
        
        prefactor = (k_r - F_r / r) * (nx * ny)**2 * r**2 / V
        
        # Bond type label (for downstream filtering)
        type_map = {1: 'S', 2: 'L'}
        ti = type_map.get(int(types[i]), str(types[i]))
        tj = type_map.get(int(types[j]), str(types[j]))
        bond_type = ''.join(sorted([ti, tj]))
        bond_type_2letter = bond_type.replace('S', 'S').replace('L', 'L')
        # convert SS/SL/LL convention
        bond_type_label = {
            'SS': 'SS', 'LS': 'SL', 'SL': 'SL', 'LL': 'LL'
        }.get(bond_type_2letter, bond_type_2letter)
        
        rows.append({
            'i': i,
            'j': j,
            'type': bond_type_label,
            'r': r,
            'F': F_r,
            'k_radial': k_r,
            'G_xy_contribution': prefactor,
        })
    
    return pd.DataFrame(rows)


def compute_affine_modulus_total(bond_contributions_df):
    """Sum bond contributions to get the total affine G_xy."""
    return bond_contributions_df['G_xy_contribution'].sum()


def decompose_modulus(bond_contributions_df, decomposition_col='type'):
    """
    Decompose G_xy by bond category (e.g., type, EBC rank, GMM boundary status).
    
    Args:
        bond_contributions_df: output of compute_affine_modulus_from_frame
        decomposition_col: column name to group by
    
    Returns:
        DataFrame with one row per category, columns:
            count: number of bonds
            total_contribution: sum of G_xy contributions
            fraction_of_total: fraction of overall G_xy
            mean_contribution: mean per-bond contribution
    """
    total = bond_contributions_df['G_xy_contribution'].sum()
    
    grouped = (
        bond_contributions_df.groupby(decomposition_col)['G_xy_contribution']
        .agg(['count', 'sum', 'mean'])
        .rename(columns={'sum': 'total_contribution', 'mean': 'mean_contribution'})
    )
    grouped['fraction_of_total'] = grouped['total_contribution'] / total
    
    return grouped


# ============================================================
# Main: load GSD and process
# ============================================================

def process_gsd_frame(gsd_path, frame_index, U0_dict, kappa, r0_dict,
                       bond_cutoff=0.1, kT=1.0, psf_path=None):
    """
    Load one frame from a GSD trajectory and compute affine G_xy.
    
    Args:
        gsd_path: path to .gsd file
        frame_index: which frame to read (typically -1 for last frame)
        U0_dict: e.g. {'SS': 12, 'SL': 18, 'LL': 24}
        kappa: e.g. 60
        r0_dict: e.g. {'SS': 2.0, 'SL': 3.0, 'LL': 4.0} for radii R_S=1, R_L=2
        bond_cutoff: identifies bonds from particle distances (default range 0.1 units)
        kT: thermal energy
        psf_path : the path to the psf file listing all bonds
    
    Returns:
        bond_contributions_df, decomposition_by_type, total_G_xy
    """
    with gsd.hoomd.open(gsd_path, 'r') as traj:
        frame = traj[frame_index]
        colloids = np.where((frame.particles.typeid == 1) | (frame.particles.typeid == 2))[0]
        positions = np.array(frame.particles.position[colloids])
        types = np.array(frame.particles.typeid[colloids])
        box_lengths = np.array(frame.configuration.box[:3])
        N = len(positions)
        
        # Check if bonds are provided from PSF, otherwise calculate from GSD positions
        if psf_path and os.path.exists(psf_path) == True:
            bonds, n_particles = load_bonds_from_psf(psf_path)
        else:
            # WARNING: this is slow -- O(N^2)
            print("No PSF bond data, identifying bonds from distances")
            bonds = []
            for i in range(N):
                for j in range(i + 1, N):
                    dr = positions[j] - positions[i]
                    dr -= box_lengths * np.round(dr / box_lengths)
                    r = np.linalg.norm(dr)
                    _, _, r0 = get_bond_potential_params(
                        types[i], types[j], U0_dict, kappa, r0_dict
                    )
                    cutoff = r0 + bond_cutoff 
                    if r < cutoff:
                        bonds.append([i, j])
            bonds = np.array(bonds)
    
    # Compute contributions
    bond_df = compute_affine_modulus_from_frame(
        positions, types, bonds, box_lengths,
        U0_dict, kappa, r0_dict, kT=kT
    )
    
    total_G = compute_affine_modulus_total(bond_df)
    decomp_by_type = decompose_modulus(bond_df, 'type')
    
    return bond_df, decomp_by_type, total_G



# ============================================================
# RUN THE ANALYSIS
# ============================================================

if __name__ == '__main__':
    gsd_paths = ["quench/quench_clean_small.gsd",
                "quench/quench_clean_bi.gsd", 
                "quench/quench_clean_large.gsd"
               ]
    psf_files = ["../analyze-topology/viz/100-0/100-0_psfs_standard/all_bonds.psf",
                 "../analyze-topology/viz/50-50/50-50_psfs_standard/all_bonds.psf",
                 "../analyze-topology/viz/0-100/0-100_psfs_standard/all_bonds.psf"
                ]
    labels = ['small','bi','large']

    for p in range(len(gsd_paths)):
        gsd_path = gsd_paths[p]
        label = labels[p]
        psf_file = psf_files[p]
        if label == 'large':
          U0_dict = {'SS': 24.0, 'SL': 18.0, 'LS': 18.0, 'LL': 12.0}  # in kT units
          r0_dict = {'SS': 4.0, 'SL': 3.0, 'LS': 3.0, 'LL': 2.0}     # contact distances (radii sum)
        else:
          U0_dict = {'SS': 12.0, 'SL': 18.0, 'LS': 18.0, 'LL': 24.0}  # in kT units
          r0_dict = {'SS': 2.0, 'SL': 3.0, 'LS': 3.0, 'LL': 4.0}     # contact distances (radii sum)

        kappa = 60.0  # attraction range parameter
        kT = 0.1      # thermal energy
    
        # Process last frame
        bond_df, decomp, total_G = process_gsd_frame(
            gsd_path, frame_index=-1,
            U0_dict=U0_dict, kappa=kappa, r0_dict=r0_dict,
            bond_cutoff=3/60, #3/kappa
            kT=kT, psf_path=psf_file
        )
    
        print(f"--{label}--")
        print(f"Total affine G_xy = {total_G:.4f}")
        print(f"\nBy bond type:")
        print(decomp)
        print(f"\nMean G_xy contribution per bond, by type:")
        print(bond_df.groupby('type')['G_xy_contribution'].describe())
    
        # Save bond-level data for combining with EBC analysis
        bond_df.to_csv(f"{data_outpath}/affine_modulus_per_bond_{label}.csv", index=False)
