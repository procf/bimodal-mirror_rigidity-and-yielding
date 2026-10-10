########################
""" MODULE LIBRARY """
########################
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx

import os
import re
import glob

from scipy.spatial import cKDTree
from statistics import mean
from scipy.spatial.distance import pdist
from itertools import combinations # for geometry

# tetrahedral classification [optional]
from collections import Counter
from collections import defaultdict



########################
""" DATA HANDLING """
########################
small_comps = [100, 50, 0]
compositions = ['100-0', '50-50', '0-100']
size_ratios = ['1-0', '1-2', '0-2']
volume_fractions = ['20']
attraction_strengths = ['12'] 
depletion_scaling = ['bysize'] 
attraction_range = ['60']

# choose: 'standard' or 'weighted' 
clustering_style = ['standard']

eta0 = 0.3
kT = 0.1
L = 70
system_volume = L**3

# create "data" subfolder if it doesn't exit
data_outpath = "data"
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)


#################
# FUNCTIONS
#################

# ============================================================
# Get position from igraph by original id, not igraph index
# ============================================================
def positions_from_igraph(g):
    """
    Extract particle positions from an igraph Graph, sorted by particle ID.
    
    Args:
        g: igraph Graph with vertex attributes 'x', 'y', 'z' and either
           'id' or 'name' identifying the particle.
    
    Returns:
        positions: (N, 3) numpy array, sorted by particle ID
        node_labels: (N,) array of integer particle IDs in sorted order
    """
    # Get node IDs (matches your existing code)
    node_labels = g.vs['id'] if 'id' in g.vs.attributes() else g.vs['name']
    node_labels = np.array(list(map(int, node_labels)))
    
    # Get positions in the order they appear in the graph
    x = np.array(g.vs['x'], dtype=float)
    y = np.array(g.vs['y'], dtype=float)
    z = np.array(g.vs['z'], dtype=float)
    positions_unsorted = np.column_stack([x, y, z])
    
    # Sort by particle ID
    sort_idx = np.argsort(node_labels)
    positions = positions_unsorted[sort_idx]
    node_labels_sorted = node_labels[sort_idx]
    
    return positions, node_labels_sorted

# ============================================================
# Assign stiffness
# ============================================================

def assign_stiffness(g, k_SS, k_SL, k_LL):

    stiffness = []
    bond_type = []

    for e in g.es:
        u, v = e.tuple
        tu = g.vs[u]["type"]
        tv = g.vs[v]["type"]

        if tu == 1 and tv == 1:
            stiffness.append(k_SS)
            bond_type.append("SS")
        elif tu == 2 and tv == 2:
            stiffness.append(k_LL)
            bond_type.append("LL")
        else:
            stiffness.append(k_SL)
            bond_type.append("SL")

    g.es["stiffness"] = stiffness
    g.es["bond_type"] = bond_type


# ============================================================
# Calculate Edge Betweenness
# ============================================================

def edge_betweenness(g):

    ebc_raw_unweighted = np.array(g.edge_betweenness())
    N = g.vcount()
    ebc_unweighted = ebc_raw_unweighted / (N * (N - 1) / 2)

    # EBC assumes low weight, less cost, easier path
    # ...opposite logic of stiffness, so invert value
    stiffness = np.array(g.es["stiffness"], dtype=float)
    inv_stiff = 1.0 / stiffness
    ebc_raw_weighted = np.array(g.edge_betweenness(weights=inv_stiff))
    #N = g.vcount()
    ebc_weighted = ebc_raw_weighted / (N * (N - 1) / 2)

    return ebc_unweighted, ebc_weighted


# ============================================================
# Define a "mechnical backbone" as the high EBC values within some percentile
# ============================================================
def mechanical_backbone(g, backbone_percentile, weighted=True):
    """
    returns:
      - backbone_edges : the bonds that have EBC values within the backbone_percentile
      - mech_vals : an optional stiffness weighted EBC value
    """

    if g.ecount() == 0:
        return [], np.array([])

    if weighted:
        stiffness = np.array(g.es["stiffness"], dtype=float)
        inv_stiff = 1.0 / stiffness

        # approximate current flow as EBC 
        # EBC assumes low weight, less cost, easier path
        # ...opposite logic of stiffness, so invert value
        cf_raw = np.array(g.edge_betweenness(weights=inv_stiff))
        N = g.vcount()
        cf = cf_raw / (N * (N - 1) / 2)
        # -> path weighted EBC

        # a floppy bridge may have high EBC but low force transmission
        # ...multiple by stiffness to weight that correctly
        mech_vals = cf * stiffness
        # -> mechanically weighted EBC

    else:
        # approximate current flow as EBC if sparse, tree-like, tenuous network...
        cf_raw = np.array(g.edge_betweenness())
        N = g.vcount()
        cf = cf_raw / (N * (N - 1) / 2)
        mech_vals = cf
        # -> pure topological EBC

    # Normalize once
    # Track scores relative to strongest mechanical edge in each frame
    max_val = mech_vals.max() if len(mech_vals) else 1
    mech_vals = mech_vals / max_val

    threshold = np.percentile(mech_vals, backbone_percentile)
    backbone_ids = np.where(mech_vals > threshold)[0]
    backbone_edges = [g.es[i].tuple for i in backbone_ids]

    return backbone_edges, mech_vals


# ============================================================
# TETRAHEDRAL STRUCTURE CLASSIFICATION
# -- NOTE: the tetrahedral classification scheme is not used
#          in this paper we only consider tetrahedra or non-tetrahedra;
#          however, the classification code is included as optional.
#          This tetrahedra aggregate classification data is used in the 
#          maniscript "Size matters more than packing in bimodal colloidal
#           gel compositions" |  https://doi.org/10.48550/arXiv.2608.16874
# ============================================================

def build_tetrahedron_graph(tetrahedra):
    """
    Build adjacency graph of tetrahedra.

    Returns:
        tg: igraph.Graph where nodes = tetrahedra
    """
    tg = ig.Graph()
    tg.add_vertices(len(tetrahedra))

    # Convert to sets for fast overlap checks
    tet_sets = [set(t) for t in tetrahedra]

    edges = []

    for i in range(len(tetrahedra)):
        for j in range(i + 1, len(tetrahedra)):
            overlap = len(tet_sets[i].intersection(tet_sets[j]))

            # Face-sharing (3 nodes) or edge-sharing (2 nodes)
            if overlap >= 2:
                edges.append((i, j))

    tg.add_edges(edges)

    return tg

def extract_particle_subgraph(g, tetrahedra, component):
    nodes = set()
    for idx in component:
        nodes.update(tetrahedra[idx])

    subg = g.subgraph(list(nodes))
    return subg

def classify_clusters(g, tetrahedra, tg):
    comps = tg.components()
    cluster_info = []
    
    for comp in comps:
        size = len(comp)
        
        if size == 1:
            label = "ρ1T"
        elif size == 2:
            label = "ρ2T"  # always a single edge (face- or edge-share)
        else:
            # Get the subgraph of tg restricted to this component
            comp_subgraph = tg.subgraph(comp)
            n_edges = comp_subgraph.ecount()
            n_nodes = comp_subgraph.vcount()  # = size
            max_degree = max(comp_subgraph.degree())
            
            if size == 3:
                # Two possibilities: chain (2 edges) or triangle (3 edges)
                if n_edges == 2:
                    label = "ρ3T_chain"
                elif n_edges == 3:
                    label = "ρ3T_triangle"
                else:
                    label = "ρ3T_other"  # shouldn't happen
            
            elif size == 4:
                # Possibilities: linear chain (3 edges, max_deg=2),
                #                star/branch (3 edges, max_deg=3),
                #                cycle (4 edges, max_deg=2),
                #                triangle+pendant (4 edges, max_deg=3),
                #                K_4 (6 edges)
                if n_edges == 3:
                    if max_degree == 2:
                        label = "ρ4T_chain"
                    else:  
                        label = "ρ4T_tree"
                elif n_edges == 4:
                    if max_degree == 2:
                        label = "ρ4T_cycle"
                    else:
                        label = "ρ4T_triangle_pendant"
                else:
                    label = "ρ4T_dense"  # >4 edges = denser than tree+1
            
            elif size == 5:
                # Pentagonal bipyramid is a specific particle-level pattern
                subg = extract_particle_subgraph(g, tetrahedra, comp)
                if subg.vcount() == 7 and subg.ecount() == 16:
                    label = "ρ5T_bipyramid"
                else:
                    # Classify by tg topology instead
                    if n_edges == 4 and max_degree == 2:
                        label = "ρ5T_chain"
                    elif n_edges == 4 and max_degree > 2:
                        label = "ρ5T_tree"
                    else:
                        label = "ρ5T_dense"
            
            else:  # size >= 6
                # For large components, summarize by topology rather than enumerate
                # cyclomatic = E - N + 1 (number of independent cycles)
                cyclomatic = n_edges - n_nodes + 1
                if cyclomatic == 0:
                    if max_degree == 2:
                        label = f"ρ{size}T_chain"
                    else:
                        label = f"ρ{size}T_tree"  # branched but acyclic
                else:
                    label = f"ρ{size}T_dense"  # has cycles
        
        cluster_info.append((comp, label))
    
    return cluster_info

def count_motifs(t_info):
    labels = [label for _, label in t_info]
    return Counter(labels)


def tetra_edges_from_nodes(g, tet):
    """
    Return set of edge IDs belonging to one tetrahedron (4 nodes).
    """
    edges = set()

    for i in range(len(tet)):
        for j in range(i+1, len(tet)):
            u, v = tet[i], tet[j]
            eid = g.get_eid(u, v, directed=False, error=False)
            if eid != -1:
                edges.add(eid)

    return edges

def motif_edge_set(g, tetrahedra, comp):
    """
    Exact union of tetrahedral edges for one motif component.
    """
    edges = set()

    for tidx in comp:
        tet = tetrahedra[tidx]
        edges |= tetra_edges_from_nodes(g, tet)

    return edges

def bond_type_counts_from_edges(g, edge_ids):
    counts = {"SS":0, "SL":0, "LL":0}

    for eid in edge_ids:
        bt = g.es[eid]["bond_type"]
        counts[bt] += 1

    return counts

def analyze_motifs(g, tetrahedra, tg, t_info):
    """Per-component analysis. Stores edge_ids for downstream overlap computations."""
    results = []
    for comp, label in t_info:
        edge_ids = motif_edge_set(g, tetrahedra, comp)
        edge_ids_set = set(edge_ids)

        nodes = set()
        for tidx in comp:
            nodes.update(tetrahedra[tidx])

        bond_counts = bond_type_counts_from_edges(g, edge_ids)
        total_bonds = sum(bond_counts.values())
        bond_fraction = {
            k: (v / total_bonds if total_bonds > 0 else 0)
            for k, v in bond_counts.items()
        }

        if len(comp) > 1:
            comp_subgraph = tg.subgraph(comp)
            n_adjacencies = comp_subgraph.ecount()
            cyclomatic = n_adjacencies - len(comp) + 1
            max_degree = max(comp_subgraph.degree())
        else:
            n_adjacencies = 0
            cyclomatic = 0
            max_degree = 0

        results.append({
            "label": label,
            "n_tetra": len(comp),
            "n_adjacencies": n_adjacencies,
            "cyclomatic": cyclomatic,
            "max_degree": max_degree,
            "n_nodes": len(nodes),
            "n_edges": len(edge_ids),
            "edge_ids": edge_ids_set,  # NEW
            "bond_counts": bond_counts,
            "bond_fraction": bond_fraction,
        })
    return results

def add_overlap_to_motifs(pt_results, edge_set, label):
    """
    Add per-component overlap fields with a given edge set.
    
    Adds three new keys per result:
        n_edges_in_{label}: count of motif edges in edge_set
        frac_edges_in_{label}: fraction of motif edges in edge_set
        in_{label}: True if any motif edge is in edge_set
    
    Args:
        pt_results: output of analyze_motifs (list of per-component dicts)
        edge_set: iterable of edge IDs (will be converted to set)
        label: name to use in field keys (e.g., 'backbone', 'bridges', 'bridges_w')
    """
    edge_set = set(edge_set)
    
    for r in pt_results:
        comp_edges = r['edge_ids']
        n_overlap = len(comp_edges & edge_set)
        r[f'n_edges_in_{label}'] = n_overlap
        r[f'frac_edges_in_{label}'] = (
            n_overlap / len(comp_edges) if comp_edges else 0
        )
        r[f'in_{label}'] = n_overlap > 0
    
    return pt_results  # also modified in place


def aggregate_tetrahedra_by_motif(results):
    """
    Sum tetrahedra-counts and component-counts within each motif label.
    Works with any label scheme including dynamically-generated labels
    like 'ρ7T_chain'.
    """
    agg = defaultdict(lambda: {'n_components': 0, 'n_tetra': 0})
    for r in results:
        agg[r['label']]['n_components'] += 1
        agg[r['label']]['n_tetra'] += r['n_tetra']
    return dict(agg)


def aggregate_tetrahedra_by_topology(results):
    """
    Aggregate at a coarser level: by topology class regardless of size.
    Useful for 'fraction of tetrahedra in chains' etc. across all sizes.
    """
    agg = defaultdict(lambda: {'n_components': 0, 'n_tetra': 0})
    for r in results:
        topology = parse_topology_class(r['label'])
        agg[topology]['n_components'] += 1
        agg[topology]['n_tetra'] += r['n_tetra']
    return dict(agg)


def parse_topology_class(label):
    """
    Map a fine-grained label to a coarse topology class.
    
    Returns one of: 'isolated', 'pair', 'chain', 'tree', 'cycle', 'dense', 
                    'bipyramid', 'triangle', 'triangle_pendant', 'other'
    """
    if label == "ρ1T":
        return "isolated"
    elif label == "ρ2T":
        return "pair"
    elif label == "ρ5T_bipyramid":
        return "bipyramid"
    elif "_chain" in label:
        return "chain"
    elif "_tree" in label or "_branch" in label:
        return "tree"
    elif "_cycle" in label:
        return "cycle"
    elif "_dense" in label:
        return "dense"
    elif "_triangle_pendant" in label:
        return "triangle_pendant"
    elif "_triangle" in label:
        return "triangle"
    else:
        return "other"

def build_tetrahedra_dataframe(results, config_label=None):
    """Build per-component dataframe. Auto-discovers overlap fields."""
    rows = []
    
    standard_keys = {
        'label', 'n_tetra', 'n_adjacencies', 'cyclomatic', 'max_degree',
        'n_nodes', 'n_edges', 'edge_ids', 'bond_counts', 'bond_fraction'
    }
    
    for comp_id, r in enumerate(results):
        row = {}
        if config_label is not None:
            row.update(config_label)
        
        row['component_id'] = comp_id
        row['label'] = r['label']
        row['topology_class'] = parse_topology_class(r['label'])
        row['n_tetra'] = r['n_tetra']
        row['n_adjacencies'] = r['n_adjacencies']
        row['cyclomatic'] = r['cyclomatic']
        row['max_degree'] = r['max_degree']
        row['n_nodes'] = r['n_nodes']
        row['n_edges'] = r['n_edges']
        
        bc = r['bond_counts']
        row['bond_count_SS'] = bc.get('SS', 0)
        row['bond_count_SL'] = bc.get('SL', 0)
        row['bond_count_LL'] = bc.get('LL', 0)
        
        # Auto-pick up any overlap fields added by add_overlap_to_motifs
        for key, value in r.items():
            if key not in standard_keys and not isinstance(value, (set, list, dict)):
                row[key] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================
# COMPARE CLUSTER BRIDGES TO MOTIFS 
# ============================================================
def motif_edges_by_label(g, tetrahedra, t_info):
    """
    build motif edge dictionary"
    """
    motif_dict = {}

    for comp, label in t_info:
        edge_ids = motif_edge_set(g, tetrahedra, comp)

        motif_dict.setdefault(label, []).append(edge_ids)

    return motif_dict


def eid_to_motif_labels(g, tetrahedra, t_info):
    motif_dict = motif_edges_by_label(g, tetrahedra, t_info)

    eid_map = defaultdict(set)

    for label, motif_list in motif_dict.items():
        for eset in motif_list:
            for eid in eset:
                eid_map[eid].add(label)

    return eid_map

def annotate_overlap_df(g, overlap_df, eid_map, use_topology_class=False):
    labels_attr = "id" if "id" in g.vs.attributes() else "name"
    pid_to_vid = {int(g.vs[i][labels_attr]): i for i in range(g.vcount())}
    
    labels = []
    for u_pid, v_pid in zip(overlap_df["source_particle"], overlap_df["target_particle"]):
        u_pid = int(u_pid)
        v_pid = int(v_pid)
        if u_pid not in pid_to_vid or v_pid not in pid_to_vid:
            labels.append("missing_vertex")
            continue
        u = pid_to_vid[u_pid]
        v = pid_to_vid[v_pid]
        eid = g.get_eid(u, v, directed=False, error=False)
        if eid == -1:
            labels.append("missing_edge")
        elif eid in eid_map:
            raw_labels = eid_map[eid]
            if use_topology_class:
                # Map each label to its topology class, then dedupe
                topology_classes = sorted(set(parse_topology_class(lab) for lab in raw_labels))
                labels.append(",".join(topology_classes))
            else:
                labels.append(",".join(sorted(raw_labels)))
        else:
            labels.append("none")
    
    out = overlap_df.copy()
    out["motif_label"] = labels  # raw labels
    out["motif_topology_class"] = [
        ",".join(sorted(set(parse_topology_class(l) for l in lbl.split(","))))
        if lbl not in ("missing_vertex", "missing_edge", "none") else lbl
        for lbl in labels
    ]
    return out


# ============================================================
# SAVE BRIDGE TYPE DATA 
# ============================================================

def cluster_bridge_eids(g, cluster_bridge_df):
    """Get edge IDs from cluster_bridge_df entries."""
    labels_attr = "id" if "id" in g.vs.attributes() else "name"
    pid_to_vid = {int(g.vs[i][labels_attr]): i for i in range(g.vcount())}
    
    eids = set()
    for row in cluster_bridge_df.itertuples(index=False):
        u_pid = int(row.source_particle)
        v_pid = int(row.target_particle)
        if u_pid in pid_to_vid and v_pid in pid_to_vid:
            eid = g.get_eid(pid_to_vid[u_pid], pid_to_vid[v_pid], 
                            directed=False, error=False)
            if eid != -1:
                eids.add(eid)
    
    return eids

# ----------------------------------------------------------
# Build full edge dataframe for every bond in igraph network
# ----------------------------------------------------------

def build_bridge_edge_dataframe(
    g,
    cluster_bridge_df,
    size_ratio,
    phi,
    composition,
    D0,
    scaled_D0,
    kappa,
    cluster_style
):
    """
    Returns dataframe with one row per graph edge.

    Columns:
    size-ratio, phi, composition, D0, scaled_D0, kappa,
    cluster_style,
    source_particle, target_particle,
    type_source, type_target,
    bond_type,
    bridge_type
    """

    # ------------------------------------------------------
    # Vertex labels (particle IDs)
    # ------------------------------------------------------
    node_labels = g.vs["id"] if "id" in g.vs.attributes() else g.vs["name"]
    node_labels = list(map(int, node_labels))

    # ------------------------------------------------------
    # Determine single vs multi bridge sets
    # ------------------------------------------------------
    tmp = cluster_bridge_df.copy()

    sc = tmp["source_cluster"].to_numpy()
    tc = tmp["target_cluster"].to_numpy()

    cmin = np.minimum(sc, tc)
    cmax = np.maximum(sc, tc)

    tmp["cluster_pair"] = list(zip(cmin, cmax))

    pair_counts = (
        tmp.groupby("cluster_pair")
           .size()
           .reset_index(name="n_bonds")
    )

    single_pairs = set(
        pair_counts.loc[pair_counts["n_bonds"] == 1, "cluster_pair"]
    )

    multi_pairs = set(
        pair_counts.loc[pair_counts["n_bonds"] > 1, "cluster_pair"]
    )

    # ------------------------------------------------------
    # Edge sets in particle IDs
    # ------------------------------------------------------
    single_edge_set = set(
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v, cpair in zip(
            tmp["source_particle"],
            tmp["target_particle"],
            tmp["cluster_pair"]
        )
        if cpair in single_pairs
    )

    multi_edge_set = set(
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v, cpair in zip(
            tmp["source_particle"],
            tmp["target_particle"],
            tmp["cluster_pair"]
        )
        if cpair in multi_pairs
    )

    # ------------------------------------------------------
    # Build rows for ALL graph edges
    # ------------------------------------------------------
    rows = []

    for e in g.es:

        iu, iv = e.tuple

        pu = int(node_labels[iu])
        pv = int(node_labels[iv])

        source_particle = min(pu, pv)
        target_particle = max(pu, pv)

        edge_key = (source_particle, target_particle)

        # ----------------------------------------------
        # Bridge type
        # ----------------------------------------------
        if edge_key in single_edge_set:
            bridge_type = "single"
        elif edge_key in multi_edge_set:
            bridge_type = "multi"
        else:
            bridge_type = "nonbridge"

        # ----------------------------------------------
        # Particle types
        # ----------------------------------------------
        type_u = g.vs[iu]["type"] if "type" in g.vs.attributes() else None
        type_v = g.vs[iv]["type"] if "type" in g.vs.attributes() else None

        # reorder types to match particle ordering
        if pu <= pv:
            type_source = type_u
            type_target = type_v
        else:
            type_source = type_v
            type_target = type_u

        # ----------------------------------------------
        # Bond type
        # ----------------------------------------------
        bond_type = e["bond_type"] if "bond_type" in g.es.attributes() else None

        rows.append({
            "size-ratio": size_ratio,
            "phi": phi,
            "composition": composition,
            "D0": D0,
            "scaled_D0": scaled_D0,
            "kappa": kappa,
            "cluster_style": cluster_style,

            "source_particle": source_particle,
            "target_particle": target_particle,

            "type_source": type_source,
            "type_target": type_target,

            "bond_type": bond_type,
            "bridge_type": bridge_type
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_topology(g, k_SS, k_SL, k_LL,
                    backbone_percentile=90, # 90 -> only top 10% of mechanically relevant edges
                    ):

    assign_stiffness(g, k_SS, k_SL, k_LL)
    ebc_unweighted, ebc_weighted = edge_betweenness(g)
    backbone, mech_vals = mechanical_backbone(g, backbone_percentile, weighted=False)
    backbone_w, mech_vals_w = mechanical_backbone(g, backbone_percentile, weighted=True) 

    # Backbone composition
    comp_counts = {'SS':0,'SL':0,'LL':0}
    for e in backbone:
        eid = g.get_eid(*e)
        comp_counts[g.es[eid]["bond_type"]] += 1

    total = len(backbone)
    comp = {k:(v/total if total>0 else 0) for k,v in comp_counts.items()}

    comp_counts_w = {'SS':0,'SL':0,'LL':0}
    for e in backbone_w:
        eid = g.get_eid(*e)
        comp_counts_w[g.es[eid]["bond_type"]] += 1

    total_w = len(backbone_w)
    comp_w = {k:(v/total_w if total_w>0 else 0) for k,v in comp_counts_w.items()}

    # ========================================================
    # GEOMETRY
    # ========================================================
    # --- helper: get neighbors ---
    def neighbors_set(v):
        return set(g.neighbors(v))

    # --- helper: edge existence ---
    def has_edge(u, v):
        return g.are_adjacent(u, v)

    # 1. TETRAHEDRA (exact K4)
    tetrahedra = [c for c in g.cliques(min=4, max=4)]  # already exact

    # EDGE SETS
    def get_edges_from_nodes(nodes):
        edges = set()
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if has_edge(nodes[i], nodes[j]):
                    edges.add(g.get_eid(nodes[i], nodes[j]))
        return edges

    tetra_edges = set().union(*[get_edges_from_nodes(c) for c in tetrahedra]) if tetrahedra else set()

    # OVERLAP : geom and mech-relevant high EBC
    def overlap_fraction(struct_edges, backbone_pairs):
        backbone_eids = set(g.get_eid(u,v) for (u,v) in backbone_pairs)
        frac_bt = len(backbone_eids & struct_edges) / len(backbone_eids) if len(backbone_eids) else 0
        frac_tb = len(backbone_eids & tetra_edges) / len(tetra_edges) if len(tetra_edges) else 0
        return frac_bt, frac_tb

    # how much of my mechanical backbone is made of tetrahedra (rigid geometric motifs)
    # frac_backbone_tetra | fraction of the backbone that is tetrahedral (mech edges supported by geometry)
    # frac_tetra_backbone | fraction of tetrahedra that are in the backbone (geom motifs that are mech relevant)
    frac_backbone_tetra, frac_tetra_backbone = overlap_fraction(tetra_edges, backbone) # unweighted
    frac_backbone_tetra_w, frac_tetra_backbone_w = overlap_fraction(tetra_edges, backbone_w) # weighted

    # ---- composition tracking ----
    def structure_composition(cliques):
        comp_counts = {'SS':0, 'SL':0, 'LL':0}
        total_edges = 0

        for c in cliques:
            edges = get_edges_from_nodes(c)
            for eid in edges:
                bond_type = g.es[eid]["bond_type"]
                comp_counts[bond_type] += 1
                total_edges += 1

        if total_edges == 0:
            return {k:0 for k in comp_counts}

        return {k: v/total_edges for k,v in comp_counts.items()}

    # comp in and out of tetrahedra
    def edge_set_composition(edge_set):
        comp_counts = {'SS':0, 'SL':0, 'LL':0}

        for eid in edge_set:
            bond_type = g.es[eid]["bond_type"]
            comp_counts[bond_type] += 1

        total = sum(comp_counts.values())

        if total == 0:
            return {k:0 for k in comp_counts}

        return {k: v/total for k,v in comp_counts.items()}

    # unique edges only
    tetra_comp = edge_set_composition(tetra_edges)

    # --- split high EBC edges ---
    backbone_eids = set(g.get_eid(u,v) for (u,v) in backbone)
    backbone_w_eids = set(g.get_eid(u,v) for (u,v) in backbone_w)

    ## ADDITIONAL TETRAHEDRAL STRUCTURES
    tg = build_tetrahedron_graph(tetrahedra)
    t_info = classify_clusters(g, tetrahedra, tg)
    pt_motif_counts = count_motifs(t_info)
    #pt_results = analyze_motifs(g, tetrahedra, t_info)
    pt_results = analyze_motifs(g, tetrahedra, tg, t_info)
    #pt_summary = aggregate_by_motif(pt_results)

    # Per-component overlap (for tetrahedra.csv)
    backbone_eids_set = set(
        g.get_eid(u, v, directed=False, error=False) for (u, v) in backbone
    )
    backbone_eids_set.discard(-1)
    backbone_w_eids_set = set(
        g.get_eid(u, v, directed=False, error=False) for (u, v) in backbone_w
    )
    backbone_w_eids_set.discard(-1)

    add_overlap_to_motifs(pt_results, backbone_eids_set, 'backbone')
    add_overlap_to_motifs(pt_results, backbone_w_eids_set, 'backbone_w')

    # your backbone edges (already filtered)
    backbone_edges = set(backbone_eids)

    g_backbone = g.subgraph_edges(backbone_eids, delete_vertices=False) 

    # ...and create a dataframe with all the edge info
    # ==========================================================
    # tetrahedron index -> cluster label
    # ==========================================================
    tetra_idx_to_label = {}

    for comp_eid, label in t_info:
        for tidx in comp_eid:
            tetra_idx_to_label[tidx] = label


    # ==========================================================
    # edge -> ALL motif memberships
    # ==========================================================
    edge_to_labels = {e.index: set() for e in g.es}

    for tidx, tet_nodes in enumerate(tetrahedra):

        label = tetra_idx_to_label.get(tidx, "other")

        tet_edges = get_edges_from_nodes(tet_nodes)

        for eid in tet_edges:
             edge_to_labels[eid].add(label)


    # ==========================================================
    # backbone sets
    # ==========================================================
    backbone_eids   = set(g.get_eid(u, v) for (u, v) in backbone)
    backbone_w_eids = set(g.get_eid(u, v) for (u, v) in backbone_w)


    # ==========================================================
    # build dataframe
    # ==========================================================
    rows = []
    for e in g.es:
        eid = e.index
        i, j = e.tuple
        labels = edge_to_labels[eid]
        rows.append({
            "eid": eid,
            "i": i,
            "j": j,
            "type": e["bond_type"],
            "ebc": ebc_unweighted[eid],
            "ebc_w": ebc_weighted[eid],
            # Topology-class flags (new scheme)
            "in_isolated": "ρ1T" in labels,
            "in_pair": "ρ2T" in labels,
            "in_chain": any("_chain" in lab for lab in labels),
            "in_tree": any("_tree" in lab or "_branch" in lab for lab in labels),
            "in_cycle": any("_cycle" in lab for lab in labels), # or "ρ3T_triangle" in labels,
            "in_dense": any("_dense" in lab for lab in labels),
            "in_bipyramid": "ρ5T_bipyramid" in labels,
            "in_triangle": any("ρ3T_triangle" in lab for lab in labels),
            "in_triangle_pendant": any("_triangle_pendant" in lab for lab in labels),
            # Derived flags
            "multiT": len(labels) > 1,
            "noT": len(labels) == 0,
            # Readable membership list
            #"motifs": sorted(labels) if labels else ["noT"],
            "motifs": sorted(labels)[0] if labels else "noT",
            # Backbone flags
            "backbone": eid in backbone_eids,
            "backbone_w": eid in backbone_w_eids
        })
    edge_data_df = pd.DataFrame(rows)


    return {
        'ebc_unweighted': ebc_unweighted,
        'ebc_weighted': ebc_weighted,

        'n_backbone': len(backbone),
        'n_backbone_w': len(backbone_w),
        'backbone_edges': backbone,
        'backbone_edges_w': backbone_w,
        'backbone_composition': comp,
        'backbone_composition_w': comp_w,

        "n_tetra": len(tetrahedra),
        'frac_backbone_tetra': frac_backbone_tetra, 
        'frac_tetra_backbone': frac_tetra_backbone,
        'frac_backbone_tetra_w': frac_backbone_tetra_w, 
        'frac_tetra_backbone_w': frac_tetra_backbone_w,

        'tetrahedra': tetrahedra,
        'tetra_comp': tetra_comp,

        't_info': t_info,
        'pt_motif_counts': pt_motif_counts,
        #'pt_summary': pt_summary,
        'pt_results': pt_results,

        'edge_data_df': edge_data_df,
    }
 



###########################
""" ANALYZE ALL SYSTEMS """
###########################
tetra_dfs = []
cdf_list = []
edge_data_dfs = []
ph_vals_dfs = []
ph_w_vals_dfs = []
bridge_dfs = []
bridge_dfs_w = []
bridge_motifs_dfs = []
system_dfs = []
for size in size_ratios:
  radii = size.split('-')
  R_C1 = float(radii[0])
  R_C2 = float(radii[1])
  for phi in volume_fractions:
      for comp_1 in small_comps:
          comp_2 = 100 - comp_1
          comp = f'{comp_1}-{comp_2}'
          # skip invalid fake mixtures for monodisperse case
          #print(size, comp, R_C1, R_C2, f'0-{int(R_C2)}', (size == f'0-{int(R_C2)}'))
          if size != '1-0' and comp == '100-0':
              #print('...skip bimodal size or mono_large size for mono_small comp')
              continue
          if size == '1-0' and comp != '100-0':
              #print('---skip mono_small size for bimodal comp or mono_large comp')
              continue
          if size != f'0-{int(R_C2)}' and comp == '0-100':
              #print('++++skip bimodal size or mono_small size for mono_large comp')
              continue
          if size == f'0-{int(R_C2)}' and comp != '0-100':
              #print('////skip mono_large size for bimodal comp or mono_small comp')
              continue
          if size == f'0-{int(R_C2)}' and comp == '0-100':
              size = f'{int(R_C2)}-0'
              #print('reformat size for mono_large size and mono_large comp')
              if R_C2 != float(int(R_C2)):
                  print('ERROR: large monomodal data processing script cannot accept float particle size: {R_C2}')
                  continue
          #print(size, comp)
          for D0 in attraction_strengths:
              for scaling in depletion_scaling:
                  for kappa in attraction_range:
                      if scaling == 'uniform':
                          if (size == '1-0') and (comp == '100-0'):
                             print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                             continue
                          elif (size == f'{int(R_C2)}-0') and (comp == '0-100'):
                             print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                             continue
                          elif comp != '100-0':
                             sim_path = (f'/projects/props/Rob/colloids/bimodal/uniform-potential/r{size}'
                                f'/DPD/L70/poly0.0/seedNone/phi{phi}/{comp}/potential-morse/'
                                f'{D0}kT/kappa{kappa}/') 
                             data_path = (f'{sim_path}analysis-bi-colloids/data/') 
                      else:
                          if (size == '1-0') and (comp == '100-0'):
                             if scaling == 'bysize':
                                sim_path = (f'/projects/props/Rob/colloids/bimodal/r{size}/DPD/L{L}/mono/seedNone/phi{phi}'
                                   f'/potential-morse-bysize/{D0}kT/kappa{kappa}/')
                                data_path = (f'{sim_path}analysis-DPD/data/')
                                print(data_path)
                             else:
                                print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                                continue
                          elif (size == f'{int(R_C2)}-0') and (comp == '0-100'):
                             if scaling == 'bysize':
                                sim_path = (f'/projects/props/Rob/colloids/bimodal/r{size}/DPD/L{L}/phi{phi}'
                                   f'/potential-morse-bysize/{D0}kT/kappa{kappa}/')
                                data_path = (f'{sim_pat}analysis-DPD/data/')
                             else:
                                print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                                continue
                          elif (comp != '100-0') and (comp != '0-100'):
                             sim_path = (f'/projects/props/Rob/colloids/bimodal/r{size}'
                                f'/DPD/L70/poly0.0/seedNone/phi{phi}/{comp}/potential-morse-{scaling}/'
                                f'{D0}kT/kappa{kappa}/') 
                             data_path = (f'{sim_path}analysis-bi-colloids/data/') 
                          else:
                             continue

                      graph_path = f"{data_path}graph_lastframe.graphml"
                      if os.path.exists(graph_path) == False:
                          print(f"No GraphML file: {graph_path}")

                          if "DPD" in sim_path:
                            colloid1_typeid = 1
                            colloid2_typeid = 2
                          else:
                            colloid1_typeid = 0
                            colloid2_typeid = 1

                          cut_off = round(3/int(kappa),2)

                          gsd_path = f"{sim_path}Gelation_Colloids.gsd"
                          if os.path.exists(gsd_path) == False:
                              print(f"No GSD file: {gsd_path}")
                              print(f"Cannot make graph for this dataset")
                              continue

                          edgelist_path = f"{data_path}frame-edges"
                          if os.path.exists(edgelist_path) == False:
                              print(f"Saving edgelist data to:")
                              print(f"  {edgelist_path}")

                              traj = gsd.hoomd.open(gsd_path, 'r')
                              nframes = len(traj)

                              # set path and filename
                              edge_dir_path = data_outpath+'/frame-edges'
                              edge_output = edge_dir_path+'/edgelist' # + <#>.csv in python loop  
                              # check for existing CSV data 
                              if os.path.exists(edge_dir_path) == False:
                                os.mkdir(edge_dir_path)

                              if (os.path.exists(edge_output+str(nframes-1)+'.csv') == False):
                                colloids = np.where((traj[-1].particles.typeid == colloid1_typeid) | (traj[-1].particles.typeid == colloid2_typeid))[0]
                                ncolloids = len(colloids)
                                radii = 0.5*traj[-1].particles.diameter[colloids]
                                pos = traj[i].particles.position[colloids]

                                rcut = cut_off
                                lbox = traj[-1].configuration.box[:3]

                                # cKDTree's periodic mode needs coordinates in [0, L)
                                wrapped = np.mod(pos + 0.5 * lbox, lbox)
                                wrapped[wrapped >= lbox] = 0.0  # guard against float round-off landing on L
                                tree = cKDTree(wrapped, boxsize=lbox)

                                # candidate pairs: the largest possible center-center bond distance
                                max_dist = rcut + 2.0 * radii.max()
                                pairs = tree.query_pairs(r=max_dist, output_type='ndarray')  # rows are (i, j) with i < j

                                if len(pairs) > 0:
                                    i, j = pairs[:, 0], pairs[:, 1]
                                    # center-center separation with minimum-image convention
                                    rij = pos[i] - pos[j]
                                    rij -= lbox * np.round(rij / lbox)
                                    # surface-surface distance
                                    hij = np.linalg.norm(rij, axis=1) - (radii[i] + radii[j])
                                    pairs = pairs[hij <= rcut]
                                    # sort to order by i, then j
                                    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]

                                # save the edgelist
                                np.savetxt(f"{edge_output}{nframes-1}.csv", pairs.reshape(-1, 2), fmt='%d', delimiter=',',
                                           header='i,j', comments='')


                          #################
                          # MAKE THE GRAPH
                          #################

                          # get the number of particles
                          traj = gsd.hoomd.open(gsd_path, 'r')
                          nframes = len(traj)
                          colloids = np.where((traj[-1].particles.typeid == colloid1_typeid) | (traj[-1].particles.typeid == colloid2_typeid))[0]
                          ncolloids = len(colloids)
                          # do the same for all the colloid subpopulations
                          colloid1 = np.where(traj[-1].particles.typeid == colloid1_typeid)[0]
                          ncolloid1 = len(colloid1)
                          colloid2 = np.where(traj[-1].particles.typeid == colloid2_typeid)[0]
                          ncolloid2 = len(colloid2)
                          if (ncolloid1 + ncolloid2) != ncolloids:
                            print("ERROR: ncolloid1 + ncolloid2 != ncolloids in gofr calc; gofr was NOT calculated")
                            exit(1)

                          # import all data into one dataframe
                          frame_dfs_all = []

                          edge_output = f'{data_outpath}/frame-edges/edgelist' # + <#>.csv in f90
                          #pos_output = f'{data_outpath}/frame-pos/positions_frame' # + <#>.csv in f90
                          for frame in range(nframes):
                              # loop through all frames
                              edge_file = edge_output+str(frame)+'.csv'
                              # import CSV data
                              edge_df = pd.read_csv(edge_file)
                              # rename colums as needed
                              edge_df = edge_df.rename(columns={"i": "source", "j": "target"})
                              edge_df.insert(loc=0, column='frame', value=frame)
                              frame_dfs_all.append(edge_df)


                          #allpos_df = pd.concat(pos_frame_dfs_all, ignore_index=True)
                          alledge_df = pd.concat(frame_dfs_all, ignore_index=True)

                          # network analysis
                          #for frame in range(nframes):
                          frame = nframes-1

                          # get particle positions for percolation measurement
                          pos = traj[frame].particles.position
                          typeID = traj[frame].particles.typeid
                          radii = 0.5*traj[frame].particles.diameter

                          df = alledge_df[alledge_df['frame'] == frame][["source", "target"]]

                          # create the network from edge list
                          g = nx.from_pandas_edgelist(df)

                          # if a node is not in the network, add it
                          for particle in range(ncolloids):
                            if (  not(   g.has_node(particle)   )  ):
                              g.add_node(particle)

                          # add node attributes
                          node_attrs = {
                              i: {
                                  "type": int(typeID[i]),
                                  "radius": float(radii[i]),
                                  "x": float(pos[i][0]),
                                  "y": float(pos[i][1]),
                                  "z": float(pos[i][2]),
                              }
                              for i in range(ncolloids)
                          }

                          nx.set_node_attributes(g, node_attrs)


                          graph_name = f"{data_outpath}graph_lastframe.graphml"
                          nx.write_graphml(g, graph_name)
                          print(f"Saved GraphML → {graph_name}")

                      ####################################
                      #### CONTINUE WITH THE ANALYSIS ####
                      ####################################

                      # load the graph
                      g = ig.Graph.Read_GraphML(graph_path)

                      if size == f'{int(R_C2)}-0':
                          results = analyze_topology(
                              g,
                              k_SS = R_C2,
                              k_SL = (2*R_C1*R_C2)/(R_C1+R_C2) if (2*R_C1*R_C2)/(R_C1+R_C2) else 1.0,
                              k_LL = R_C1 if R_C1 else 1.0,
                          )
                      else:
                          results = analyze_topology(
                              g,
                              k_SS = R_C1 if R_C1 else 1.0,
                              k_SL = (2*R_C1*R_C2)/(R_C1+R_C2) if (2*R_C1*R_C2)/(R_C1+R_C2) else 1.0,
                              k_LL = R_C2 if R_C2 else 1.0, 
                          )

                      # unweighted
                      backbone_comp = results['backbone_composition']
                      backbone_SS = backbone_comp['SS']
                      backbone_SL = backbone_comp['SL']
                      backbone_LL = backbone_comp['LL']
                      # weighted
                      backbone_comp_w = results['backbone_composition_w']
                      backbone_SS_w = backbone_comp_w['SS']
                      backbone_SL_w = backbone_comp_w['SL']
                      backbone_LL_w = backbone_comp_w['LL']

                      tetra_comp = results['tetra_comp']

                      results_dict = {'size-ratio'          :[size],
                                      'phi'                 :[phi],
                                      'composition'         :[comp],
                                      'D0'                  :[D0],
                                      'scaled_D0'           :[scaling],
                                      'kappa'               :[kappa],
                                      # TOPOLOGICAL BACKBONE (EBC >= P90)
                                      'n_backbone'          :results['n_backbone'],
                                      'backbone_SS'         :[backbone_SS],
                                      'backbone_SL'         :[backbone_SL],
                                      'backbone_LL'         :[backbone_LL],
                                      'backbone_SS_w'       :[backbone_SS_w],
                                      'backbone_SL_w'       :[backbone_SL_w],
                                      'backbone_LL_w'       :[backbone_LL_w],

                                      'n_tetra': results['n_tetra'],
                                      'frac_backbone_tetra': results['frac_backbone_tetra'],
                                      'frac_tetra_backbone': results['frac_tetra_backbone'],
                                      'frac_backbone_tetra_w': results['frac_backbone_tetra_w'],
                                      'frac_tetra_backbone_w': results['frac_tetra_backbone_w'],

                                      'tetra_comp_SS': tetra_comp['SS'],
                                      'tetra_comp_SL': tetra_comp['SL'],
                                      'tetra_comp_LL': tetra_comp['LL'],

                      }

                      # After computing pt_results = analyze_motifs(g, tetrahedra, t_info):
                      pt_results = results['pt_results']
                      tetrahedra_df = build_tetrahedra_dataframe(
                          pt_results,
                          config_label={
                              'size-ratio': size,
                              'phi': phi,
                              'composition': comp,
                              'D0': D0,
                              'scaled_D0': scaling,
                              'kappa': kappa,
                          }
                      )

                      res_df = pd.DataFrame(results_dict)
                      print(f' - topological analysis complete for: {R_C1}:{R_C2}, phi={phi}, {comp}, {D0}kT, kappa={kappa}')
                      system_dfs.append(res_df)

                      edge_data_df = results['edge_data_df']

                      for cluster_style in clustering_style:
                          cluster_path = f"{data_path}GMM/cluster_bridge_edges_{cluster_style}.csv"
                          if os.path.exists(cluster_path) == False:
                              print(f" - No cluster bridge data: {cluster_path}")
                              print(f"   ...skipping cluster comparison")
                              continue

                          else:
                              print(f' - Compare to GMM clustering: {cluster_style}') 
                              cluster_bridge_df = pd.read_csv(cluster_path)
                              u = cluster_bridge_df['source_particle'].to_numpy()
                              v = cluster_bridge_df['target_particle'].to_numpy()
                              u_min = np.minimum(u, v)
                              v_max = np.maximum(u, v)
                              edge_tuples = list(zip(u_min, v_max))
 
                              # are there any singly-connected clusters?
                              sc = cluster_bridge_df['source_cluster'].to_numpy()
                              tc = cluster_bridge_df['target_cluster'].to_numpy()
                              c_min = np.minimum(sc, tc)
                              c_max = np.maximum(sc, tc)
                              cluster_bridge_df['cluster_pair'] = list(zip(c_min, c_max))
                              pair_counts = cluster_bridge_df.groupby('cluster_pair').size().reset_index(name='n_bonds')
                              single_bond_pairs = pair_counts[pair_counts['n_bonds'] == 1]
                              single_edges_df = cluster_bridge_df[cluster_bridge_df['cluster_pair'].isin(single_bond_pairs['cluster_pair'])]
                              non_single_pairs = pair_counts[pair_counts['n_bonds'] > 1]['cluster_pair']
                              non_single_edges_df = cluster_bridge_df[
                                  cluster_bridge_df['cluster_pair'].isin(non_single_pairs)
                              ]
                              n_singlyconnected = len(single_bond_pairs)
                              #print('  - n_singly-connected:',len(single_bond_pairs))
                              frac_singlyconnected = len(single_bond_pairs) / len(pair_counts)
                              #print('  - frac singly-connected:',len(single_bond_pairs) / len(pair_counts))
                              single_edge_set = set(
                                  (min(int(u), int(v)), max(int(u), int(v)))
                                  for u, v in zip(single_edges_df['source_particle'],
                                                  single_edges_df['target_particle'])
                              )

                              # recalculate full set of EBC values
                              bridge_set = set(edge_tuples)
                              is_bridge = np.array([
                                  tuple(sorted(e.tuple)) in bridge_set
                                  for e in g.es
                              ])
                              mech_vals_raw = np.array(g.edge_betweenness())
                              N = g.vcount()
                              mech_vals = mech_vals_raw / (N * (N - 1) / 2)

                              # get all the correct node labels from the igraph
                              node_labels = g.vs['id'] if 'id' in g.vs.attributes() else g.vs['name']
                              node_labels = list(map(int, node_labels))

                              # match edges from cluster data and full graph
                              edge_to_eid = {
                                  (min(node_labels[u], node_labels[v]),
                                   max(node_labels[u], node_labels[v])): i
                                  for i, (u, v) in enumerate(e.tuple for e in g.es)
                              }
                              single_ebc = [
                                  mech_vals[edge_to_eid[(min(u,v), max(u,v))]]
                                   for u, v in zip(single_edges_df['source_particle'],
                                                  single_edges_df['target_particle'])
                                  if (min(u,v), max(u,v)) in edge_to_eid
                              ]
                              non_single_ebc = [
                                  mech_vals[edge_to_eid[(min(u,v), max(u,v))]]
                                  for u, v in zip(non_single_edges_df['source_particle'],
                                                  non_single_edges_df['target_particle'])
                                  if (min(u,v), max(u,v)) in edge_to_eid
                              ]
                              bridge_edge_set = set(
                                  (min(int(u), int(v)), max(int(u), int(v)))
                                  for u, v in zip(cluster_bridge_df['source_particle'],
                                                  cluster_bridge_df['target_particle'])
                              )

                              g_edges_particle = [
                                  (min(node_labels[u], node_labels[v]),
                                   max(node_labels[u], node_labels[v]))
                                  for u, v in (e.tuple for e in g.es)
                              ]
                              is_bridge = np.array([
                                  edge in bridge_edge_set
                                  for edge in g_edges_particle
                              ])

                              ## Optional checks:
                              #len(is_bridge) == len(mech_vals)  # should be True
                              #print("n bridge edges:", np.sum(is_bridge))

                              ####################
                              # isolate SB and MB
                              ####################

                              is_single = np.array([edge in single_edge_set for edge in g_edges_particle])
                              is_multi = is_bridge & ~is_single


                              ##############################
                              # save ECDF and CCDF ebc data:
                              ##############################

                              def ecdf(x):
                                  x = np.sort(np.asarray(x))
                                  y = np.arange(1, len(x)+1) / len(x)
                                  return x, y

                              def ccdf(x):
                                  x = np.sort(np.asarray(x))
                                  y = 1 - np.arange(1, len(x)+1) / len(x)
                                  return x, y

                              def make_dist_df(values, group_name, extra_meta=None):
                                  """
                                  Build long-form dataframe containing raw values + ECDF + CCDF.

                                  values         : 1D array of numbers
                                  group_name     : e.g. 'single bridges'
                                  extra_meta     : dict of additional labels
                                  """
                                  values = np.asarray(values)

                                  if len(values) == 0:
                                      return pd.DataFrame()

                                  xe, ye = ecdf(values)
                                  xc, yc = ccdf(values)

                                  df = pd.DataFrame({
                                      'group': group_name,
                                      'ebc': xe,         # sorted values
                                      'ecdf': ye,
                                      'ccdf': yc
                                  })

                                  # summary info repeated per row (useful later)
                                  df['n'] = len(values)
                                  df['mean'] = values.mean()
                                  df['median'] = np.median(values)

                                  return df

                              single_vals = mech_vals[is_single]
                              multi_vals  = mech_vals[is_multi]
                              non_vals    = mech_vals[~is_bridge]
                              bridge_vals = mech_vals[is_bridge]

                              df_single = make_dist_df(single_vals, 'single_bridge')
                              df_multi  = make_dist_df(multi_vals,  'multi_bridge')
                              df_non    = make_dist_df(non_vals,    'non_bridge') 
                              df_allbr  = make_dist_df(bridge_vals, 'all_bridge') 

                              df_cdf_dataset = pd.concat(
                                  [df_single, df_multi, df_non, df_allbr],
                                  ignore_index=True
                              )
                              df_cdf_dataset.insert(loc=0, column='cluster_style', value=cluster_style)
                              df_cdf_dataset.insert(loc=0, column='kappa', value=kappa)
                              df_cdf_dataset.insert(loc=0, column='scaled_D0', value=scaling)
                              df_cdf_dataset.insert(loc=0, column='D0', value=D0)
                              df_cdf_dataset.insert(loc=0, column='composition', value=comp)
                              df_cdf_dataset.insert(loc=0, column='phi', value=phi)
                              df_cdf_dataset.insert(loc=0, column='size-ratio', value=size)

                              cdf_list.append(df_cdf_dataset)

                              #################################
                              # Compare bridges and tetrahedra
                              #################################

                              # compare to motifs
                              eid_map = eid_to_motif_labels(g, results['tetrahedra'], results['t_info'])
                              bridge_motif_df = annotate_overlap_df(g, overlap_df, eid_map)

                              motif_counts = bridge_motif_df["motif_label"].value_counts()
                              motif_norms  = bridge_motif_df["motif_label"].value_counts(normalize=True)
                              bridge_motif_df = pd.DataFrame({
                                  'size-ratio': size,
                                  'phi': phi,
                                  'composition': comp,
                                  'D0': D0,
                                  'scaled_D0': scaling,
                                  'kappa': kappa,
                                  'cluster_style': cluster_style,
                                  'n_singlyconnected': n_singlyconnected,
                                  'frac_singlyconnected': frac_singlyconnected,
                                  'motif': motif_counts.index,
                                  'count': motif_counts.values,
                                  'norm': motif_norms.reindex(motif_counts.index).values
                              })
                              bridge_motifs_dfs.append(bridge_motif_df)

                              ########################################
                              # Compare bridges and EBC>=P90 backbone
                              ########################################

                              ## UNWEIGHTED EBC

                              backbone_edges = results['backbone_edges']
                              # undirected set
                              backbone_set = set(
                                  (min(u, v), max(u, v))
                                  for (u, v) in backbone_edges
                              )

                              backbone_edges_particle = [
                                  (min(node_labels[u], node_labels[v]),
                                   max(node_labels[u], node_labels[v]))
                                  for (u, v) in backbone_edges
                              ]
                              backbone_set = set(backbone_edges_particle)
                              mask = np.fromiter((e in backbone_set for e in edge_tuples), dtype=bool)
                              overlap_df = cluster_bridge_df[mask]

                              if len(overlap_df) != 0:
                                  overlap_df.insert(loc=0, column='cluster_style', value=cluster_style)
                                  overlap_df.insert(loc=0, column='kappa', value=kappa)
                                  overlap_df.insert(loc=0, column='scaled_D0', value=scaling)
                                  overlap_df.insert(loc=0, column='D0', value=D0)
                                  overlap_df.insert(loc=0, column='composition', value=comp)
                                  overlap_df.insert(loc=0, column='phi', value=phi)
                                  overlap_df.insert(loc=0, column='size-ratio', value=size)
                                  bridge_dfs.append(overlap_df)
                              else:
                                  #print(f' - size-rato: {size}, phi: {phi}, composition: {comp}, D0: {D0}, scaling: {scaling}, kappa: {kappa}')
                                  print(f'    No overlap between high EBC edges and GMM cluster bridges')


                              ## WEIGHTED EBC

                              backbone_edges_w = results['backbone_edges_w']
                              # undirected set
                              backbone_set_w = set(
                                  (min(u, v), max(u, v))
                                  for (u, v) in backbone_edges_w
                              )
                              #print('are weighted and unweighted edges the same?',(backbone_set_w == backbone_set))

                              backbone_edges_w_particle = [
                                  (min(node_labels[u], node_labels[v]),
                                   max(node_labels[u], node_labels[v]))
                                  for (u, v) in backbone_edges_w
                              ]
                              backbone_set_w = set(backbone_edges_w_particle)
                              mask_w = np.fromiter((e in backbone_set_w for e in edge_tuples), dtype=bool)
                              overlap_df_w = cluster_bridge_df[mask_w]
                              if len(overlap_df_w) != 0:
                                  overlap_df_w.insert(loc=0, column='cluster_style', value=cluster_style)
                                  overlap_df_w.insert(loc=0, column='kappa', value=kappa)
                                  overlap_df_w.insert(loc=0, column='scaled_D0', value=scaling)
                                  overlap_df_w.insert(loc=0, column='D0', value=D0)
                                  overlap_df_w.insert(loc=0, column='composition', value=comp)
                                  overlap_df_w.insert(loc=0, column='phi', value=phi)
                                  overlap_df_w.insert(loc=0, column='size-ratio', value=size)
                                  bridge_dfs_w.append(overlap_df_w)
                              else:
                                  #print(f' - size-rato: {size}, phi: {phi}, composition: {comp}, D0: {D0}, scaling: {scaling}, kappa: {kappa}')
                                  print(f'    No overlap between high EBC edges (weighted) and GMM cluster bridges')


                              ########################################################
                              # OPTIONAL: Save copy of bridge data next to the igraph
                              ########################################################
                              edge_df_all = build_bridge_edge_dataframe(
                                  g=g,
                                  cluster_bridge_df=cluster_bridge_df,
                                  size_ratio=size,
                                  phi=phi,
                                  composition=comp,
                                  D0=D0,
                                  scaled_D0=scaling,
                                  kappa=kappa,
                                  cluster_style=cluster_style
                              )
                              edge_df_all.to_csv(f"{data_outpath}/bridge_data_{plot_label}.csv", index=False)


                              #################################################
                              # ADD cluster bridge information to edge_data_df
                              #################################################

                              # 1. particle -> cluster map
                              #    - built from all bridge rows; assumes each particle belongs to one cluster
                              particle_to_cluster = {}
                              for row in cluster_bridge_df.itertuples(index=False):
                                  particle_to_cluster[int(row.source_particle)] = int(row.source_cluster)
                                  particle_to_cluster[int(row.target_particle)] = int(row.target_cluster)

                              # 2. count bonds per cluster-cluster pair
                              tmp = cluster_bridge_df.copy()
                              tmp["cluster_i"] = np.minimum(
                                  tmp["source_cluster"].astype(int),
                                  tmp["target_cluster"].astype(int)
                              )
                              tmp["cluster_j"] = np.maximum(
                                  tmp["source_cluster"].astype(int),
                                  tmp["target_cluster"].astype(int)
                              )
                              tmp["cluster_pair"] = list(zip(tmp["cluster_i"], tmp["cluster_j"]))
                              pair_counts = (
                                  tmp.groupby("cluster_pair")
                                     .size()
                                     .to_dict()
                              )

                              # 3. edge -> bridge classification
                              edge_to_bridge = {}
                              for row in cluster_bridge_df.itertuples(index=False):
                                  u = int(row.source_particle)
                                  v = int(row.target_particle)
                                  edge = (min(u, v), max(u, v))
                                  cpair = (
                                      min(int(row.source_cluster), int(row.target_cluster)),
                                      max(int(row.source_cluster), int(row.target_cluster))
                                  )
                                  n = pair_counts[cpair]
                                  edge_to_bridge[edge] = "single" if n == 1 else "multi"

                              # 4. annotate edge_data_df
                              def annotate_row(row):
                                  u = int(row["i"])
                                  v = int(row["j"])
                                  edge = (min(u, v), max(u, v))
                                  # bridge type
                                  bridge = edge_to_bridge.get(edge, "none")
                                  # cluster ids from particle map
                                  ci = particle_to_cluster.get(u, -1)
                                  cj = particle_to_cluster.get(v, -1)
                                  # if non-bridge internal bond, force same cluster if possible
                                  if bridge == "none":
                                     if ci == -1 and cj != -1:
                                          ci = cj
                                     elif cj == -1 and ci != -1:
                                          cj = ci
                                     elif ci == -1 and cj == -1:
                                          ci = cj = -1
                                     else:
                                          # if they disagree, keep source cluster of i
                                          cj = ci
                                  return pd.Series([bridge, ci, cj])

                              # and don't forget to remap the edge_df labels!!!
                              #node_labels = g.vs['id'] if 'id' in g.vs.attributes() else g.vs['name']
                              #node_labels = list(map(int, node_labels))
                              edge_data_df["i"] = edge_data_df["i"].map(lambda x: node_labels[int(x)])
                              edge_data_df["j"] = edge_data_df["j"].map(lambda x: node_labels[int(x)])

                              edge_data_df_c = edge_data_df.copy()
                              edge_data_df_c[["bridge", "cluster_i", "cluster_j"]] = (
                                  edge_data_df_c.apply(annotate_row, axis=1)
                              )

                              edge_data_df_c.insert(loc=0, column='cluster_style', value=cluster_style)
                              edge_data_df_c.insert(loc=0, column='kappa', value=kappa)
                              edge_data_df_c.insert(loc=0, column='scaled_D0', value=scaling)
                              edge_data_df_c.insert(loc=0, column='D0', value=D0)
                              edge_data_df_c.insert(loc=0, column='composition', value=comp)
                              edge_data_df_c.insert(loc=0, column='phi', value=phi)
                              edge_data_df_c.insert(loc=0, column='size-ratio', value=size)
                              edge_data_dfs.append(edge_data_df_c)


                              #################################################
                              # ADD cluster bridge information to tetrahdera_df
                              #################################################
                              node_path = f"{data_path}GMM/node_df_lcc.csv"
                              if os.path.exists(node_path) == False:
                                  print(f" - No node data: {node_path}")
                                  print(f"   Cannot save bridge data to tetrahedra_df")
                                  continue

                              else:
                                node_df_lcc = pd.read_csv(node_path)
                                types = node_df_lcc['TypeID'].to_numpy()

                                # Get bridge edge IDs
                                bridge_eids = cluster_bridge_eids(g, cluster_bridge_df)

                                # Update pt_results with bridge overlap
                                pt_results = results['pt_results']  # from earlier
                                add_overlap_to_motifs(pt_results, bridge_eids, 'bridges')

                                # Singly-connected bridges only
                                sc = cluster_bridge_df['source_cluster'].to_numpy()
                                tc = cluster_bridge_df['target_cluster'].to_numpy()
                                c_min = np.minimum(sc, tc)
                                c_max = np.maximum(sc, tc)
                                cluster_bridge_df['cluster_pair'] = list(zip(c_min, c_max))
                                pair_counts = cluster_bridge_df.groupby('cluster_pair').size().reset_index(name='n_bonds')
                                single_bond_pairs = pair_counts[pair_counts['n_bonds'] == 1]
                                single_edges_df = cluster_bridge_df[cluster_bridge_df['cluster_pair'].isin(single_bond_pairs['cluster_pair'])]
                                singly_bridge_eids = cluster_bridge_eids(g, single_edges_df)
                                add_overlap_to_motifs(pt_results, singly_bridge_eids, 'single_bridges')

                                # NOW build the tetrahedra.csv with all overlap info
                                tetrahedra_df = build_tetrahedra_dataframe(
                                    pt_results,
                                    config_label={
                                        'size-ratio': size,
                                        'phi': phi,
                                        'composition': comp,
                                        'D0': D0,
                                        'scaled_D0': scaling,
                                        'kappa': kappa,
                                        'cluster_style': cluster_style,
                                    }
                                )
                                tetra_dfs.append(tetrahedra_df)



all_results_df = pd.concat(system_dfs)
all_results_df.to_csv(f'{data_outpath}/topology.csv',index = False)
print('All results saved to "topology.csv"')

all_tetra_df = pd.concat(tetra_dfs, ignore_index=True)
all_tetra_df.to_csv(f'{data_outpath}/tetrahedra.csv',index = False)
print('Detailed tetrahedral structure results saved to "tetrahedra.csv"')

if len(bridge_motifs_dfs) != 0:
  bridge_motif_results_df = pd.concat(bridge_motifs_dfs)
  bridge_motif_results_df.to_csv(f'{data_outpath}/motif-bridges.csv',index = False)
  print('GMM cluster bridges classified into singly-connected and geometric motifs and saved to "motif-bridges.csv"')

if len(bridge_dfs) != 0:
  bridge_results_df = pd.concat(bridge_dfs)
  bridge_results_df.to_csv(f'{data_outpath}/EBC-bridges.csv',index = False)
  print('Overlap between high EBC and GMM cluster bridges saved to "EBC-bridges.csv"')

if len(bridge_dfs_w) != 0:
  bridge_results_df_w = pd.concat(bridge_dfs_w)
  bridge_results_df_w.to_csv(f'{data_outpath}/EBCw-bridges.csv',index = False)
  print('Overlap between high EBC (weighted) and GMM cluster bridges saved to "EBCw-bridges.csv"')

if len(edge_data_dfs) != 0:
  edge_labels_df = pd.concat(edge_data_dfs)
  edge_labels_df.to_csv(f'{data_outpath}/all_edge_data.csv',index = False)
  print('Labeled edge data saved to "all_edge_data.csv"')

if len(cdf_list) != 0:
  all_length_scales_df = pd.concat(cdf_list)
  all_length_scales_df.to_csv(f"{data_outpath}/ebc_ccdf-ecdf_data.csv", index=False)
  print('All EBC CCDF and ECDF data saved to "ebc_ccdf-ecdf_data.csv"')
