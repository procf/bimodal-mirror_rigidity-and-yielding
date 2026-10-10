## Apply GMM-based clustering to the particle 
## trajectories from a colloidal gel simulation
## NOTE: before running this script make sure the
##       network has been created with 
##       sim-build-particle-network.py
"""
## Cluster the network using GMM
##
## How to use:
##     python sim-build-particle-network.py brush_density
##
## INPUT:
## use lcc network (this is the elastic part) 
##    - node_df_lcc 
##        # keys : 'Tag#', 'x', 'y', 'z', 'Radius', 'TypeID'  
##        saved to data_outpath+'/node_df_lcc.csv' 
##    - edge_df_lcc 
##        # keys : 'source', 'target', 'edge_type'  
##        saved to data_outpath+'/edgelist_lcc.csv'
##    - g 
##        # full network: 'pos' [x,y,z], 'radius', 'type_id', 'edge_type' 
##        saved to data_outpath+'/lcc-particle-network_g.pkl'
##    - lcc_df 
##        saved to data_outpath+'/full_net.csv'
##        # keys : 'cut_off', 'n_components', 'lcc_size', 'ncolloids', 'avg_degree', 
##        saved to data_outpath+'/lcc.csv'
##    - angle_df_lcc
##        # keys : 'angle_id', 'edge_1', 'type_1', 'edge_2', 'type_2', 'angle_degree' 
##        saved to data_outpath+'/angle_dist_lcc.csv'
##
## OUTPUT:
##    - embeddings10d 
##        saved to data_outpath+'/lcc_node_embeddings_10D.csv' (requires mapper to reload as a graph)
##    - bic_df 
##        # keys : 'k' 'BIC'  
##        saved to data_outpath+'/bic_scores.csv'
##    - clusters_df 
##        # keys : 'node', 'cluster' 
##        saved to 'clusters.csv'
##    - weighted_cluster_edges_df 
##        # keys : 'source_cluster' 'target_cluster' 'weight' 
##        saved to data_outpath+'/weighted_cluster_edges.csv'
##    - clustered_df, lcc_clustered_df
##        # keys : 'Tag#', 'x', 'y', 'z', 'Radius', 'TypeID', 'Cluster'
##        saved to data_outpath+'/clustered_df.csv'
##        saved to data_outpath+'/lcc_clustered_df.csv'
##    - cluster_diameters_df
##        # keys : 'Cluster' 'Diameter' 'Physical Diameter'
##        saved to data_outpath+'/cluster_diameters_{clustering_style}.csv'
##    - G_clusters
##        # clustered LCC network : 'cluster_id', 'size', 'diameter', 'physical_diameter'
##        saved to data_outpath+'/lcc-cluster-network_G_clusters.pkl'
##    - cluster_cauchyborn_df
##        # keys : 'average_xi', 'system_volume', 'phi_C_particles', 'phi_C_hull_sum', 
##                   'phi_C_spheres', 'z_c_old', 'z_c', 'z_c_weighted', 
##        saved to data_outpath+'/cluster_cauchy_born_full.csv'
##    - iso_df -- FILTERED for clusters with z>= 2.4
##        # keys : "average_xi", "phi_C_spheres", 
##                   "z_c", "z_c_weighted", G_c, G_b, G_prime
##        saved to data_outpath+'/cluster_cauchy_born_iso-clusters.csv'
"""
## (Rob Campbell)


########################
""" MODULE LIBRARY """
########################
# load data and build a network
import numpy as np
import pandas as pd
import networkx as nx
import os
import sys
import pickle

# for mapping bond information into network space
from node2vec import Node2Vec
import umap

# for clustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

# for faster BIC minimization
import multiprocessing
from multiprocessing import Pool

# for cluster analysis
from collections import defaultdict
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

# for plotting
import matplotlib.pyplot as plt

# for polydisperse cluster diameter
from scipy.spatial.distance import pdist, squareform

##########################
""" INPUT PARAMETERS """
##########################

#sys args: gel_file bimodal_bool data_file clustering_style

clustering_style = str(sys.argv[4])
print(f'CLUSTERING STYLE: {clustering_style}')
#clustering_style: choose 'standard' or 'weighted'

gel_file = str(sys.argv[1]) 
kappa_m = 60
bimodal_bool = sys.argv[2].strip().lower() == 'true'
if bimodal_bool == False:
    print(f' - bimodal_bool "{bimodal_bool}" = False')

if bimodal_bool == True:
  R_C2 = 2
  colloid1_typeid = 1
  colloid2_typeid = 2
else:
  colloid_typeid = 0
R_C1 = 1
PBC = True
cut_off = 3/kappa_m #round((d_g+(2*hb))/R_C_real,3)
R_C = R_C1

# filepath to folder where data files will be created
posedge_outpath = str(sys.argv[3]) #'data_'+tag
data_outpath = posedge_outpath+'/GMM'
if PBC == False:
  data_outpath = data_outpath+'/non-periodic'

# create "data" subfolder if it doesn't exit
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)

#################
""" LOAD DATA """
#################

edge_df_lcc = pd.read_csv(data_outpath+'/edgelist_lcc.csv')
node_df_all = pd.read_csv(data_outpath+'/node_df_all.csv')
node_df_lcc = pd.read_csv(data_outpath+'/node_df_lcc.csv')
pos_lcc = node_df_lcc[['Tag#','x','y','z']]
net_df = pd.read_csv(data_outpath+'/net.csv')
ncolloids = net_df['ncolloids'].values[0]
lcc_df = pd.read_csv(data_outpath+'/lcc.csv')
ncolloids_lcc = lcc_df['ncolloids'].values[0]
L_X = lcc_df['L_X'].values[0]
L_Y = lcc_df['L_Y'].values[0]
L_Z = lcc_df['L_Z'].values[0]
Lbox = np.array([L_X, L_Y, L_Z])
with open(data_outpath + '/lcc-particle-network_g.pkl', 'rb') as fr:
    g = pickle.load(fr)


#####################
""" VECTORIZATION """
#####################
print(' - Starting GMM vectorization...')

## STEP 1: vectorize edgelist (convert to network space)
##   - takes ~1min for 1230 colloids
##   - saves data to CSV

# ensure node labels are integers
g = nx.relabel_nodes(g, lambda x: int(x))
nodes_list = list(g.nodes())

# premap radii
radius_dict = dict(zip(node_df_lcc['Tag#'], node_df_lcc['Radius']))
radii = np.array([radius_dict[int(node)] for node in nodes_list])
std = radii.std()
if np.isclose(std, 0):
    radii_norm = np.zeros_like(radii, dtype=float)
else:
    radii_norm = (radii - radii.mean()) / std
# map node -> radius
node_to_radius_norm = {int(node): radii_norm[i] for i, node in enumerate(nodes_list)}
node_to_radius_raw = {int(node): radii[i] for i, node in enumerate(nodes_list)}

if clustering_style == 'standard':

  embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D.csv'
  if os.path.exists(embeddings_file_10D) == False:

    # (a)  convert from 3D real space to 128D network space with node2vec
    n_dims=128 # 128 dimensions to match Nabizadeh 2023
    node2vec = Node2Vec(g, dimensions=n_dims, walk_length=80, num_walks=20, workers=1)
    model = node2vec.fit(window=10, min_count=1)
    node_embeddings = model.wv
    print('   - '+str(n_dims)+'D node embeddings created')

    # [optional] save the full embeddings
    #embeddings_file_128D = data_outpath+'/node_embeddings.txt'
    #node_embeddings.save_word2vec_format(embeddings_file_128D)

    # (b)  dimensional reduction with UMAP (convert 128D to 10D)
    #      [optional] tune n_components (or other params)
    #                 see: https://umap-learn.readthedocs.io/en/latest/parameters.html
    #embeddings_128d = np.array([node_embeddings[str(node)] for node in g.nodes()])
    #embeddings_128d = np.array([
    #    node_embeddings[str(int(node))] 
    #    #for node in g.nodes() 
    #    for node in nodes_list 
    #    if str(int(node)) in node_embeddings
    #])
    valid_nodes = [
        int(node) for node in nodes_list
        if str(int(node)) in node_embeddings
    ]

    embeddings_128d = np.array([
        node_embeddings[str(node)] for node in valid_nodes
    ])

    n_components = 10
    n_neighbors = 12
    min_dist = 0
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embeddings_10d = reducer.fit_transform(embeddings_128d)

    # make sure no nodes were lost
    assert len(nodes_list) == embeddings_10d.shape[0]

    print('   - embeddings converted from '+str(n_dims)+'D to 10D')

    # save the 10D embeddings
    embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D.csv'
    f = open(embeddings_file_10D,'w')
    for j, node in enumerate(valid_nodes):
        f.write("%d," % node)
        for i in range(n_components - 1):
            f.write("%f," % embeddings_10d[j, i])
        f.write("%f\n" % embeddings_10d[j, n_components - 1])
    f.close()
    print('   - EMBEDDINGS SAVED TO:',embeddings_file_10D)

  else:
      print('   - WARNING: 10D embeddings already exist, not re-embedding in 128D and re-performing dimensional reduction to 10D for this data.')


elif clustering_style == 'weighted':
  if bimodal_bool != True:
      print('WARNING: Weighted and standard clustering are the same for monomodal data. This system is monomodal.')

  # weight edges by Derjaguin approx
  # Node2Vec will apply weights saved as 'weight' ...probably
  for u, v in g.edges():
    r_u = radius_dict[int(u)]
    r_v = radius_dict[int(v)]
    
    g[u][v]['weight'] = (2*r_u*r_v)/(r_u + r_v)  # Derjaguin approx (average that includes curvature effect) 

  embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D_w.csv'
  if os.path.exists(embeddings_file_10D) == False:

    # (a)  convert from 3D real space to 128D network space with node2vec
    n_dims=128 # 128 dimensions to match Nabizadeh 2023
    node2vec = Node2Vec(g, dimensions=n_dims, walk_length=80, num_walks=20, workers=1, weight_key='weight')
    model = node2vec.fit(window=10, min_count=1)
    node_embeddings = model.wv
    print('   - '+str(n_dims)+'D node embeddings created')

    # [optional] save the full embeddings
    #embeddings_file_128D = data_outpath+'/node_embeddings.txt'
    #node_embeddings.save_word2vec_format(embeddings_file_128D)

    # (b)  dimensional reduction with UMAP (convert 128D to 10D)
    #      [optional] tune n_components (or other params)
    #                 see: https://umap-learn.readthedocs.io/en/latest/parameters.html
    #embeddings_128d = np.array([node_embeddings[str(node)] for node in g.nodes()])
    #embeddings_128d = np.array([
    #    node_embeddings[str(int(node))]
    #    #for node in g.nodes()
    #    for node in nodes_list
    #    if str(int(node)) in node_embeddings
    #])
    valid_nodes = [
        int(node) for node in nodes_list
        if str(int(node)) in node_embeddings
    ]

    embeddings_128d = np.array([
        node_embeddings[str(node)] for node in valid_nodes
    ])

    n_components = 10
    n_neighbors = 12
    min_dist = 0
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embeddings_10d = reducer.fit_transform(embeddings_128d)

    # make sure no nodes were lost
    assert len(nodes_list) == embeddings_10d.shape[0]

    print('   - embeddings converted from '+str(n_dims)+'D to 10D')

    # save the 10D embeddings
    embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D_w.csv'
    f = open(embeddings_file_10D,'w')
    for j, node in enumerate(valid_nodes):
        f.write("%d," % node)
        for i in range(n_components - 1):
            f.write("%f," % embeddings_10d[j, i])
        f.write("%f\n" % embeddings_10d[j, n_components - 1])
    f.close()
    print('   - EMBEDDINGS SAVED TO:',embeddings_file_10D)

  else:
      print('   - WARNING: weighted 10D embeddings already exist, not re-embedding in 128D and re-performing dimensional reduction to 10D (with weights) for this data.')

else:
  print(f'ERROR: clustering_style must be "standard" or "weighted", value "{clustering_style}" not accepted.')
  exit(1)

####################
""" MINIMIZE BIC """
####################
## STEP 2: minimize BIC to find optimal number of clusters
## takes 10-15min for 1000 k_values and 1230 colloids
print('   - Minimizing BIC (may take 20+min)...')

# A truly rigorous search tests all possibilities from 1 cluster to separated particles: k = range(1,ncolloids_lcc)
#   but most systems of our size have k_optimal < 1000 ; plot BIC curve to confirm minimum was found
min_nodes_per_cluster = 3 #10
k_max = min(ncolloids_lcc//min_nodes_per_cluster, min(1000, ncolloids_lcc))  # safely capped 
k_values = range(1, k_max)


# load embeddings
if clustering_style == 'standard':
  embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D.csv'
elif clustering_style == 'weighted':
  embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D_w.csv'

mapper = np.genfromtxt(embeddings_file_10D, delimiter=',')
embeddings_10d = mapper[:,1:]

bic_file = f'{data_outpath}/bic_scores_{clustering_style}.csv'
if os.path.exists(bic_file) == False:
    # Fit GMM for each k and calculate BIC
    bic_scores = []
    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(embeddings_10d)
        bic_scores.append(gmm.bic(embeddings_10d))

    # [optional] save BIC scores to CSV
    bic_df = pd.DataFrame({
        'k': list(k_values),
        'BIC': bic_scores
    })
    bic_df.to_csv(bic_file, index=False)
    print('   - BIC VALUES SAVED TO:',bic_file)

else:
    print('   - WARNING: BIC scores already exist, not re-minimizing BIC for this data.')


# Load BIC data to find the optimal k 
# (the number of clusters that give with the minimum BIC score)

bic_file = f'{data_outpath}/bic_scores_{clustering_style}.csv'
bic_results_df = pd.read_csv(bic_file)
bic_scores = list(bic_results_df['BIC'])
optimal_k = int(bic_results_df.loc[bic_results_df['BIC'].idxmin(), 'k'])
print(f"   - Optimal number of clusters (k) for 10D embeddings: {optimal_k}")

## [OPTIONAL] Plot the BIC scores
plt.figure(figsize=(8, 6))
plt.plot(bic_results_df['k'], bic_results_df['BIC'], marker='o', label='embeddings (10D)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('BIC Score')
plt.title('BIC Score vs. Number of Clusters')
plt.axvline(x=optimal_k, c='royalblue')
plt.text(x=optimal_k+10, y=10000, s='k='+str(optimal_k), c='royalblue')
plt.legend()
plt.savefig(f'{data_outpath}/BIC_minimization_{clustering_style}.pdf', format='pdf', dpi=300, bbox_inches='tight')


#####################################
""" GMM CLUSTERING WITH OPTIMAL_K """
#####################################
## STEP 3: generate the cluster network from optimal_k

# load embeddings
if clustering_style == 'standard':
  embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D.csv'
elif clustering_style == 'weighted':
  embeddings_file_10D = data_outpath+'/lcc_node_embeddings_10D_w.csv'

mapper = np.genfromtxt(embeddings_file_10D, delimiter=',')
embeddings_10d = mapper[:,1:]

# Fit GMM with the optimal number of clusters
gmm_final = GaussianMixture(n_components=optimal_k, random_state=0, 
                            covariance_type='full', n_init=10).fit(embeddings_10d)

# Predict the cluster for each node
# `cluster_labels` is an array of cluster labels
cluster_labels = gmm_final.predict(embeddings_10d)

# [optional] reload network information if you do not have it from above
#g = pickle.load(data_outpath+'/particle-network_G.pkl')
#g = nx.relabel_nodes(g, lambda x: int(x))

# match nodes to their assigned clusters and collect cluster information
c = 0                              # index for looping through cluster_labels
cluster_dict = defaultdict(dict)   # cluster dictionary #1 (index:'node','cluster')
cluster_c = 0                      # cluster index
Clusters = {}                      # cluster dictionary #2 (index:nodes)
Nodes= []                          # list to collect nodes
Cluster_index = []                 # list to collect cluster indices

for node in g.nodes():
    # loop through the GMM results with this index
    idx = cluster_labels[c]
    # collect node and cluster information
    cluster_dict[cluster_c]['node'] = node
    cluster_dict[cluster_c]['cluster'] = idx
    Nodes.append(node)
    Cluster_index.append(idx+1) # start cluster numbering from 1 not 0
    # group nodes into clusters
    if idx in Clusters:
        Clusters[idx].add(node)
    else:
        Clusters[idx] = set()
        Clusters[idx].add(node)
        pass
    c += 1
    cluster_c += 1
    
Final_Cluster_Dict = {'Cluster':Cluster_index,
            'Node':Nodes}
clusters_df = pd.DataFrame.from_dict(Final_Cluster_Dict)

# save clustered data
cluster_file = f'{data_outpath}/clusters_{clustering_style}.csv'
clusters_df.to_csv(cluster_file,index = False)

print('   - clustering complete |',str(optimal_k),'clusters (#1-'+str(optimal_k)+')')
#clusters_df.head()

# group nodes into clusters
clusters = clusters_df['Cluster'].to_numpy()
nodes = clusters_df['Node'].to_numpy()

# use ALL colloids, unclustered particles (colloids not in the LCC) will be assigned cluster 0
sorted_clusters = np.zeros(ncolloids)
sorted_clusters[nodes] = clusters

# save full clustered network data
clustered_df = node_df_all.copy()
clustered_df['Cluster'] = sorted_clusters
clustered_df.to_csv(f'{data_outpath}/clustered_df_{clustering_style}.csv',index = False)

# remove unclustered particles (cluster = 0)
sorted_clusters = sorted_clusters[sorted_clusters != 0]

lcc_clustered_df = node_df_lcc.copy()
lcc_clustered_df['Cluster'] = sorted_clusters
lcc_clustered_df.to_csv(f'{data_outpath}/lcc_clustered_df_{clustering_style}.csv',index = False)

#######
## STEP 4: Verify clustering AND labeling
# Check that clusters are actually connected on the graph
#   - disconnected clusters suggest either:
#      (a) labeling was incorrect
#      (b) clusters need to be manually cleaned (outliers reassigned correctly)

def unwrap_cluster_positions(positions_df, box, edges, global_nodes, start_nodes=None):
    """
    Unwraps positions for the given node set (global_nodes) *and* the provided edges.
    This function unwraps each connected component inside the provided node set separately.
    Returns ndarray shape (n_nodes, 3) in the same order as global_nodes.
    """
    # map global Tag# → local index
    global_to_local = {gid: idx for idx, gid in enumerate(global_nodes)}

    # local edges (only those that connect two nodes in this cluster)
    local_edges = [
        (global_to_local[i], global_to_local[j])
        for (i, j) in edges
        if (i in global_to_local and j in global_to_local)
    ]

    # reorder positions according to global_nodes -> shape (n,3)
    pos_df_indexed = positions_df.set_index("Tag#")
    positions = pos_df_indexed.loc[global_nodes][["x","y","z"]].values.astype(float)

    n = len(positions)
    unwrapped = positions.copy()
    visited = np.zeros(n, dtype=bool)

    # build adjacency list for local indices
    adj = [[] for _ in range(n)]
    for a, b in local_edges:
        adj[a].append(b)
        adj[b].append(a)

    # find connected components in the local graph (indices)
    comps = []
    for idx in range(n):
        if not visited[idx]:
            # BFS/DFS to get this component
            comp_nodes = []
            stack = [idx]
            visited[idx] = True
            while stack:
                u = stack.pop()
                comp_nodes.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp_nodes)

    # For each component, perform unwrapping starting from an arbitrary node in that component
    for comp in comps:
        # pick a root
        root = comp[0]
        comp_visited = {root}
        queue = [root]
        while queue:
            i = queue.pop(0)
            for j in adj[i]:
                if j not in comp_visited:
                    # minimum-image delta using box
                    delta = positions[j] - positions[i]
                    delta -= box * np.round(delta / box)
                    unwrapped[j] = unwrapped[i] + delta
                    comp_visited.add(j)
                    queue.append(j)
        # At this point, the component nodes are internally unwrapped relative to each other

    return unwrapped


# functions for computing cluster volume
def cluster_max_distance(unwrapped_positions, radii=None): 
    dist_matrix = squareform(pdist(unwrapped_positions))
    
    if radii is None:
        return np.max(dist_matrix)
    else:
        radii_sum = radii[:, None] + radii[None, :]
        return np.max(dist_matrix + radii_sum)


def cluster_convex_hull_volume(unwrapped_positions):
    """Return convex hull volume of the cluster (requires at least 4 non-coplanar points)."""
    if len(unwrapped_positions) < 4:
        return 0.0
    try:
        hull = ConvexHull(unwrapped_positions)
        return float(hull.volume)
    except Exception:
        return 0.0

# Function to compute the angle between two vectors in 3D
def angle_between_vectors(v1, v2):
    # Normalize vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # Dot product and calculate the angle
    dot_product = np.dot(v1_u, v2_u)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def pbc_vec(a, b, box):
    dv = b - a
    return dv - box * np.round(dv / box)


def find_nearest_cluster(component_nodes, df, current_cluster_id, cluster_trees,
                         graph, position_columns=('x', 'y', 'z')):
    """
    Find the best cluster to reassign a disconnected fragment to.

    Strategy (option a, edge-based):
      1. Restrict candidates to clusters that share at least one edge in
         `graph` with the fragment. This guarantees the merged cluster's
         induced subgraph will be connected.
      2. Among those, pick the cluster with the most edges to the fragment
         (strongest topological link).
      3. Break ties by spatial proximity using cluster_trees.

    Falls back to pure spatial nearest if no edge-sharing candidate exists.
    This shouldn't happen when `graph` restricted to `df` is one LCC, but is
    included for robustness.
    """
    component_nodes = set(int(n) for n in component_nodes)

    # Tag# -> Cluster lookup (fast, built once per call)
    node_to_cluster = dict(zip(df['Tag#'].astype(int),
                               df['Cluster'].astype(int)))

    # Count edges from fragment to each external cluster
    edge_counts = {}
    for n in component_nodes:
        if n not in graph:
            continue
        for nbr in graph.neighbors(n):
            nbr_int = int(nbr)
            if nbr_int in component_nodes:
                continue
            nbr_cluster = node_to_cluster.get(nbr_int)
            if nbr_cluster is None or nbr_cluster == current_cluster_id:
                continue
            edge_counts[nbr_cluster] = edge_counts.get(nbr_cluster, 0) + 1

    if edge_counts:
        # Primary: most edges. Secondary: spatial nearest.
        max_edges = max(edge_counts.values())
        candidates = [c for c, k in edge_counts.items() if k == max_edges]

        if len(candidates) == 1:
            return candidates[0]

        comp_xyz = df[df['Tag#'].isin(component_nodes)][list(position_columns)].values
        best_cluster, best_dist = None, np.inf
        for c in candidates:
            d = cluster_trees[c].query(comp_xyz)[0].min()
            if d < best_dist:
                best_dist, best_cluster = d, c
        return best_cluster

    # Fallback: pure spatial nearest (defensive — shouldn't be hit)
    print(f"     NOTE: fragment of cluster {current_cluster_id} has no graph "
          f"neighbors outside itself; falling back to spatial nearest.")
    comp_xyz = df[df['Tag#'].isin(component_nodes)][list(position_columns)].values
    min_dist, nearest_cluster = np.inf, None
    for cluster, tree in cluster_trees.items():
        cluster = int(cluster)
        if cluster == current_cluster_id:
            continue
        d = tree.query(comp_xyz)[0].min()
        if d < min_dist:
            min_dist, nearest_cluster = d, cluster
    return nearest_cluster


# Check clustering and collect cluster data
connected_clusters = 0
unconnected_clusters = 0
unconnected_cluster_list = []
notes_list = []

# Ensure cluster IDs are ints (avoid float .0 keys)
lcc_clustered_df['Cluster'] = lcc_clustered_df['Cluster'].astype(int)

cluster_nodes_unique = np.sort(lcc_clustered_df['Cluster'].unique())

# Build initial KD-trees (updated once per cluster)
def build_cluster_trees(df):
    return {
        cid: cKDTree(df[df['Cluster'] == cid][['x','y','z']].values)
        for cid in df['Cluster'].unique()
    }

cluster_trees = build_cluster_trees(lcc_clustered_df)

# get ready to add more clusters if needed during reclustering
next_cluster_id = lcc_clustered_df['Cluster'].max() + 1


# check for disconnected clusters and update cluster assignment as needed:
for cluster_id in cluster_nodes_unique:
    mask = lcc_clustered_df['Cluster'] == cluster_id
    cluster_nodes = lcc_clustered_df.loc[mask, 'Tag#'].values

    # subgraph for this cluster
    cluster_subgraph = g.subgraph(cluster_nodes)

    # ensure node labels are integers
    cluster_subgraph = nx.relabel_nodes(cluster_subgraph, lambda x: int(x))

    # skip fully connected clusters
    if nx.is_connected(cluster_subgraph):
        connected_clusters += 1
        continue

    # split or reassign disconnected components
    unconnected_clusters += 1
    notes = []

    components = list(nx.connected_components(cluster_subgraph))
    sizes = np.array([len(c) for c in components])
    total_size = sizes.sum()

    # find the LCC
    lcc_nodes = max(components, key=len)

    # get information about the disconnect
    lcc_size = len(lcc_nodes)
    n_disconnected = total_size - lcc_size
    percent_disconnected = round(n_disconnected / total_size * 100, 2)

    if percent_disconnected <= 10:
        correction = "reassign"

        # loop through each cluster cc
        for comp_nodes in components:
            comp_nodes = set(comp_nodes)

            # keep the LCC of this cluster unchanged
            if comp_nodes == lcc_nodes:
                continue

            # reassign small components to the nearest other cluster
            nearest = find_nearest_cluster(
                component_nodes=comp_nodes,
                df=lcc_clustered_df,
                current_cluster_id=cluster_id,
                cluster_trees=cluster_trees,
                graph=g
            )
            notes.append(
                f"Moved {len(comp_nodes)}-particle fragment from Cluster {cluster_id} → {nearest}"
            )

            lcc_clustered_df.loc[
                lcc_clustered_df['Tag#'].isin(comp_nodes),
                'Cluster'
            ] = nearest

        # after all reassingments, update KD-trees 
        cluster_trees = build_cluster_trees(lcc_clustered_df)

    else:
        correction = "split"
        added_clusters = 0

        for comp_nodes, comp_size in zip(components, sizes):

            comp_nodes = set(comp_nodes)
            size_fraction = comp_size / total_size

            # keep the LCC as this cluster
            if comp_nodes == lcc_nodes:
                continue

            # save large components as new clusters with a new cluster ID
            if size_fraction > 0.10:

                lcc_clustered_df.loc[
                    lcc_clustered_df['Tag#'].isin(comp_nodes),
                    'Cluster'
                ] = next_cluster_id

                notes.append(
                    f"Created new cluster {next_cluster_id} from {len(comp_nodes)}-particle fragment of Cluster {cluster_id}"
                )

                next_cluster_id += 1
                added_clusters += 1

            # reassign small components to neighboring clusters
            else:
                cluster_trees = build_cluster_trees(lcc_clustered_df)
                nearest = find_nearest_cluster(
                    component_nodes=comp_nodes,
                    df=lcc_clustered_df,
                    current_cluster_id=cluster_id,
                    cluster_trees=cluster_trees,
                    graph=g
                )
                notes.append(
                    f"Reassigned tiny {len(comp_nodes)}-particle fragment of Cluster {cluster_id} → {nearest}"
                )

                lcc_clustered_df.loc[
                    lcc_clustered_df['Tag#'].isin(comp_nodes),
                    'Cluster'
                ] = nearest

        notes.append(f"Added {added_clusters} new clusters")
        # Update KD-trees after reassignments
        cluster_trees = build_cluster_trees(lcc_clustered_df)

    unconnected_cluster_list.append({
        'Cluster': cluster_id,
        'Size': len(cluster_subgraph),
        'LCC_Size': lcc_size, 
        #'LCC_Diameter': topological_diameter,
        'N_Disconnected': n_disconnected,
        '%_Disconnected': percent_disconnected,
        'correction': correction,
        'notes':notes
        })

if unconnected_clusters == 0:
    print(f"     {connected_clusters} Clusters (all fully-connected)")
else:
    percent_connection_error = round(unconnected_clusters/(unconnected_clusters+connected_clusters),2)*100
    print('     WARNING: disconnected clusters suggests an error occured...') 
    print(f"     {unconnected_clusters} Disconnected Clusters ({percent_connection_error}% clustering error)")

    # Convert the disconnected cluster data to a df for easy viewing
    unconnected_clusters_df = pd.DataFrame(unconnected_cluster_list)
    disconnected_data = unconnected_clusters_df[['Cluster','Size','LCC_Size','N_Disconnected','%_Disconnected','correction']]
    notes_array = unconnected_clusters_df['notes'].to_numpy()
    clusters = pd.unique(unconnected_clusters_df['Cluster'])
    for c in range(len(clusters)):
        cluster = clusters[c]
        print(disconnected_data.loc[disconnected_data['Cluster'] == cluster])
        notes_cluster = notes_array[c] 
        for note in notes_cluster:
            print(f"     -- {note}")

    disconnected_data.to_csv(f'{data_outpath}/unconnected_clusters_{clustering_style}.csv',index = False)

    # find new cluster info
    cluster_nodes_unique = np.unique(lcc_clustered_df['Cluster'])
    print(f"     ---")
    print(f"     {len(cluster_nodes_unique)} Clusters after processing (all fully-connected)")


physical_cluster_diameters = {}
cluster_diameters = {}
cluster_hull_volumes = {}
cluster_particle_counts = {}
cluster_inner_degree = {}
cluster_center_x = {}
cluster_center_y = {}
cluster_center_z = {}
cluster_angle_dfs = []

cluster_type1_counts = {}
cluster_type2_counts = {}
cluster_bonds_SS = {}
cluster_bonds_SL = {}
cluster_bonds_LL = {}

for cluster_id in cluster_nodes_unique:
    mask = lcc_clustered_df['Cluster'] == cluster_id
    cluster_nodes = lcc_clustered_df.loc[mask, 'Tag#'].values
    cluster_pos_df = lcc_clustered_df.loc[mask, ['Tag#','x','y','z','Radius','TypeID']]

    # subgraph for this cluster
    cluster_subgraph = g.subgraph(cluster_nodes)

    # ensure node labels are integers
    cluster_subgraph = nx.relabel_nodes(cluster_subgraph, lambda x: int(x))

    # Count bond types
    n_SS = 0
    n_SL = 0
    n_LL = 0

    # get typeid lookup
    typeid_dict = cluster_pos_df.set_index("Tag#")['TypeID'].to_dict()

    for u, v in cluster_subgraph.edges():
        t_u = typeid_dict[int(u)]
        t_v = typeid_dict[int(v)]
    
        if t_u == 1 and t_v == 1:
            n_SS += 1
        elif t_u == 2 and t_v == 2:
            n_LL += 1
        else:
            n_SL += 1

    # Choose geometry node set: if the cluster is connected, use it whole;
    # otherwise fall back to its LCC for consistency with topological_diameter.
    # (With the edge-based reassignment in find_nearest_cluster this branch
    # should rarely fire, but it keeps physical/topological/hull/n_particles
    # all referring to the same point set when it does.)
    if nx.is_connected(cluster_subgraph):
        geometry_subgraph = cluster_subgraph
    else:
        sub_lcc = max(nx.connected_components(cluster_subgraph), key=len)
        geometry_subgraph = cluster_subgraph.subgraph(sub_lcc)
        n_disc = cluster_subgraph.number_of_nodes() - geometry_subgraph.number_of_nodes()
        print(f"     NOTE: cluster {cluster_id} geometry restricted to cluster LCC "
              f"({n_disc} of {cluster_subgraph.number_of_nodes()} nodes excluded)")

    geometry_nodes = list(geometry_subgraph.nodes())
    geometry_edges = list(geometry_subgraph.edges())

    # unwrap chosen geometry nodes
    if len(geometry_nodes) > 0:
        if PBC == True:
            unwrap_pos = unwrap_cluster_positions(cluster_pos_df, Lbox, geometry_edges, list(geometry_nodes))
            # compute true max distance (physical diameter = max pairwise distance + 2*particle_radius)
        if PBC == False:
            pos_df_indexed = cluster_pos_df.set_index("Tag#")
            unwrap_pos = pos_df_indexed.loc[list(geometry_nodes)][["x","y","z"]].values.astype(float)

        radii = cluster_pos_df.set_index("Tag#").loc[geometry_nodes]['Radius'].values
        physical_diameter = cluster_max_distance(unwrap_pos, radii)

        # convex hull volume (optional)
        hull_vol = cluster_convex_hull_volume(unwrap_pos)

        # particle count
        n_particles = len(geometry_nodes)

        # cluster center (in unwrapped coordinates)
        cluster_center = unwrap_pos.mean(axis=0)

        if PBC == True:
          # wrap back into simulation box for placement of coarse-grained particle
          cluster_center_wrapped = cluster_center % Lbox
          cluster_center = cluster_center_wrapped.copy()

        typeids = cluster_pos_df.set_index("Tag#").loc[geometry_nodes]['TypeID'].values
        n_type1 = np.sum(typeids == 1)
        n_type2 = np.sum(typeids == 2)

    else:
        physical_diameter = 0.0
        hull_vol = 0.0
        n_particles = 0
        cluster_center = np.array([0,0,0])
        n_type1 = 0
        n_type2 = 0

    topological_diameter = nx.diameter(cluster_subgraph)

    # get interior average degree of each cluster
    nnodes_cluster = cluster_subgraph.number_of_nodes()
    nedges_cluster = nx.number_of_edges(cluster_subgraph)
    inner_avg_degree = 2 * nedges_cluster / nnodes_cluster

    # get interior angle information for each cluster
    # Prepare a list to store angle data
    angle_data = []

    # Retrieve node positions and edge types
    pos = nx.get_node_attributes(cluster_subgraph, 'pos')
    edge_types = nx.get_edge_attributes(cluster_subgraph, 'edge_type')

    # Angle ID counter
    angle_id_counter = 0

    # Calculate the angle between edges at each node
    angle_dict = {}
    for node in cluster_subgraph.nodes():
        neighbors = list(cluster_subgraph.neighbors(node))
        if len(neighbors) < 2:
            continue  # No angle if less than two edges meet at the node

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                # Get position vectors for two neighbors
                if PBC == True:
                  vec1 = pbc_vec(np.array(pos[node]), np.array(pos[neighbors[i]]), Lbox) # PBC
                  vec2 = pbc_vec(np.array(pos[node]), np.array(pos[neighbors[j]]), Lbox) # PBC
                else:
                  vec1 = np.array(pos[neighbors[i]]) - np.array(pos[node])
                  vec2 = np.array(pos[neighbors[j]]) - np.array(pos[node])

                # Calculate the angle between the two vectors
                angle = angle_between_vectors(vec1, vec2)

                # Store the angle as a node attribute (angles between edges meeting at this node)
                cluster_subgraph.nodes[node][f'angle_{neighbors[i]}_{neighbors[j]}'] = angle

                # Store the angle between the two neighbors at the node in a dictionary
                angle_dict[(node, neighbors[i], neighbors[j])] = angle

                # Retrieve edge types for the two edges
                edge_1 = (node, neighbors[i]) if (node, neighbors[i]) in edge_types else (neighbors[i], node)
                edge_2 = (node, neighbors[j]) if (node, neighbors[j]) in edge_types else (neighbors[j], node)

                type_1 = edge_types[edge_1]
                type_2 = edge_types[edge_2]

                # Increment angle_id
                angle_id_counter += 1

                # Store the information in a list of tuples
                angle_data.append((angle_id_counter, edge_1, type_1, edge_2, type_2, angle))

    # Create a DataFrame from the angle data
    angle_df = pd.DataFrame(angle_data, columns=["angle_id", "edge_1", "type_1", "edge_2", "type_2", "angle_degree"])
    cluster_angle_dfs.append(angle_df)


    cluster_diameters[int(cluster_id)] = int(topological_diameter)
    physical_cluster_diameters[int(cluster_id)] = float(physical_diameter)
    cluster_hull_volumes[int(cluster_id)] = float(hull_vol)
    cluster_particle_counts[int(cluster_id)] = int(n_particles)
    cluster_inner_degree[int(cluster_id)] = float(inner_avg_degree)
    cluster_center_x[int(cluster_id)] = cluster_center[0]
    cluster_center_y[int(cluster_id)] = cluster_center[1]
    cluster_center_z[int(cluster_id)] = cluster_center[2]

    cluster_type1_counts[int(cluster_id)] = int(n_type1)
    cluster_type2_counts[int(cluster_id)] = int(n_type2)
    cluster_bonds_SS[int(cluster_id)] = int(n_SS)
    cluster_bonds_SL[int(cluster_id)] = int(n_SL)
    cluster_bonds_LL[int(cluster_id)] = int(n_LL)

# DataFrame
cluster_diameters_df = pd.DataFrame.from_dict(physical_cluster_diameters, orient='index', columns=['Physical Diameter'])
cluster_diameters_df['Cluster'] = cluster_diameters_df.index
cluster_diameters_df['Topological Diameter'] = pd.Series(cluster_diameters)
cluster_diameters_df['ConvexHullVolume'] = pd.Series(cluster_hull_volumes)
cluster_diameters_df['N_particles'] = pd.Series(cluster_particle_counts)
cluster_diameters_df['N_type1'] = pd.Series(cluster_type1_counts)
cluster_diameters_df['N_type2'] = pd.Series(cluster_type2_counts)
cluster_diameters_df['N_SS_bonds'] = pd.Series(cluster_bonds_SS)
cluster_diameters_df['N_SL_bonds'] = pd.Series(cluster_bonds_SL)
cluster_diameters_df['N_LL_bonds'] = pd.Series(cluster_bonds_LL)
cluster_diameters_df['Inner_Avg_Degree'] = pd.Series(cluster_inner_degree)
cluster_diameters_df['CenterX'] = pd.Series(cluster_center_x)
cluster_diameters_df['CenterY'] = pd.Series(cluster_center_y)
cluster_diameters_df['CenterZ'] = pd.Series(cluster_center_z)
cluster_diameters_df.reset_index(drop=True, inplace=True)
cluster_diameters_df.to_csv(f'{data_outpath}/cluster_diameters_{clustering_style}.csv', index=False)




#############################
""" BUILD CLUSTER NETWORK """
#############################
## STEP 5: Create a new graph G_clusters with weighted edges

# set df to be indexed by Tag# for easier searching
lcc_clustered_df = lcc_clustered_df.set_index("Tag#")

# create a new graph of the clusters!
G_clusters = nx.Graph()

# Add nodes for each cluster scaled by cluster size (number of particles)
for cluster_id in np.unique(lcc_clustered_df['Cluster']):
    cluster_size = np.sum(lcc_clustered_df['Cluster'] == cluster_id)
    c_df = cluster_diameters_df.loc[cluster_diameters_df['Cluster'] == cluster_id]
    cluster_diameter = c_df['Topological Diameter'].values[0]
    physical_cluster_diameter = c_df['Physical Diameter'].values[0]
    pos = (c_df['CenterX'].values[0], c_df['CenterY'].values[0], c_df['CenterZ'].values[0])
    G_clusters.add_node(cluster_id, size=cluster_size, pos=pos, diameter=cluster_diameter, physical_diameter=physical_cluster_diameter)

# Add edges between clusters (this is the number of particle-particle contacts that connect each cluster)
cluster_bridge_edges = []
for index, row in edge_df_lcc.iterrows():
    source = int(row['source'])
    target = int(row['target'])

    source_cluster = lcc_clustered_df.loc[source, 'Cluster']
    target_cluster = lcc_clustered_df.loc[target, 'Cluster']

    if source_cluster != target_cluster:

        # --- aggregation logic ---
        if G_clusters.has_edge(source_cluster, target_cluster):
            G_clusters[source_cluster][target_cluster]['weight'] += 1
        else:
            G_clusters.add_edge(source_cluster, target_cluster, weight=1)

        # --- record particle-level ID ---
        type_source = lcc_clustered_df.loc[source, 'TypeID']
        type_target = lcc_clustered_df.loc[target, 'TypeID']

        # optional: canonical ordering (so S-L == L-S consistently)
        #pair_type = tuple(sorted((type_source, type_target)))

        def label_pair(t1, t2):
            if t1 == 1 and t2 == 1:
                return 'S-S'
            elif t1 == 2 and t2 == 2:
                return 'L-L'
            else:
                return 'S-L'
        pair_label = label_pair(type_source, type_target)
       

        cluster_bridge_edges.append({
            'source_cluster': int(source_cluster),
            'target_cluster': int(target_cluster),
            'source_particle': source,
            'target_particle': target,
            'type_source': int(type_source),
            'type_target': int(type_target),
            #'pair_type': pair_type
            'bond_type': pair_label
        })

# save bridge info
cluster_bridge_df = pd.DataFrame(cluster_bridge_edges)
cluster_bridge_df.to_csv(f'{data_outpath}/cluster_bridge_edges_{clustering_style}.csv', index=False)

# Create a list to store the edges with their weights
edges_with_weights = []

# Iterate over the edges in G_clusters to extract source, target, and weight
for (source_cluster, target_cluster, weight) in G_clusters.edges(data='weight'):
    edges_with_weights.append({
        'source_cluster': int(source_cluster),
        'target_cluster': target_cluster,
        'weight': weight
    })

# Convert the list to a DataFrame
weighted_cluster_edges_df = pd.DataFrame(edges_with_weights)

if PBC == True:
    # function to add edge lengths to the graph:
    def add_edge_lengths(G, box=None):
        """
        Compute physical edge lengths for all edges in a graph G, using node 'pos'.
        Optionally apply periodic boundaries.

        Parameters
        ----------
        G : networkx.Graph
            Graph where each node has attribute 'pos' = (x, y, z)
        box : array-like or None
            If provided, should be shape (3,) = [Lx, Ly, Lz].
            Uses minimum-image convention for periodic boundary conditions.

        Returns
        -------
        None (modifies G in place, adds edge attribute 'length')
        """

        use_pbc = box is not None
        if use_pbc:
            box = np.asarray(box, dtype=float)

        for u, v, data in G.edges(data=True):
            # positions from node attributes
            r1 = np.asarray(G.nodes[u]['pos'], dtype=float)
            r2 = np.asarray(G.nodes[v]['pos'], dtype=float)

            delta = r2 - r1

            if use_pbc:
                # minimum-image
                delta -= box * np.round(delta / box)

            length = np.linalg.norm(delta)

            # write edge attribute
            data['length'] = length

    add_edge_lengths(G_clusters, box=Lbox) # with PBC
    # Extract all edge lengths from the Graph
    rows = []
    for u, v, data in G_clusters.edges(data=True):
        rows.append({
            "source_cluster": u,
            "target_cluster": v,
            "edge_length": data.get("length", np.nan)
        })

if PBC == False:
    # function to add edge lengths to the graph:
    def add_edge_lengths(G, box=None):
        """
        Compute physical edge lengths for all edges in a graph G, using node 'pos'.
        Optionally apply periodic boundaries.

        Parameters
        ----------
        G : networkx.Graph
            Graph where each node has attribute 'pos' = (x, y, z)
        box : array-like or None
            If provided, should be shape (3,) = [Lx, Ly, Lz].
            Uses minimum-image convention for periodic boundary conditions.

        Returns
        -------
        None (modifies G in place, adds edge attribute 'length')
        """

        use_pbc = box is not None
        if use_pbc:
            box = np.asarray(box, dtype=float)

        for u, v, data in G.edges(data=True):
            # positions from node attributes
            r1 = np.asarray(G.nodes[u]['pos'], dtype=float)
            r2 = np.asarray(G.nodes[v]['pos'], dtype=float)

            delta = r2 - r1

            if use_pbc:
                # minimum-image
                delta -= box * np.round(delta / box)

            length = np.linalg.norm(delta)

            # write edge attribute
            data['length'] = length 

    add_edge_lengths(G_clusters) # without PBC

    # Extract all edge lengths from the Graph
    rows = []
    for u, v, data in G_clusters.edges(data=True):
        rows.append({
            "source_cluster": u,
            "target_cluster": v,
            "edge_length": data.get("length", np.nan)
        })

edge_len_df = pd.DataFrame(rows)
# Merge the lengths back into edge_df
weighted_cluster_edges_df = weighted_cluster_edges_df.merge(edge_len_df,
                       on=["source_cluster", "target_cluster"],
                       how="left")

weighted_cluster_edges_df.to_csv(f'{data_outpath}/weighted_cluster_edges_{clustering_style}.csv',index = False)

# SAVE GRAPH AS PKL FILE
particle_network_filename = f"lcc-cluster-network_G_clusters_{clustering_style}.pkl"
with open(data_outpath+'/'+particle_network_filename, 'wb') as fr:
  pickle.dump(G_clusters, fr)
print(f' - Graph saved to "lcc-cluster-network_G_clusters_{clustering_style}.pkl" (PKL file)')


# cluster-angles
# Prepare a list to store angle data
cluster_angle_data = []

cluster_pos = nx.get_node_attributes(G_clusters, 'pos')

# Angle ID counter
cluster_angle_id_counter = 0

# ensure node labels are integers
G_clusters = nx.relabel_nodes(G_clusters, lambda x: int(x))

# Calculate the angle between edges at each node
cluster_angle_dict = {}
for node in G_clusters.nodes():
    neighbors = list(G_clusters.neighbors(node))
    if len(neighbors) < 2:
        continue  # No angle if less than two edges meet at the node

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            # Get position vectors for two neighbors
            if PBC == True:
              vec1 = pbc_vec(np.array(cluster_pos[node]), np.array(cluster_pos[neighbors[i]]), Lbox) # PBC
              vec2 = pbc_vec(np.array(cluster_pos[node]), np.array(cluster_pos[neighbors[j]]), Lbox) # PBC
            else:
              vec1 = np.array(cluster_pos[neighbors[i]]) - np.array(cluster_pos[node])
              vec2 = np.array(cluster_pos[neighbors[j]]) - np.array(cluster_pos[node])

            # Calculate the angle between the two vectors
            angle = angle_between_vectors(vec1, vec2)

            # Store the angle as a node attribute (angles between edges meeting at this node)
            G_clusters.nodes[node][f'angle_{neighbors[i]}_{neighbors[j]}'] = angle

            # Store the angle between the two neighbors at the node in a dictionary
            cluster_angle_dict[(node, neighbors[i], neighbors[j])] = angle

            # Increment angle_id
            cluster_angle_id_counter += 1

            # Retrieve edge types for the two edges
            edge_1 = (node, neighbors[i])
            edge_2 = (node, neighbors[j])

            # Store the information in a list of tuples
            cluster_angle_data.append((cluster_angle_id_counter, edge_1, edge_2, angle))

# Create a DataFrame from the angle data
cluster_angle_df = pd.DataFrame(cluster_angle_data, columns=["angle_id", "edge_1", "edge_2", "angle_degree"])

## bin cluster sizes to determine the cluster length scale xi

# average length scale
diams = cluster_diameters_df['Physical Diameter']
average_xi = sum(diams) / len(diams)

bin_size = 1 # bin physical size by integer distances

# Define the number of bins or specific bin edges
max_diam = int(np.ceil(cluster_diameters_df['Physical Diameter'].max()))
bins = np.linspace(0,max_diam,max_diam+1)

counts, bin_edges = np.histogram(
    cluster_diameters_df['Physical Diameter'],
    bins=bins,
    density=False
)

# Normalize to probability
normalized_counts = counts / counts.sum()

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, normalized_counts, width=(bin_edges[1]-bin_edges[0]))
plt.axvline(x=average_xi,c='black')
plt.text(x=average_xi+0.2,y=0.14,s='$\\langle \\xi \\rangle=$'+str(round(average_xi,2))+' edges', c='black', size=15)

#plt.xlabel('$D$ (cluster diameter, number of edges)',fontsize=15)
plt.xlabel('$D_c$ (cluster diameter, [$D/D_{colloid}$])',fontsize=15)
plt.ylabel('$P(D_c)$',fontsize=15)
plt.title('Cluster Size Distribution',fontsize=15)
plt.savefig(f'{data_outpath}/cluster_sizedist_{clustering_style}.pdf', format='pdf', dpi=300, bbox_inches='tight')

diams = np.array(list(physical_cluster_diameters.values()))
average_xi = sum(diams) / len(diams)
#print(' - xi:',average_xi) # 'R_C (if simulation data)', '$\mu m$ (if experimental data)')

system_volume = Lbox[0] * Lbox[1] * Lbox[2]

# cluster volume fraction
# volume of particles
R_p = 1
particle_vol = 4.0/3.0 * np.pi * R_p**3
total_cluster_particle_volume = sum(cluster_particle_counts.values()) * particle_vol
system_volume = Lbox[0] * Lbox[1] * Lbox[2]
phi_C_particles = total_cluster_particle_volume / system_volume
#print(' - phi_C_particles:',phi_C_particles)

# volume from convex hull
phi_C_hull_sum = sum(cluster_hull_volumes.values()) / system_volume
#print(' - phi_C_hull_sum:',phi_C_hull_sum)

# as spheres
diams = np.array(list(physical_cluster_diameters.values()))
volumes = 4.0/3.0 * np.pi * (0.5 * diams)**3
phi_C_spheres = np.sum(volumes) / system_volume
#print(' - phi_C:',phi_C_spheres)
#phi_C = sum(4*np.pi*(0.5*np.array(diams))**3/3)/system_volume
#print(' - phi_C:',phi_C)

#print("CLUSTER VOLUME FRACTIONS")
#print((np.array(cluster_particle_counts.values()) * particle_vol)/volumes)


# cluster average coordination number
n_clusters = G_clusters.number_of_nodes()
if n_clusters > 0:
    total_edges = G_clusters.number_of_edges()   # if graph is simple, E
    avg_degree_unweighted = 2.0 * total_edges / n_clusters

    # if you have weights and want weighted average degree:
    weighted_degrees = dict(G_clusters.degree(weight='weight'))
    avg_degree_weighted = np.mean(list(weighted_degrees.values()))
else:
    avg_degree_unweighted = 0.0
    avg_degree_weighted = 0.0

#print(' - z_c (unweighted avg degree):', avg_degree_unweighted)
#print(' - z_c (weighted avg degree):', avg_degree_weighted)

# calculate average cluster-bridge bond stiffness
series = cluster_bridge_df['bond_type'].value_counts(normalize=True)
weights = {'S-S': 1, 'S-L': 1.5, 'L-L': 2}
stiffness = sum(weights[bond_type] * frac for bond_type, frac in series.items())

G_prime = stiffness * ((z_c_weighted*phi_C_spheres)/average_xi)

# compile outputs
data={
      'average_xi'             :[average_xi],
      'system_volume'          :[system_volume],
      'phi_C_particles'        :[phi_C_particles],
      'phi_C_hull_sum'         :[phi_C_hull_sum],
      'phi_C_spheres'          :[phi_C_spheres],
      'z_c'                    :[avg_degree_unweighted],
      'z_c_weighted'           :[avg_degree_weighted],
      'kappa'                  :[stiffness]
      'G_prime'                :[G_prime]
      }

cluster_cauchyborn_df = pd.DataFrame(data)

print(f"z_c_unique = {cluster_cauchyborn_df['z_c'].values[0]}")
print(f"z_c_weighted = {cluster_cauchyborn_df['z_c_weighted'].values[0]}")
print(f"phi_C = {cluster_cauchyborn_df['phi_C_spheres'].values[0]}")
print(f"xi = {cluster_cauchyborn_df['average_xi'].values[0]}")
print(f"G_prime = {cluster_cauchyborn_df['G_prime'].values[0]}")

cluster_cauchyborn_df.to_csv(f'{data_outpath}/cluster_cauchy_born_full_{clustering_style}.csv',index = False)
