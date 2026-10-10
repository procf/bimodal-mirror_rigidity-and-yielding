## Extract data to CSV and run primary network analysis on 
## the particle trajectories from a colloid simulation
##
"""
## Load data and build a network
##
## How to use:
##     python sim-build-particle-network.py 
##
## - Load data from a GSD gel_file and build a network, G
## - Collect angle distribution for G
##
## OUTPUT:
##    - node_df_all, node_df_lcc 
##        # keys : 'Tag#', 'x', 'y', 'z', 'Radius', 'TypeID'  
##        saved to data_outpath+'/node_df_all.csv' 
##        saved to data_outpath+'/node_df_lcc.csv' 
##    - edge_df_all, edge_df_lcc
##        # keys : 'source', 'target', 'edge_type'  
##        saved to data_outpath+'/edgelist_all.csv'
##        saved to data_outpath+'/edgelist_lcc.csv'
##    - G, g
##        # full network: 'pos' [x,y,z], 'radius', 'type_id', 'edge_type' 
##        saved to data_outpath+'/particle-network_G.pkl'
##        saved to data_outpath+'/lcc-particle-network_g.pkl'
##    - net_df, lcc_df
##        saved to data_outpath+'/full_net.csv'
##        # keys : 'cut_off', 'n_components', 'lcc_size', 'ncolloids', 'avg_degree', 
##        saved to data_outpath+'/network.csv'
##        saved to data_outpath+'/lcc.csv'
##    - angle_df, lcc_angle_df
##        # keys : 'angle_id', 'edge_1', 'type_1', 'edge_2', 'type_2', 'angle_degree' 
##        saved to data_outpath+'/angle_dist_all.csv'
##        saved to data_outpath+'/angle_dist_lcc.csv'
"""
## (Rob Campbell)

########################
""" MODULE LIBRARY """
########################
# load data and build a network
import numpy as np
import gsd.hoomd
import pandas as pd
import networkx as nx
import os
import re
import glob
import sys
import pickle
# for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# angle gradient matrix
import scipy.linalg as la

##########################
""" INPUT PARAMETERS """
##########################

#sys args: gel_file bimodal_bool data_file

gel_file = str(sys.argv[1]) 
bimodal_bool = bool(sys.argv[2])

if bimodal_bool == True:
  R_C2 = 2 
  colloid1_typeid = 1
  colloid2_typeid = 2
else:
  colloid_typeid = 1 
kappa_m = 60 
R_C1 = 1
PBC = True
cut_off = 3/kappa_m 
R_C = R_C1

kT = 0.1
V_colloid = (4/3)*np.pi*R_C**3

# filepath to folder where data files will be created
posedge_outpath = str(sys.argv[3]) #'data_'+tag
data_outpath = posedge_outpath+'/GMM' 
if PBC == False:
  data_outpath = data_outpath+'/non-periodic'

# create "data" subfolder if it doesn't exit
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)


###############################
""" BUILD NETWORK FROM DATA """
###############################


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

## posCSV
"""
# create a CSV of particle (i.e. node) position information for each frame
# (tag, x, y, z, typeID, radius)
"""
def posCSV_calc(filename):
  # open the simulation GSD file as "traj" (trajectory)
  traj = gsd.hoomd.open(filename, 'r')

  # set path and filename
  dir_path = posedge_outpath+'/frame-pos'
  pos_output = dir_path+'/positions_frame' # + <#>.csv in python loop

  # check for existing CSV data 
  if os.path.exists(dir_path) == False:
    os.mkdir(dir_path)
  if os.path.exists(dir_path) == True:
    # NOTE: this counts ALL CSV files in this directory
    nframes = 0
    # set the pattern for files ending in <number>.csv
    pattern = re.compile(r'positions_frame\d+\.csv$')
    # Iterate directory
    for filename in os.listdir(dir_path):
      # Check if the file has a CSV extension
      if pattern.match(filename):
        nframes += 1

    if nframes == len(traj):
      print(' - position data CSV files already seem to exist for all frames. Not creating new CSV files.')
      return

  nframes = len(traj)
  solvents = np.where(traj[-1].particles.typeid == 0)[0]
  nsolvents = len(solvents)
  if bimodal_bool == True:
    colloids = np.where((traj[-1].particles.typeid == colloid1_typeid) | (traj[-1].particles.typeid == colloid2_typeid))[0]
  else:
    colloids = np.where(traj[-1].particles.typeid == [colloid_typeid])[0]
  ncolloids = len(colloids)
  typeid = traj[-1].particles.typeid[colloids]
  radii = 0.5*traj[-1].particles.diameter[colloids]

  for f in range(nframes):
    rpos = traj[f].particles.position[colloids]

    # make a data frame to export to CSV
    df_pos = pd.DataFrame()
    df_pos['tag'] = colloids-nsolvents
    df_pos['x'] = rpos[:,0]
    df_pos['y'] = rpos[:,1]
    df_pos['z'] = rpos[:,2] 
    df_pos['typeID'] = typeid
    df_pos['radius'] = radii

    df_pos.to_csv(pos_output+str(f)+'.csv', index=False)
  print(" - Position data saved to CSV for "+str(nframes)+" frames")

## edgelistCSV
  """
  # create a CSV file of all the bonded particle pairs (i.e. edges of the network) for each frame
  # i,j position for each bond/edge
  """

  def edgelistCSV_calc(filename):
    # open the simulation GSD file as "traj" (trajectory)
    traj = gsd.hoomd.open(filename, 'r')

    # set path and filename
    #edge_dir_path = data_outpath+'/frame-edges'
    edge_dir_path = posedge_outpath+'/frame-edges'
    edge_output = edge_dir_path+'/edgelist' # + <#>.csv in python loop

    # check for existing CSV data 
    if os.path.exists(edge_dir_path) == False:
      os.mkdir(edge_dir_path)
    if os.path.exists(edge_dir_path) == True:
      # NOTE: this counts ALL edgelist CSV files in this directory
      nframes = 0
      # set the pattern for files ending in <number>.csv
      pattern = re.compile(r'edgelist\d+\.csv$')
      # Iterate directory
      for file in os.listdir(edge_dir_path):
        # Check if the file has a CSV extension
        if pattern.match(file):
          nframes += 1

      if nframes == len(traj):
        print(' - edgelist data CSV files already seem to exist for all frames. Not creating new CSV files.')
        return

    nframes = len(traj)
    if bimodal_bool == True:
      colloids = np.where((traj[-1].particles.typeid == colloid1_typeid) | (traj[-1].particles.typeid == colloid2_typeid))[0]
    else:
      colloids = np.where(traj[-1].particles.typeid == [colloid_typeid])[0]
    ncolloids = len(colloids)
    radii = 0.5*traj[-1].particles.diameter[colloids]
    rcut = cut_off
    lbox = traj[-1].configuration.box[:3]

    # create an array of xyz positon of all colloids in all frames    
    allpos = np.zeros((nframes,ncolloids,3))
    for i in range(0,nframes):
      allpos[i,:,:] = traj[i].particles.position[colloids]

    for f in range(nframes):
      edge_file = f"{edge_output}{f}.csv"
      pos = allpos[f,:,:]

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

      # save the edgelist (header-only file if there are no bonds)
      np.savetxt(edge_file, pairs.reshape(-1, 2), fmt='%d', delimiter=',',
        header='i,j', comments='')

    print(f"Edgelist calculation complete for {nframes} frames")


# create the edgelist
posCSV_calc(gel_file)	
  
# extract additional data from the GSD file 
traj = gsd.hoomd.open(gel_file, 'r')
nframes = len(traj)
solvents = np.where(traj[-1].particles.typeid == 0)[0]
nsolvents = len(solvents)
if bimodal_bool == True:
  colloids = np.where((traj[-1].particles.typeid == colloid1_typeid) | (traj[-1].particles.typeid == colloid2_typeid))[0]
else:
  colloids = np.where(traj[-1].particles.typeid == [colloid_typeid])[0]
ncolloids = len(colloids)
pos = traj[-1].particles.position[colloids]
radius = 0.5*traj[-1].particles.diameter[colloids]
typeid = traj[-1].particles.typeid[colloids]
Lbox = traj[0].configuration.box[:3]

system_volume = Lbox[0] * Lbox[1] * Lbox[2]
data = {
  "Tag#": colloids-nsolvents,
  "x": pos[:,0],
  "y": pos[:,1],
  "z": pos[:,2],
  "Radius": radius,
  "TypeID": typeid,
}
df = pd.DataFrame(data)

# create empty dataframe for all network data  
network_frames_df = pd.DataFrame()
  
if PBC == True:
  edgelistCSV_calc(gel_file)
  # import all data into one dataframe
  edge_output = posedge_outpath+'/frame-edges/edgelist' # + <#>.csv in f90
  frame_dfs = []
  for frame in range(nframes):
    # loop through all frames
    filepath = edge_output+str(frame)+'.csv'
    # import CSV data
    edge_df = pd.read_csv(filepath)
    # rename colums as needed
    edge_df = edge_df.rename(columns={"i": "source", "j": "target"})
    edge_df.insert(loc=0, column='frame', value=frame)
    frame_dfs.append(edge_df)

  alledge_df = pd.concat(frame_dfs, ignore_index=True) 

  # only look at the last fram 
  edge_df = alledge_df[alledge_df['frame'] == (nframes-1)][["source", "target"]]

elif PBC == False:
  #....
  #######
  ## STEP 2: create edgelist for the network (no periodic boundaries)
  """
  # create a CSV file of all the bonded particle pairs (i.e. edges of the network) for each frame
  # i,j position for each bond/edge
  """
  ## create an edgelist    
  edgelist_i = []
  edgelist_j = []

  # create list of the contact distance (r_i + r_j) for all possible i-j pairs
  radii = df['Radius'].values
  radii_sum = radii[:, np.newaxis] + radii

  # get all possible interaction distances r_ij
  x = df['x'].values
  y = df['y'].values
  z = df['z'].values

  dx = x[:, np.newaxis] - x
  dy = y[:, np.newaxis] - y
  dz = z[:, np.newaxis] - z
    
  rij = np.sqrt(dx*dx + dy*dy + dz*dz)

  # calculate surface-surface distance
  hij = rij - radii_sum

  # filter valid edges only
  valid_edges = (hij <= cut_off) & (rij > 0)

  # Find indices of valid edges
  edge_indices = np.transpose(np.where(valid_edges))

  edgelist_i = edge_indices[:, 0] #+ 1 # +1 indexes from 1, not 0
  edgelist_j = edge_indices[:, 1] #+ 1 # +1 indexes from 1, not 0
        
  # covert to df   
  edge_df = pd.DataFrame({'target': edgelist_i,
                          'source': edgelist_j})
    
  edge_df[['target', 'source']] = edge_df.apply(sorted, axis=1, result_type='broadcast')

  # Remove duplicates
  edge_df.drop_duplicates(inplace=True)

  # track the type of particle-particle interaction in each edge
  edge_id = []
  for idx, edge_data in edge_df.iterrows():
    source_tag = edge_data['source']
    source_type_id = df.loc[df['Tag#'] == source_tag, 'TypeID'].values[0]

    target_tag = edge_data['target']
    target_type_id = df.loc[df['Tag#'] == target_tag, 'TypeID'].values[0]

    # Create the edge_type by combining source and target type_ids
    edge_type = f"{source_type_id}-{target_type_id}"
    edge_id.append(edge_type)

  # Add the new 'edge_type' column to edge_df
  edge_df['edge_type'] = edge_id
####

#######
## CREATE A NEW GRAPH
G = nx.Graph()

# track the type of particle-particle interaction in each edge
edge_id = []
for idx, edge_data in edge_df.iterrows():
    source_tag = edge_data['source']
    source_type_id = df.loc[df['Tag#'] == source_tag, 'TypeID'].values[0]

    target_tag = edge_data['target']
    target_type_id = df.loc[df['Tag#'] == target_tag, 'TypeID'].values[0]

    # Create the edge_type by combining source and target type_ids
    edge_type = f"{source_type_id}-{target_type_id}"
    edge_id.append(edge_type)

# Add the new 'edge_type' column to edge_df
edge_df['edge_type'] = edge_id

# Add nodes to the graph from edge_df (Tag# as the node ID, and positions as attributes)
for _, row in df.iterrows():
    G.add_node(row['Tag#'], pos=(row['x'], row['y'], row['z']), radius=row['Radius'], type_id=row['TypeID'])

# Add edges to the graph from edge_df (target and source define the edge connections, edge_type as attribute)
for _, row in edge_df.iterrows():
    G.add_edge(row['target'], row['source'], edge_type=row['edge_type'])

if PBC == True:
  add_edge_lengths(G, box=Lbox) # with PBC
elif PBC == False:
  add_edge_lengths(G) # withot PBC
# Extract all edge lengths from the Graph
rows = []
for u, v, data in G.edges(data=True):
    rows.append({
        "source": u,
        "target": v,
        "edge_length": data.get("length", np.nan)
    })
edge_len_df = pd.DataFrame(rows)
# Merge the lengths back into edge_df
# Note: Your edge_df uses ['target','source'], so match on both
edge_df = edge_df.merge(edge_len_df,
                       on=["source", "target"],
                       how="left")


# Access the node positions
pos = nx.get_node_attributes(G, 'pos')

print(' - Full network:')

# save as csv files
df.to_csv(data_outpath+'/node_df_all.csv',index=False) 
print('   - GSD data saved as CSV: "node_df_all.csv"')

edgelist_file = data_outpath+'/edgelist_all.csv'
edge_df.to_csv(edgelist_file,index=False) 
print('   - Edgelist calculated, saved to "edgelist_all.csv"')

# SAVE GRAPH AS PKL FILE
particle_network_filename = "particle-network_G.pkl"
with open(data_outpath+'/'+particle_network_filename, 'wb') as fr:
  pickle.dump(G, fr)
print('   - Graph saved to "particle-network_G.pkl" (PKL file)')

phi = (ncolloids*V_colloid) / system_volume

# calculate the total number of edges      
nedges = nx.number_of_edges(G)

# calculate the average degree of the network, i.e. avg contact number in gels
avg_degree = 2 * nedges / ncolloids

# number of connected components
n_cc = nx.number_connected_components(G)

# return the indices of the nodes of the largest connected components
lcc_nodes = max(nx.connected_components(G), key=len)

# calculate the size of the largest connected component
lcc_size = len(lcc_nodes)

# compile outputs
data={
      'cut_off'                :[cut_off],
      'n_components'           :[n_cc], 
      'lcc_size'               :[lcc_size],
      'ncolloids'              :[ncolloids],
      'avg_degree'             :[avg_degree],
      'phi'                    :[phi],
      'system_volume'          :[system_volume],
      'L_X'                    :[Lbox[0]],
      'L_Y'                    :[Lbox[1]],
      'L_Z'                    :[Lbox[2]],
      }

net_df = pd.DataFrame(data)


###################################
""" FUNCTION TO UNWRAP POSITIONS"""
###################################

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

###############################
""" GET LCC DATA SEPARATELY """
###############################

g = G.subgraph(lcc_nodes).copy() # Create a subgraph containing only the lcc

# filter data for lcc only (used in plotting)
node_df_lcc = df[df['Tag#'].isin(lcc_nodes)].copy()

edge_df_lcc = edge_df[(edge_df['target'].isin(lcc_nodes)) & (edge_df['source'].isin(lcc_nodes))].copy()
ncolloids_lcc = len(node_df_lcc)
print(' - LCC contains',ncolloids_lcc,'colloids ('+str(round(ncolloids_lcc/ncolloids,4)*100)+'% of all colloids)')

# Prepare a list to store angle data
lcc_angle_data = []

# save lcc data for easier access later
lcc_node_file = data_outpath+'/node_df_lcc.csv'
node_df_lcc.to_csv(lcc_node_file,index=False)
lcc_edgelist_file = data_outpath+'/edgelist_lcc.csv'
edge_df_lcc.to_csv(lcc_edgelist_file,index=False)
print('   - LCC data saved to "node_df_lcc.csv", "edgelist_lcc.csv"')

# SAVE GRAPH AS PKL FILE
lcc_particle_network_filename = "lcc-particle-network_g.pkl"
with open(data_outpath+'/'+lcc_particle_network_filename, 'wb') as fr:
  pickle.dump(g, fr)
print('   - LCC graph saved to "lcc-particle-network_g.pkl" (PKL file)')


phi_lcc = (ncolloids_lcc*V_colloid) / system_volume


# calculate the total number of edges      
nedges_lcc = nx.number_of_edges(g)

# calculate the average degree of the network, i.e. avg contact number in gels
avg_degree_lcc = 2 * nedges_lcc / ncolloids_lcc

# compile outputs
data={
      'cut_off'                :[cut_off],
      'ncolloids'              :[ncolloids_lcc],
      'avg_degree'             :[avg_degree_lcc],
      'phi'                    :[phi_lcc],
      'system_volume'          :[system_volume],
      'L_X'                    :[Lbox[0]],
      'L_Y'                    :[Lbox[1]],
      'L_Z'                    :[Lbox[2]],
      }

lcc_df = pd.DataFrame(data)


##########################
""" COLLECT ANGLE DATA """
##########################

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


## FULL NETWORK

# Prepare a list to store angle data
angle_data = []

# Angle ID counter
angle_id_counter = 0

# ensure node labels are integers
G = nx.relabel_nodes(G, lambda x: int(x))

# Retrieve node positions and edge types
pos = nx.get_node_attributes(G, 'pos')
edge_types = nx.get_edge_attributes(G, 'edge_type')
node_types = nx.get_node_attributes(G, 'type_id')

def directed_type(u, v, node_types):
   return f"{int(node_types[u])}-{int(node_types[v])}"

# Calculate the angle between edges at each node
angle_dict = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
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
            G.nodes[node][f'angle_{neighbors[i]}_{neighbors[j]}'] = angle
            
            # Store the angle between the two neighbors at the node in a dictionary
            angle_dict[(node, neighbors[i], neighbors[j])] = angle
            
            # Retrieve edge types for the two edges
            edge_1 = (node, neighbors[i]) if (node, neighbors[i]) in edge_types else (neighbors[i], node)
            edge_2 = (node, neighbors[j]) if (node, neighbors[j]) in edge_types else (neighbors[j], node)            
            type_1 = directed_type(node, neighbors[i], node_types)
            type_2 = directed_type(node, neighbors[j], node_types)

            # Increment angle_id
            angle_id_counter += 1
            
            # Store the information in a list of tuples
            angle_data.append((angle_id_counter, edge_1, type_1, edge_2, type_2, angle))

# Create a DataFrame from the angle data
angle_df = pd.DataFrame(angle_data, columns=["angle_id", "edge_1", "type_1", "edge_2", "type_2", "angle_degree"])

## LCC angles

# ensure node labels are integers
g = nx.relabel_nodes(g, lambda x: int(x))

# Prepare a list to store angle data
lcc_angle_data = []

# Retrieve node positions and edge types
#lcc_pos = nx.get_node_attributes(g, 'pos')
unwrapped_pos_arr = unwrap_cluster_positions(node_df_lcc[['Tag#','x','y','z']], Lbox, edges=list(g.edges()), global_nodes=list(g.nodes()))
nodes = list(g.nodes())
# Convert array → dict mapping node → (x,y,z)
unwrapped_pos_lcc = {
    nodes[i]: unwrapped_pos_arr[i]
    for i in range(len(nodes))
}
for n in g.nodes():
    g.nodes[n]['pos'] = unwrapped_pos_lcc[n]
lcc_pos = nx.get_node_attributes(g, 'pos')
unwrapped_df = pd.DataFrame(
    unwrapped_pos_arr,
    index=nodes,
    columns=['x_unwrapped', 'y_unwrapped', 'z_unwrapped']
)
unwrapped_df.index.name = 'Tag#'
node_df_lcc = (
    node_df_lcc
    .set_index('Tag#')
    .join(unwrapped_df, how='left')
    .reset_index()
)

lcc_edge_types = nx.get_edge_attributes(g, 'edge_type')
lcc_node_types = nx.get_node_attributes(g, 'type_id')

# Angle ID counter
lcc_angle_id_counter = 0

# Calculate the angle between edges at each node
lcc_angle_dict = {}
for node in g.nodes():
    neighbors = list(g.neighbors(node))
    if len(neighbors) < 2:
        continue  # No angle if less than two edges meet at the node

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            # Get position vectors for two neighbors
            if PBC == True:
              vec1 = pbc_vec(np.array(lcc_pos[node]), np.array(lcc_pos[neighbors[i]]), Lbox) # PBC
              vec2 = pbc_vec(np.array(lcc_pos[node]), np.array(lcc_pos[neighbors[j]]), Lbox) # PBC
            else:
              vec1 = np.array(lcc_pos[neighbors[i]]) - np.array(lcc_pos[node])
              vec2 = np.array(lcc_pos[neighbors[j]]) - np.array(lcc_pos[node])

            # Calculate the angle between the two vectors
            angle = angle_between_vectors(vec1, vec2)

            # Store the angle as a node attribute (angles between edges meeting at this node)
            g.nodes[node][f'angle_{neighbors[i]}_{neighbors[j]}'] = angle

            # Store the angle between the two neighbors at the node in a dictionary
            lcc_angle_dict[(node, neighbors[i], neighbors[j])] = angle

            # Retrieve edge types for the two edges
            edge_1 = (node, neighbors[i]) if (node, neighbors[i]) in lcc_edge_types else (neighbors[i], node)
            edge_2 = (node, neighbors[j]) if (node, neighbors[j]) in lcc_edge_types else (neighbors[j], node)
            type_1 = directed_type(node, neighbors[i], lcc_node_types)
            type_2 = directed_type(node, neighbors[j], lcc_node_types)

            # Increment angle_id
            lcc_angle_id_counter += 1

            # Store the information in a list of tuples
            lcc_angle_data.append((lcc_angle_id_counter, edge_1, type_1, edge_2, type_2, angle))

# Create a DataFrame from the angle data
lcc_angle_df = pd.DataFrame(lcc_angle_data, columns=["angle_id", "edge_1", "type_1", "edge_2", "type_2", "angle_degree"])


print(' - Save data:')

net_df.to_csv(data_outpath+'/net.csv',index = False)
print('   - Basic network information calculated, saved to "net.csv"')

lcc_df.to_csv(data_outpath+'/lcc.csv',index = False)
print('   - Basic LCC network information calculated, saved to "lcc.csv"')


print(' - Angle analysis:')

angle_df.to_csv(data_outpath+'/angle_dist_all.csv',index = False)
lcc_angle_df.to_csv(data_outpath+'/angle_dist_lcc.csv',index = False)

tags = ['Full network:', 'LCC:']
angle_dfs = [angle_df, lcc_angle_df]
messages = [
    '      Full angle distribution saved to "angle_dist.csv"',
    '      LCC angle distribution saved to "lcc_angle_dist.csv"',
    ]

for a in range(len(angle_dfs)):

    plt.figure()

    tag = tags[a]
    df = angle_dfs[a]
    message = messages[a]
    print('    '+tag)
    ## OPTIONAL: plot angle distribution
    print(list(pd.unique(df['type_1'])))
    print(list(pd.unique(df['type_1'])))
    bond_types = list(pd.unique(df['type_1']))
    modality = ''
    if len(bond_types) > 1:
        if len(bond_types) != 4:
            print('      ERROR: not monomodal or bimodal (not 1 or 3 bond types)')
            print('      Bond types =',bond_types)
            modality = 'unknown'
        else:
            print('      Bimodal system identified')
            print('      Bond types =',bond_types)
            modality = 'bimodal'
    else:
        print('      Monomodal system identified')
        print('      Bond types =',bond_types)
        modality = 'monomodal'

    plot_message = ''

    if modality == 'monomodal':

        nbins = 100
        # Plot the distribution of angles
        plt.hist(df['angle_degree'], bins=nbins, color='blue', alpha=0.7, density=True)

        # Convert the y-axis values to percentages
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        plt.title('Distribution of Angles Between Edges')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Probability (%)')
        plt.ylim(top=0.03)
        plt.axvline(x=60, color='black', ls='--', lw=0.75)
        #plt.axvline(x=88.85,color='black',ls='--',lw=0.75)
        if 'LCC' in tag:
            #plt.savefig(data_outpath+'/lcc_monodisp_angledist.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.savefig(data_outpath+'/lcc_monodisp_angledist.png', format='png', dpi=300, bbox_inches='tight')
            plot_message = ' and "lcc_monodisp_angledist.png"'
        else:
            #plt.savefig(data_outpath+'/monodisp_angledist.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.savefig(data_outpath+'/monodisp_angledist.png', format='png', dpi=300, bbox_inches='tight')
            plot_message = ' and "monodisp_angledist.png"'

        

    if modality == 'bimodal':

        # '1-1' + '1-1'
        ss_ss_df = df[(df['type_1'] == '1-1') & (df['type_2'] == '1-1')]
    
        # '1-1' + '1-2' OR '2-1' + '1-1'
        ss_sL_df = df[((df['type_1'] == '1-1') & (df['type_2'] == '1-2')) | ((df['type_1'] == '2-1') & (df['type_2'] == '1-1'))] 

        # '2-1' + '1-2'
        Ls_sL_df = df[(df['type_1'] == '1-2') & (df['type_2'] == '1-2')]

        # '1-1' + '2-2' is not possible

        # '1-2' + '2-2' OR '2-2' + '2-1'
        sL_LL_df = df[((df['type_1'] == '1-2') & (df['type_2'] == '2-2')) | ((df['type_1'] == '2-2') & (df['type_2'] == '2-1'))]
    
        # '1-2' + '2-1'
        sL_Ls_df = df[(df['type_1'] == '1-2') & (df['type_2'] == '2-1')]

        # '2-2' + '2-2'
        LL_LL_df = df[(df['type_1'] == '2-2') & (df['type_2'] == '2-2')]
    
            # Plot the distribution of angles
        nbins = 30

        if len(ss_ss_df) > 0:
            plt.hist(ss_ss_df['angle_degree'], bins=nbins, color='teal', alpha=0.3, label="ss-ss", density=True)
    
        if len(ss_sL_df) > 0:
            plt.hist(ss_sL_df['angle_degree'], bins=nbins, color='darkturquoise', alpha=0.3, label="ss-sL", density=True)
    
        if len(Ls_sL_df) > 0:
            plt.hist(Ls_sL_df['angle_degree'], bins=nbins, color='thistle', alpha=0.3, label="Ls-sL", density=True)
        
        if len(sL_LL_df) > 0:
            plt.hist(sL_LL_df['angle_degree'], bins=nbins, color='orchid', alpha=0.7, label="sL-LL", density=True)
    
        if len(sL_Ls_df) > 0:
            plt.hist(sL_Ls_df['angle_degree'], bins=nbins, color='mediumpurple', alpha=0.3, label="sL-Ls", density=True)
    
        if len(LL_LL_df) > 0:
            plt.hist(LL_LL_df['angle_degree'], bins=nbins, color='crimson', alpha=0.3, label="LL-LL", density=True)

        # Convert the y-axis values to percentages
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    
        plt.title('Distribution of Angles Between Edges',fontsize=14)
        plt.xlabel('Angle (degrees)',fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.ylim(0,0.03)
        plt.legend(fontsize=14)
        if 'LCC' in tag:
            plt.savefig(data_outpath+'/lcc_bimodal_angledist.png', format='png', dpi=300, bbox_inches='tight')
            plot_message = ' and "lcc_bimodal_angledist.png"'
            plt.close()
        else:
            plt.savefig(data_outpath+'/bimodal_angledist.png', format='png', dpi=300, bbox_inches='tight')
            plot_message = ' and "bimodal_angledist.png"'
            plt.close()

    print(message+plot_message)

