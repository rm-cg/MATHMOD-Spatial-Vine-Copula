# DAY 6: Empirical Kendall's Tau Matrix and C-Vine Star Topology

import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
import os

print("Empirical Kendall's Tau and C-Vine Star Topology...")

# YOUR NEW CLEAN PATH
base_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
input_file = os.path.join(base_path, 'Uniform_Pseudo_Obs_Matrix.csv')
output_csv = os.path.join(base_path, 'Tree1_MST_Edge_List.csv')

# Load the uniform pseudo-observations matrix
df = pd.read_csv(input_file)

# Isolate spatial nodes by dropping metadata or covariates
spatial_nodes = [col for col in df.columns if col not in ['Longitude', 'Latitude', 'geometry', 'LST', 'NDVI', 'NDWI']]
data_matrix = df[spatial_nodes].dropna()

print("Computing empirical pairwise Kendall's tau correlation matrix...")
tau_matrix = pd.DataFrame(index=spatial_nodes, columns=spatial_nodes)
for i in range(len(spatial_nodes)):
    for j in range(len(spatial_nodes)):
        if i == j:
            tau_matrix.iloc[i, j] = 1.0
        elif i < j:
            tau, _ = stats.kendalltau(data_matrix.iloc[:, i], data_matrix.iloc[:, j])
            tau_matrix.iloc[i, j] = tau
            tau_matrix.iloc[j, i] = tau

# THE C-VINE FIX: Find the node with the highest sum of absolute dependencies to act as the Central Hub
tau_sums = tau_matrix.abs().sum() - 1.0 # Subtract 1 to ignore self-correlation
root_node = tau_sums.idxmax()
print(f"Mathematical Central Hub Identified: {root_node}")

print("Establishing C-Vine Star Graph Topology...")
G = nx.Graph()

# Connect every single node directly to the central hub
for node in spatial_nodes:
    if node != root_node:
        weight = np.abs(tau_matrix.loc[root_node, node])
        G.add_edge(root_node, node, weight=weight)

# Save the computed edge list array
edges_df = nx.to_pandas_edgelist(G)
edges_df.to_csv(output_csv, index=False)
print(f"Tree 1 C-Vine topology flawlessly saved to: {output_csv}")
