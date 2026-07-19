
# Generates the Python script to compute the empirical Kendall's tau matrix
# and establish Kruskal's Minimum Spanning Tree (MST). This maximizes absolute rank correlation
# to form the mathematical base of your spatial vine copula.

import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
import os

print("Empirical Kendall's Tau and Kruskal's MST...")

# Define paths safely
base_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/A Spatial Copula Mathematical \
Model for Evaluating Regional Economic Dependencies using PhilSA Nighttime Light Radiance/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
input_file = os.path.join(base_path, 'Uniform_Pseudo_Obs_Matrix.csv')
output_csv = os.path.join(base_path, 'Tree1_MST_Edge_List.csv')

# Task 1 & 2: Load the uniform pseudo-observations matrix
df = pd.read_csv(input_file)
# Isolate spatial nodes by dropping metadata or covariates
spatial_nodes = [col for col in df.columns if col not in ['Longitude', 'Latitude', 'geometry', 'LST', 'NDVI', 'NDWI']]
data_matrix = df[spatial_nodes].dropna()

# Task 3 & 4: Compute pairwise Kendall's tau correlation matrix
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

# Task 5 & 6: Establish Kruskal MST graph topology (maximizing absolute rank correlation)
print("Establishing Kruskal MST graph topology...")
G = nx.Graph()
for i in range(len(spatial_nodes)):
    for j in range(i + 1, len(spatial_nodes)):
        node1, node2 = spatial_nodes[i], spatial_nodes[j]
        # We maximize absolute rank correlation, so weight = abs(tau)
        weight = np.abs(tau_matrix.loc[node1, node2])
        G.add_edge(node1, node2, weight=weight)

# Compute the Maximum Spanning Tree
mst = nx.maximum_spanning_tree(G, algorithm='kruskal')

# Save the computed edge list array
edges_df = nx.to_pandas_edgelist(mst)
edges_df.to_csv(output_csv, index=False)
print(f"Tree 1 MST topology flawlessly saved to: {output_csv}")
