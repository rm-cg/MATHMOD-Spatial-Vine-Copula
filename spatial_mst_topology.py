import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
import os

print("Executing Day 6: Empirical Kendall's Tau and Kruskal's MST...")

# Define paths safely
base_path = (
    '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/'
    'A Spatial Copula Mathematical Model for Evaluating Regional Economic Dependencies using '
    'PhilSA Nighttime Light Radiance/MATHMOD_Official_Final_Submission_Archive/'
)
csv_folder = os.path.join(base_path, 'Spatial_Matrices_CSV')
input_file = os.path.join(csv_folder, 'Uniform_Pseudo_Obs_Matrix.csv')
output_csv = os.path.join(csv_folder, 'Tree1_MST_Edge_List.csv')

# Load the uniform pseudo-observations matrix
df = pd.read_csv(input_file)

# Isolate only the spatial node columns (exclude Longitude/Latitude/geometry if present)
spatial_nodes = [col for col in df.columns if col not in ['Longitude', 'Latitude', 'geometry']]
data_matrix = df[spatial_nodes].dropna()

# Task 3 & 4: Compute empirical pairwise Kendall's tau using scipy.stats
print("Computing pairwise Kendall's tau correlation matrix...")
tau_matrix = pd.DataFrame(index=spatial_nodes, columns=spatial_nodes)

for i in range(len(spatial_nodes)):
    for j in range(len(spatial_nodes)):
        if i == j:
            tau_matrix.iloc[i, j] = 1.0
        elif i < j:
            tau, _ = stats.kendalltau(data_matrix.iloc[:, i], data_matrix.iloc[:, j])
            tau_matrix.iloc[i, j] = tau
            tau_matrix.iloc[j, i] = tau

# Task 5 & 6: Program Kruskal's Minimum Spanning Tree (MST) using absolute tau
print("Establishing Kruskal MST graph topology...")
G = nx.Graph()

for i in range(len(spatial_nodes)):
    for j in range(i + 1, len(spatial_nodes)):
        node1, node2 = spatial_nodes[i], spatial_nodes[j]
        # We maximize absolute rank correlation, so we use weight=abs(tau)
        weight = np.abs(tau_matrix.loc[node1, node2])
        G.add_edge(node1, node2, weight=weight)

# Compute the Maximum Spanning Tree (to maximize total absolute rank correlation)
mst = nx.maximum_spanning_tree(G, algorithm='kruskal')

# Save the computed edge list array
edges_df = nx.to_pandas_edgelist(mst)
edges_df.to_csv(output_csv, index=False)
print(f"Tree 1 MST topology flawlessly saved to: {output_csv}")
