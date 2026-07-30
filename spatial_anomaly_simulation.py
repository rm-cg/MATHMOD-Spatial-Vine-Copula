# Inverse Rosenblatt Transformation and Spatial Anomaly Simulation

import pandas as pd
import numpy as np
import pyvinecopulib as pv
import os

print("Executing Day 12: Inverse Rosenblatt Transformation...")

# Cleaned paths
base_csv_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
u_file = os.path.join(base_csv_path, 'Uniform_Pseudo_Obs_Matrix.csv')
output_csv = os.path.join(base_csv_path, 'Simulated_Anomaly_Propagation.csv')

# Load uniform pseudo-observations
u_df = pd.read_csv(u_file)
spatial_nodes = list(u_df.columns)
d = len(spatial_nodes)
u_data = u_df.values

# Re-initialize and fit the C-Vine to ensure we have the exact structural equations in memory
print("Re-establishing spatial C-Vine architecture...")
copula_pool = [pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student, 
               pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe]
controls = pv.FitControlsVinecop(family_set=copula_pool, selection_criterion="aic")
vine = pv.Vinecop(d)
vine.select(u_data, controls)

N_sim = 5000

# Identify the central hub index dynamically 
hub_node = 'NCR_QuezonCity'
if hub_node in spatial_nodes:
    hub_index = spatial_nodes.index(hub_node)
else:
    hub_index = 0
    hub_node = spatial_nodes

print(f"Simulating N={N_sim} extreme optical spikes (99th percentile) at the central hub ({hub_node})...")

# Generate an independent uniform hypercube
np.random.seed(42)
w_independent = np.random.uniform(0, 1, size=(N_sim, d))

# THE ANOMALY INJECTION: Force the central hub to a massive 99th percentile spike
w_independent[:, hub_index] = 0.99

# Push the independent variables through the Inverse Rosenblatt Transform to generate conditional spatial dependence
u_simulated = vine.inverse_rosenblatt(w_independent)

sim_df = pd.DataFrame(u_simulated, columns=spatial_nodes)
sim_df.to_csv(output_csv, index=False)

print("\n--- ANOMALY PROPAGATION RESULTS ---")
print(f"Condition: {hub_node} forced to 99th percentile (0.99)")
for node in spatial_nodes:
    if node != hub_node:
        # Calculate the probability that the neighbor node is dragged above the 90th percentile
        prob_spike = np.sum(sim_df[node] > 0.90) / N_sim * 100
        print(f"Probability {node} experiences a severe >90th percentile anomaly: {prob_spike:.2f}%")

print(f"\nSimulation matrix successfully saved to: {output_csv}")
