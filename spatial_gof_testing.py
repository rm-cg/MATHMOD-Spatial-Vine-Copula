# Cramér-von Mises Goodness-of-Fit (GOF) Testing

import pandas as pd
import numpy as np
import pyvinecopulib as pv
import os

print("Executing Day 11: Cramér-von Mises Goodness-of-Fit Testing...")

# Cleaned paths
base_csv_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
u_file = os.path.join(base_csv_path, 'Uniform_Pseudo_Obs_Matrix.csv')
tree1_file = os.path.join(base_csv_path, 'Tree1_Parameters_Optimization.csv')
output_csv = os.path.join(base_csv_path, 'Cramer_von_Mises_GOF_Scores.csv')

# Load matrices
u_df = pd.read_csv(u_file)
tree1_df = pd.read_csv(tree1_file)

N_boot = 500  # Set to 500 to prevent Colab RAM timeout
n = len(u_df)
results = []

def empirical_copula(u_mat):
    """Vectorized calculation of the Empirical Copula"""
    n_samples = len(u_mat)
    Cn = np.zeros(n_samples)
    for i in range(n_samples):
        # Counts how many pairs are strictly less than or equal to the current pair
        Cn[i] = np.sum((u_mat[:, 0] <= u_mat[i, 0]) & (u_mat[:, 1] <= u_mat[i, 1])) / n_samples
    return Cn

print(f"Initiating parametric bootstrap algorithm (N={N_boot} per edge)... this will take a moment.")

for index, row in tree1_df.iterrows():
    node1 = row['Source_Node']
    node2 = row['Target_Node']
    
    u1 = u_df[node1].values
    u2 = u_df[node2].values
    u_data = np.column_stack((u1, u2))
    
    # Re-select the original copula from the data to grab the exact family safely
    controls_all = pv.FitControlsBicop(family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student, 
                                                   pv.BicopFamily.clayton, pv.BicopFamily.gumbel, 
                                                   pv.BicopFamily.frank, pv.BicopFamily.joe])
    bicop_orig = pv.Bicop()
    bicop_orig.select(u_data, controls_all)
    
    # 1. Calculate Sn for the original observed data
    Cn_orig = empirical_copula(u_data)
    Ctheta_orig = bicop_orig.cdf(u_data)
    Sn_orig = np.sum((Cn_orig - Ctheta_orig)**2)
    
    # Force the bootstrap to re-estimate parameters using ONLY the selected family
    controls_boot = pv.FitControlsBicop(family_set=[bicop_orig.family])
    
    # 2. Parametric Bootstrap Loop
    bootstrap_Sn = np.zeros(N_boot)
    for b in range(N_boot):
        # Simulate an artificial dataset
        u_sim = bicop_orig.simulate(n)
        
        # Re-estimate parameters on the artificial data
        bicop_sim = pv.Bicop()
        bicop_sim.select(u_sim, controls_boot)
        
        # Compute bootstrapped Cramér-von Mises statistic
        Cn_sim = empirical_copula(u_sim)
        Ctheta_sim = bicop_sim.cdf(u_sim)
        bootstrap_Sn[b] = np.sum((Cn_sim - Ctheta_sim)**2)
        
    # 3. Calculate exact p-value
    p_val = np.sum(bootstrap_Sn >= Sn_orig) / N_boot
    
    print(f"[{node1} - {node2}] Family: {row['Copula_Family']}, Sn: {Sn_orig:.4f}, p-value: {p_val:.4f}")
    
    results.append({
        'Spatial_Edge': f"{node1} - {node2}",
        'Copula_Family': row['Copula_Family'],
        'Cramer_von_Mises_Statistic': round(Sn_orig, 4),
        'Bootstrapped_P_Value': round(p_val, 4)
    })

# Save the mathematical matrix
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(output_csv, index=False)
print(f"\nGOF testing complete. Matrix successfully saved to: {output_csv}")
