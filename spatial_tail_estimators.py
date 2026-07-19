# DAY 10: Theoretical and Empirical Spatial Tail Dependence Estimation

import pandas as pd
import numpy as np
import os

print("Executing Day 10: Spatial Tail Dependence Estimation...")

# Cleaned paths
base_csv_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
u_file = os.path.join(base_csv_path, 'Uniform_Pseudo_Obs_Matrix.csv')
tree1_file = os.path.join(base_csv_path, 'Tree1_Parameters_Optimization.csv')
output_csv = os.path.join(base_csv_path, 'Theoretical_Empirical_Tail_Dependence.csv')

# Load matrices
u_df = pd.read_csv(u_file)
tree1_df = pd.read_csv(tree1_file)

n = len(u_df)
t_upper = 0.95
t_lower = 0.05
results = []

print("Computing theoretical limits and empirical estimators for extreme anomalies...")

for index, row in tree1_df.iterrows():
    node1 = row['Source_Node']
    node2 = row['Target_Node']
    family = row['Copula_Family']
    par = row['Parameter_1']
    
    # 1. Theoretical Tail Dependence
    theo_lam_U = 0.0
    theo_lam_L = 0.0
    
    if family == 'Gumbel':
        # Gumbel has upper tail dependence: 2 - 2^(1/theta)
        theo_lam_U = 2.0 - 2.0**(1.0 / par) if par >= 1 else 0.0
    elif family == 'Clayton':
        # Clayton has lower tail dependence: 2^(-1/theta)
        theo_lam_L = 2.0**(-1.0 / par) if par > 0 else 0.0
        
    # 2. Empirical Tail Dependence
    u1 = u_df[node1].values
    u2 = u_df[node2].values
    
    # Upper-tail empirical estimator: [1 / n(1-t)] * sum(U1 > t & U2 > t)
    emp_lam_U = np.sum((u1 > t_upper) & (u2 > t_upper)) / (n * (1.0 - t_upper))
    
    # Lower-tail empirical estimator: [1 / nt] * sum(U1 <= t & U2 <= t)
    emp_lam_L = np.sum((u1 <= t_lower) & (u2 <= t_lower)) / (n * t_lower)
    
    results.append({
        'Spatial_Node_Pair': f"{node1} - {node2}",
        'Copula_Family': family,
        'Theoretical_Lambda_U': theo_lam_U,
        'Empirical_Lambda_U': emp_lam_U,
        'Theoretical_Lambda_L': theo_lam_L,
        'Empirical_Lambda_L': emp_lam_L
    })

# Save the mathematical array
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(output_csv, index=False)
print(f"Theoretical and Empirical Tail Dependence estimators successfully secured to: {output_csv}")
print("Data validation complete. Spatial asymmetry quantified.")
