# DAY 10: Theoretical and Empirical Spatial Tail Dependence Estimation (Student-t Final Patch)

import pandas as pd
import numpy as np
import scipy.stats as stats
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
    family = str(row['Copula_Family']).lower()
    
    # THE ULTIMATE FIX: Strip brackets, split the string, and use pop(0) to bypass bracket errors
    par_raw = str(row['Parameter_1']).replace('[', '').replace(']', '').replace(',', ' ').strip()
    par_elements = par_raw.split()
    
    # Safely extract the first number by popping it off the list
    par_str = par_elements.pop(0)
    par = float(par_str)
    
    # Safely extract the second number by popping it off the list (if it exists)
    if len(par_elements) > 0:
        par2_str = par_elements.pop(0)
        par2 = float(par2_str)
    else:
        # Fallback if it was saved normally
        par2_raw = str(row['Parameter_2']).replace('[', '').replace(']', '').strip()
        par2 = float(par2_raw) if par2_raw not in ['nan', 'None', ''] else np.nan
    
    # 1. Theoretical Tail Dependence
    theo_lam_U = 0.0
    theo_lam_L = 0.0
    
    if 'gumbel' in family:
        theo_lam_U = 2.0 - 2.0**(1.0 / par) if par >= 1.0 else 0.0
    elif 'clayton' in family:
        theo_lam_L = 2.0**(-1.0 / par) if par > 0.0 else 0.0
    elif 'student' in family or family == 't':
        rho = par
        nu = par2
        if not np.isnan(nu) and nu > 0 and -1.0 < rho < 1.0:
            t_val = -np.sqrt(nu + 1.0) * np.sqrt((1.0 - rho) / (1.0 + rho))
            tail_dep = 2.0 * stats.t.cdf(t_val, df=nu + 1.0)
            theo_lam_U = tail_dep
            theo_lam_L = tail_dep

    # 2. Empirical Tail Dependence
    u1 = u_df[node1].values
    u2 = u_df[node2].values
    
    emp_lam_U = np.sum((u1 > t_upper) & (u2 > t_upper)) / (n * (1.0 - t_upper))
    emp_lam_L = np.sum((u1 <= t_lower) & (u2 <= t_lower)) / (n * t_lower)
    
    results.append({
        'Spatial_Node_Pair': f"{node1} - {node2}",
        'Copula_Family': row['Copula_Family'],
        'Theoretical_Lambda_U': round(theo_lam_U, 4),
        'Empirical_Lambda_U': round(emp_lam_U, 4),
        'Theoretical_Lambda_L': round(theo_lam_L, 4),
        'Empirical_Lambda_L': round(emp_lam_L, 4)
    })

# Save the mathematical array
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(output_csv, index=False)
print(f"Theoretical and Empirical Tail Dependence estimators successfully secured to: {output_csv}")
print("Data validation complete. Spatial asymmetry quantified.")
