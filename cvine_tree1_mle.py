import pandas as pd
import numpy as np
import pyvinecopulib as pv
import os

print("Bivariate Copula Selection Pool and Tree 1 MLE Initialization...")

# Define paths safely
base_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/A Spatial Copula Mathematical Model for Evaluating \
Regional Economic Dependencies using PhilSA Nighttime Light Radiance/MATHMOD_Official_Final_Submission_Archive/\
Spatial_Matrices_CSV'
data_file = os.path.join(base_path, 'Uniform_Pseudo_Obs_Matrix.csv')
edges_file = os.path.join(base_path, 'Tree1_MST_Edge_List.csv')
output_csv = os.path.join(base_path, 'Tree1_Parameters_Optimization.csv')

# Task 1: Load the data matrices
u_df = pd.read_csv(data_file)
edges_df = pd.read_csv(edges_file)

# Task 4: Define the specific heavy-tail and asymmetric copula family selection pool
copula_pool = [
    pv.BicopFamily.gaussian,
    pv.BicopFamily.student,
    pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.frank,
    pv.BicopFamily.joe
]

# Configure the fitting controls (MLE optimization, strict family pool)
controls = pv.FitControlsBicop(family_set=copula_pool, selection_criterion="aic")

results = []

# Task 5: MLE Optimization loop for Tree 1 edges
print("Evaluating pair-copulas via MLE and Akaike Information Criterion (AIC)...")
for index, row in edges_df.iterrows():
    node1 = row['source']
    node2 = row['target']
    
    # Extract the uniform pseudo-observations for the two connected spatial nodes
    u_data = u_df[[node1, node2]].dropna().values
    
    # Fit the bivariate copula
    bicop = pv.Bicop(u_data, controls)
    
    # Task 6: Extract fitted mathematical parameters and log-likelihood
    results.append({
        'Tree_Layer': 1,
        'Source_Node': node1,
        'Target_Node': node2,
        'Copula_Family': bicop.family.name,
        'Rotation': bicop.rotation,
        'Parameter_1': bicop.parameters if len(bicop.parameters) > 0 else np.nan,
        'Parameter_2': bicop.parameters[2] if len(bicop.parameters) > 1 else np.nan,
        'Tau': bicop.tau,
        'Log_Likelihood': bicop.loglik(u_data),
        'AIC': bicop.aic(u_data)
    })

# Export the rigorous analytical table
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"Tree 1 MLE parameter optimization successfully secured to: {output_csv}")
