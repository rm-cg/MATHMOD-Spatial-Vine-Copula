# DAY 9: Deep Canonical Vine Construction and Global Joint Likelihood

import pandas as pd
import numpy as np
import pyvinecopulib as pv
import os

print("Executing Day 9: Deep Canonical Vine Construction and Global Likelihood...")

# Your new cleaned paths
base_csv_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
data_file = os.path.join(base_csv_path, 'Uniform_Pseudo_Obs_Matrix.csv')
output_metrics = os.path.join(base_csv_path, 'Global_Vine_Metrics_Summary.csv')
output_decay = os.path.join(base_csv_path, 'Independence_Decay_Check.txt')

u_df = pd.read_csv(data_file)
spatial_nodes = list(u_df.columns)
d = len(spatial_nodes)
u_data = u_df.values

# Heavy-tail copula pool + Independence Copula to test for decay
copula_pool = [
    pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student, 
    pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe
]

controls = pv.FitControlsVinecop(family_set=copula_pool, selection_criterion="aic")

print("Calculating Stepwise MLE and executing recursive h-functions...")

# THE FIX: Initialize the empty vine strictly with dimension 'd', then fit the data using select()
vine = pv.Vinecop(d)
vine.select(u_data, controls)

results = []
independence_count = 0
deep_tree_total = 0

# Extract conditional parameters and check for Independence Decay
for tree_idx in range(d - 1):
    tree_layer = tree_idx + 1
    edges_in_tree = vine.pair_copulas[tree_idx]
    
    for edge_idx, bicop in enumerate(edges_in_tree):
        fam = bicop.family.name
        
        # Monitor deep trees (Tree 3+) for spatial decay to the Independence copula
        if tree_layer >= 3:
            deep_tree_total += 1
            if fam == "Independent":
                independence_count += 1
                
        results.append({
            'Tree_Layer': tree_layer,
            'Edge_Connection': f"Edge_{edge_idx+1}",
            'Copula_Family': fam,
            'Tau': bicop.tau
        })

# Compute theoretical global metrics
global_ll = vine.loglik(u_data)
global_aic = vine.aic(u_data)

# Append global criteria to the bottom of the matrix
results.append({
    'Tree_Layer': 'GLOBAL',
    'Edge_Connection': 'FULL_VINE',
    'Copula_Family': 'JOINT',
    'Tau': np.nan,
    'Log_Likelihood': global_ll,
    'AIC_Score': global_aic
})

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(output_metrics, index=False)
print(f"Global Vine Metrics successfully secured to: {output_metrics}")

# Generate Independence Decay Verification Report
with open(output_decay, 'w') as f:
    f.write(f"Deep Tree Independence Decay Verification\n")
    f.write(f"-----------------------------------------\n")
    f.write(f"Total pairs in deep trees (Tree 3+): {deep_tree_total}\n")
    f.write(f"Pairs assigned Independence Copula: {independence_count}\n")
    if deep_tree_total > 0:
        decay_rate = (independence_count / deep_tree_total) * 100
        f.write(f"Spatial Decay Rate: {decay_rate:.2f}%\n")
        if decay_rate > 50:
            f.write("CONCLUSION: Successful spatial decay. Deep conditional networks truncate to independence.\n")
        else:
            f.write("CONCLUSION: Spatial dependence persists in deep trees. Environmental variance may still be present.\n")
            
print(f"Independence decay quantitative check saved to: {output_decay}")
print(f"Mathematical Global Log-Likelihood: {global_ll:.4f}")
print(f"Mathematical Global AIC: {global_aic:.4f}")
