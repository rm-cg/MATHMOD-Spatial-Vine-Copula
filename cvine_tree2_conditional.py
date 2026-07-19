# DAY 8: Conditional Pseudo-Observations via h-functions and Tree 2 Initialization

import pandas as pd
import numpy as np
import pyvinecopulib as pv
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

print("Executing Day 8: Conditional Pseudo-Observations and Tree 2 Initialization...")

base_csv_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
plot_folder = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/High_Res_Diagnostic_Plots'
os.makedirs(plot_folder, exist_ok=True)

data_file = os.path.join(base_csv_path, 'Uniform_Pseudo_Obs_Matrix.csv')
tree1_params_file = os.path.join(base_csv_path, 'Tree1_Parameters_Optimization.csv')
output_cond_csv = os.path.join(base_csv_path, 'Tree2_Conditional_PseudoObs.csv')
output_tree2_params = os.path.join(base_csv_path, 'Tree2_Parameters_Optimization.csv')
output_plot = os.path.join(plot_folder, 'Diagnostics_Tree2_Scatterplots.png')

u_df = pd.read_csv(data_file)
tree1_df = pd.read_csv(tree1_params_file)

copula_pool = [
    pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe
]
controls = pv.FitControlsBicop(family_set=copula_pool, selection_criterion="aic")

print("Computing analytical partial derivatives (h-functions) for conditional transformation...")
cond_pseudo_obs = pd.DataFrame(index=u_df.index)
tree2_results = []

if not tree1_df.empty:
    # THE FIX: Added so it extracts the actual string name (e.g. 'NCR_Manila')
    root_node = tree1_df['Source_Node'].mode()[0] # Extract the scalar value from the Series
    cond_pseudo_obs[root_node] = u_df[root_node]

    connected_edges = tree1_df[tree1_df['Source_Node'] == root_node]

    for index, row in connected_edges.iterrows():
        target = row['Target_Node']
        u_data = u_df[[root_node, target]].dropna().values

        # THE FIX: Empty copula initialization for the new library version
        bicop = pv.Bicop()
        bicop.select(u_data, controls)

        cond_vals = bicop.hfunc1(u_data)
        cond_pseudo_obs[f"{target}|{root_node}"] = cond_vals

        params = bicop.parameters.flatten() if len(bicop.parameters) > 0 else []
        tree2_results.append({
            'Tree_Layer': 2,
            'Source_Node': f"{target}|{root_node}",
            'Target_Node': root_node,
            'Copula_Family': bicop.family.name,
            'Parameter_1': params[0] if len(params) > 0 else np.nan,
            'Parameter_2': params[1] if len(params) > 1 else np.nan,
            'Tau': bicop.tau,
            'AIC': bicop.aic(u_data)
        })

    cond_pseudo_obs.to_csv(output_cond_csv, index=False)
    pd.DataFrame(tree2_results).to_csv(output_tree2_params, index=False)
    print(f"Conditional pseudo-observations successfully saved to: {output_cond_csv}")
    print(f"Tree 2 parameters successfully secured to: {output_tree2_params}")

    print("Generating High-Resolution Diagnostic Scatterplots for Tree 2...")
    plt.figure(figsize=(8, 6))
    if len(cond_pseudo_obs.columns) > 2:
        col1, col2 = cond_pseudo_obs.columns[-2], cond_pseudo_obs.columns[-1]
        plt.scatter(cond_pseudo_obs[col1], cond_pseudo_obs[col2], alpha=0.5, color='coral', edgecolor='black')
        plt.title(f'Tree 2 Conditional Joint Distribution\n{col1} vs {col2}')
        plt.xlabel(f'Conditional u-scores ({col1})')
        plt.ylabel(f'Conditional u-scores ({col2})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        print(f"Tree 2 Diagnostics successfully saved to: {output_plot}")
    plt.close()
else:
    print("WARNING: Tree 1 parameters missing. Cannot initialize Tree 2.")
