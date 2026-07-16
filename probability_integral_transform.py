import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

print("Executing Day 4: Probability Integral Transform (PIT)...")

# Define paths safely
base_path = (
    '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/'
    'A Spatial Copula Mathematical Model for Evaluating Regional Economic Dependencies using '
    'PhilSA Nighttime Light Radiance/MATHMOD_Official_Final_Submission_Archive/'
)
csv_folder = os.path.join(base_path, 'Spatial_Matrices_CSV')
plot_folder = os.path.join(base_path, 'High_Res_Diagnostic_Plots')
os.makedirs(plot_folder, exist_ok=True)

data_file = os.path.join(csv_folder, 'Filtered_Proxy_NTL_Matrix.csv')
params_file = os.path.join(csv_folder, 'Marginal_Parameters_Summary.csv')
output_csv = os.path.join(csv_folder, 'Uniform_Pseudo_Obs_Matrix.csv')
output_plot = os.path.join(plot_folder, 'Diagnostics_PIT_Histograms.png')

# Load data matrices
df = pd.read_csv(data_file)
params_df = pd.read_csv(params_file)

# Initialize uniform matrix dataframe
u_df = pd.DataFrame()

# Task 3 & 4: PIT Algorithm application
for index, row in params_df.iterrows():
    node = row['Spatial_Node']
    dist = row['Best_Distribution']
    shape = row['Shape_Parameter']
    scale = row['Scale_Parameter']

    if node in df.columns:
        data = df[node].dropna()
        if dist == 'Weibull':
            # Scipy uses 'c' for shape in weibull_min
            u_scores = stats.weibull_min.cdf(data, c=shape, loc=0, scale=scale)
        elif dist == 'Log-Normal':
            u_scores = stats.lognorm.cdf(data, s=shape, loc=0, scale=scale)

        u_df[node] = u_scores
        print(f"PIT mathematically applied for {node} using {dist} CDF.")

        # Task 6: Generate diagnostic PIT histogram plots
        plt.figure(figsize=(6, 4))
        plt.hist(u_scores, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'PIT Uniform Pseudo-Observations: {node}')
        plt.xlabel('Uniform u-scores')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"Diagnostic histogram perfectly saved for {node}.")

# Task 5: Export Uniform Pseudo-Observations Matrix
u_df.to_csv(output_csv, index=False)
print(f"Uniform matrix successfully exported to {output_csv}")
