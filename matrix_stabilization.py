import pandas as pd
import numpy as np
import os

print("Spatiotemporal Matrix Construction and Zero-Inflation Anomaly Filtering...")

folder_path = '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/Spatial Vine Copula/MATHMOD_Official_Final_Submission_Archive/Spatial_Matrices_CSV'
file_path = os.path.join(folder_path, 'Filtered_Proxy_NTL_Matrix.csv')

# Task 3: Load matrix and verify mathematical shape Y ∈ R^(n x d)
df = pd.read_csv(file_path)
n_snapshots, d_nodes = df.shape
print(f"Mathematical Matrix Shape Verified: Y ∈ R^({n_snapshots} x {d_nodes})")

# Task 4 & 5: Apply zero-inflation numerical offset to absolute zeros
epsilon = 0.001
print(f"Applying spatial noise filter offset: epsilon = {epsilon}")

# THE FIX: Apply offset to all 5 spatial nodes instead of just 'Radiance'
for col in df.columns:
    if col != 'Date': 
        df[col] = np.where(df[col] <= 0, epsilon, df[col])

# Task 6: Export stabilized matrix
df.to_csv(file_path, index=False)
print("Stabilized physical radiance matrix successfully exported.")
