import pandas as pd
import numpy as np
import scipy.stats as stats
import os

print("Executing Day 3: Inference Functions for Margins (IFM) and Parametric MLE...")

# Define paths using the strict PDF-safe structure
folder_path = (
    '/content/drive/MyDrive/RESEARCH PAPERS/MATHEMATICAL MODELLING/'
    'A Spatial Copula Mathematical Model for Evaluating Regional Economic Dependencies using '
    'PhilSA Nighttime Light Radiance/MATHMOD_Official_Final_Submission_Archive/'
    'Spatial_Matrices_CSV'
)
input_file = os.path.join(folder_path, 'Filtered_Proxy_NTL_Matrix.csv')
output_file = os.path.join(folder_path, 'Marginal_Parameters_Summary.csv')

# Task 1: Load matrix
df = pd.read_csv(input_file)
results = []

# Task 3, 4 & 5: IFM MLE fitting and AIC benchmarking loop
for col in df.columns:
    if col in ['Longitude', 'Latitude', 'geometry']: # Skip coordinate metadata if present
        continue
        
    data = df[col].dropna()
    if len(data) == 0: 
        continue

    # Fit Log-Normal
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(data, floc=0)
    ll_lognorm = np.sum(stats.lognorm.logpdf(data, shape_lognorm, loc=loc_lognorm, scale=scale_lognorm))
    aic_lognorm = 2 * 2 - 2 * ll_lognorm  # 2 parameters

    # Fit Weibull (weibull_min in scipy)
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data, floc=0)
    ll_weibull = np.sum(stats.weibull_min.logpdf(data, shape_weibull, loc=loc_weibull, scale=scale_weibull))
    aic_weibull = 2 * 2 - 2 * ll_weibull

    # Lock in best fit automatically
    if aic_lognorm < aic_weibull:
        best_dist = 'Log-Normal'
        best_shape = shape_lognorm
        best_scale = scale_lognorm
        best_aic = aic_lognorm
    else:
        best_dist = 'Weibull'
        best_shape = shape_weibull
        best_scale = scale_weibull
        best_aic = aic_weibull

    results.append({
        'Spatial_Node': col,
        'Best_Distribution': best_dist,
        'Shape_Parameter': best_shape,
        'Scale_Parameter': best_scale,
        'AIC_Score': best_aic
    })
    print(f"Node {col}: Selected {best_dist} (AIC: {best_aic:.2f})")

# Task 6: Export optimized mathematical parameter array
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"Marginal parameters successfully secured: {output_file}")
