
# **A Spatio-Temporal Vine Copula Mathematical Model for High-Dimensional Dependence in PhilSA Nighttime Light Radiance**
MATHMOD-Spatial-Vine-Copula
_Thesis Mathematical Modelling Project_

(current temporary sub-data: Google Engine)

This repository will contain the full execution pipeline and methodology for modeling high-dimensional spatial dependence in **Philippine Satellite Agency (PhilSA) Nighttime Light (NTL)** radiance data. The framework employs a **Spatio-Temporal C-Vine Copula** to capture non-linear, asymmetric spatial dependencies, specifically targeting extreme physical radiance anomalies like sensor blooming (overglow) and atmospheric attenuation.Repository Structure


The master directory MATHMOD_Official_Final_Submission_Archive is organized as follows:

- **Source_Code_Scripts:** Contains all executable Python scripts for preprocessing, MLE estimation, goodness-of-fit testing, and model validation.

- **Spatial_Matrices_CSV:** Stores processed matrices, including filtered radiance data, uniform pseudo-observations, parameter optimization outputs, and benchmark results.

- **High_Res_Diagnostic_Plots:** Contains high-resolution visualizations of PIT histograms, copula density contours, LISA residual cluster maps, and extreme anomaly projections.

- **ETD_Manuscript**: Houses the formal LaTeX-based thesis documentation and methodology chapters.


**Key Methodological Components**

- **Marginal Estimation:** Utilization of Inference Functions for Margins (IFM) with mixture models to account for zero-inflation (unlit/occluded areas) and right-skewed parametric distributions (Weibull/Log-Normal).

- **Spatial Dependence:** Implementation of canonical vine (C-Vine) architectures to model hierarchical dependence structures, optimized via sequential Maximum Likelihood Estimation (MLE).

- **Extreme Value Theory (EVT):** Integration of Generalized Pareto Distribution (GPD) threshold exceedance to rigorously model upper-tail dependencies and physical radiance spikes.

- **Validation Framework:** Comprehensive benchmarking against classical spatial econometrics (SAR models), machine learning baselines (Random Forest), and statistical validity tests (Cramér-von Mises, Diebold-Mariano, Vuong’s Test).


**Technical Stack**

- **Environment:** Google Colab, managed via Git for version control.

- **Core Libraries:**
  - pyvinecopulib (Copula estimation)
  - scipy (Statistical optimization)
  - pandas & geopandas (Data structures)
  - networkx (Graph topology)
  - esda
  - PySAL (Spatial autocorrelation).


**Repository Usage**

Scripts are designed to be executed sequentially according to the PhilSA_DALOI_Tracker.txt file located in the Source_Code_Scripts directory. Ensure environment is authenticated with Google Drive to access the Spatial_Matrices_CSV folder before running the Master Execution Pipeline.
