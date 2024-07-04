import numpy as np
import matplotlib.pyplot as plt

""" Parameters """
kappa = 0.5
v0 = 0.1
v_bar = 0.1
sigma = 0.1
T = 1
N = 10000
dt = T / N
M = 10000

""" Function to generate paths for CIR model """
def generate_CIR_paths(kappa, v0, v_bar, sigma, T, N, M):
    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = v0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=M)
        dW = np.sqrt(dt) * Z
        paths[:, i] = paths[:, i - 1] + kappa * (v_bar - paths[:, i - 1]) * dt + sigma * np.sqrt(paths[:, i - 1]) * dW
        paths[:, i] = np.maximum(paths[:, i], 0)  # Ensure non-negativity
        
    return paths

""" Function to generate paths for Vasicek model """
def generate_Vasicek_paths(kappa, v0, v_bar, sigma, T, N, M):
    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = v0
    
    for i in range(1, N + 1):
        Z = np.random.normal(size=M)
        dW = np.sqrt(dt) * Z
        paths[:, i] = paths[:, i - 1] + kappa * (v_bar - paths[:, i - 1]) * dt + sigma * dW
        
    return paths

""" Generate paths """
cir_paths = generate_CIR_paths(kappa, v0, v_bar, sigma, T, N, M)
vasicek_paths = generate_Vasicek_paths(kappa, v0, v_bar, sigma, T, N, M)

""" Extract final values """
cir_final_values = cir_paths[:, -1]
vasicek_final_values = vasicek_paths[:, -1]

""" Plot the PDFs """
plt.figure(figsize=(12, 6))
plt.hist(cir_final_values, bins=50, density=True, alpha=0.6, label='CIR Model')
plt.hist(vasicek_final_values, bins=50, density=True, alpha=0.6, label='Vasicek Model')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('PDF of Final Values of CIR and Vasicek Models')
plt.legend()
plt.savefig('CIR_vs_Vasicek_PDF.pdf')
plt.show()
