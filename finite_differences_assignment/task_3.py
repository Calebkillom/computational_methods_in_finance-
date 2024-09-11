import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
import mpmath
from mpmath import mp

# Set precision for mpmath
mp.dps = 50  # Decimal places of precision

# Parameters
E = mp.mpf(10)          # Strike price
r = mp.mpf(0.05)        # Risk-free rate
T = mp.mpf(0.5)         # Time to maturity
S_min = mp.mpf(5)       # Minimum stock price
S_max = mp.mpf(15)      # Maximum stock price
S_steps = np.arange(float(S_min), float(S_max) + 1, 1)  # Stock price range
volatility = 0.2       # Constant volatility value
M = 30                 # Number of time steps

# Black-Scholes formula for European Put Option
def black_scholes_put(S, E, T, r, sigma):
    S = mp.mpf(S)
    E = mp.mpf(E)
    T = mp.mpf(T)
    r = mp.mpf(r)
    sigma = mp.mpf(sigma)
    
    d1 = (mp.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * mp.sqrt(T))
    d2 = d1 - sigma * mp.sqrt(T)
    
    # Convert to float for scipy.stats.norm functions
    d1_float = float(d1)
    d2_float = float(d2)
    
    return E * np.exp(-float(r) * float(T)) * norm.cdf(-d2_float) - S * norm.cdf(-d1_float)

# Finite Difference Methods
def finite_difference_methods(S, E, T, r, sigma, N, M, method="explicit"):
    S_max = max(S)
    S_min = min(S)
    ΔS = (S_max - S_min) / N
    Δt = T / M
    grid_S = np.linspace(float(S_min), float(S_max), N + 1)
    grid_t = np.linspace(0, float(T), M + 1)
    
    V = np.zeros((N + 1, M + 1))
    V[:, -1] = np.maximum(E - grid_S, 0)

    try:
        if method == "explicit":
            for j in range(M - 1, -1, -1):
                for i in range(1, N):
                    V[i, j] = V[i, j + 1] + Δt * (0.5 * sigma ** 2 * (grid_S[i] ** 2) * 
                            (V[i + 1, j + 1] - 2 * V[i, j + 1] + V[i - 1, j + 1]) / (ΔS ** 2) - 
                            r * grid_S[i] * (V[i + 1, j + 1] - V[i - 1, j + 1]) / (2 * ΔS))

        elif method == "implicit":
            diag_main = 1 + Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2)
            diag_upper = -0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 - r / (2 * ΔS))
            diag_lower = -0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / (2 * ΔS))

            A = np.diag(diag_main) + np.diag(diag_upper[:-1], 1) + np.diag(diag_lower[:-1], -1)
            B = np.diag(1 - Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2))
            
            for j in range(M - 1, -1, -1):
                V[1:N, j] = np.linalg.solve(A, B @ V[1:N, j + 1])

        elif method == "crank-nicolson":
            diag_main = 1 + 0.5 * Δt * (sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2)
            diag_upper = -0.25 * Δt * (sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / ΔS)
            diag_lower = -0.25 * Δt * (sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 - r / ΔS)
            
            A_CN = np.diag(diag_main) + np.diag(diag_upper[:-1], 1) + np.diag(diag_lower[:-1], -1)
            B_CN = np.diag(diag_main) + np.diag(-diag_upper[:-1], 1) + np.diag(-diag_lower[:-1], -1)
            
            for j in range(M - 1, -1, -1):
                V[1:N, j] = np.linalg.solve(A_CN, B_CN @ V[1:N, j + 1])
    
    except Exception as e:
        print(f"Error in {method} method: {e}")
    
    return V[:, 0], grid_S

# Interpolation function
def interpolate_to_steps(grid_S, values, S_steps):
    interp_func = interp1d(grid_S, values, kind='linear', fill_value='extrapolate')
    return interp_func(S_steps)

# Initialize arrays to store errors
N_values = np.arange(300, 701, 50)  # Vary N from 300 to 700 with step of 50
errors_explicit = np.zeros((len(N_values), len(S_steps)), dtype=float)
errors_implicit = np.zeros((len(N_values), len(S_steps)), dtype=float)
errors_crank_nicolson = np.zeros((len(N_values), len(S_steps)), dtype=float)

# Compute errors for different N values
for k, N in enumerate(N_values):
    print(f"Computing for N = {N}...")
    
    for idx, S in enumerate(S_steps):
        BS_price = black_scholes_put(S, E, T, r, volatility)

        V_explicit, grid_S_explicit = finite_difference_methods(S_steps, E, T, r, volatility, N=N, M=M, method="explicit")
        V_implicit, grid_S_implicit = finite_difference_methods(S_steps, E, T, r, volatility, N=N, M=M, method="implicit")
        V_crank_nicolson, grid_S_crank_nicolson = finite_difference_methods(S_steps, E, T, r, volatility, N=N, M=M, method="crank-nicolson")

        V_explicit_interp = interpolate_to_steps(grid_S_explicit, V_explicit, [S])
        V_implicit_interp = interpolate_to_steps(grid_S_implicit, V_implicit, [S])
        V_crank_nicolson_interp = interpolate_to_steps(grid_S_crank_nicolson, V_crank_nicolson, [S])

        errors_explicit[k, idx] = np.abs(BS_price - V_explicit_interp[0])
        errors_implicit[k, idx] = np.abs(BS_price - V_implicit_interp[0])
        errors_crank_nicolson[k, idx] = np.abs(BS_price - V_crank_nicolson_interp[0])

# Plot errors
plt.figure(figsize=(14, 7))

for k, N in enumerate(N_values):
    plt.plot(S_steps, errors_explicit[k, :], label=f'Explicit (N={N})', marker='o')
    plt.plot(S_steps, errors_implicit[k, :], label=f'Implicit (N={N})', marker='x')
    plt.plot(S_steps, errors_crank_nicolson[k, :], label=f'Crank-Nicolson (N={N})', marker='^')

plt.xlabel('Stock Price')
plt.ylabel('Error')
plt.title('Errors of Finite Difference Methods for European Put Option Pricing with Varying Space Steps')
plt.legend()
plt.grid(True)
plt.savefig('finite_difference_errors_varying_N.pdf')
plt.show()
