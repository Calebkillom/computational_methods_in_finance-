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
volatility_range = np.arange(0.1, 0.5, 0.1)  # Range of volatility values
N = 100         # Number of space steps

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
errors_explicit_30 = []
errors_implicit_30 = []
errors_crank_nicolson_30 = []
errors_explicit_500 = []
errors_implicit_500 = []
errors_crank_nicolson_500 = []

# Compute errors
for S in S_steps:
    error_explicit_30 = []
    error_implicit_30 = []
    error_crank_nicolson_30 = []
    error_explicit_500 = []
    error_implicit_500 = []
    error_crank_nicolson_500 = []

    for sigma in volatility_range:
        BS_prices = black_scholes_put(S, E, T, r, sigma)

        V_explicit_30 = interpolate_to_steps(finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="explicit")[1],
                                              finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="explicit")[0],
                                              S_steps)

        V_implicit_30 = interpolate_to_steps(finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="implicit")[1],
                                              finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="implicit")[0],
                                              S_steps)

        V_crank_nicolson_30 = interpolate_to_steps(finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="crank-nicolson")[1],
                                                    finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="crank-nicolson")[0],
                                                    S_steps)

        V_explicit_500 = interpolate_to_steps(finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="explicit")[1],
                                               finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="explicit")[0],
                                               S_steps)

        V_implicit_500 = interpolate_to_steps(finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="implicit")[1],
                                               finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="implicit")[0],
                                               S_steps)

        V_crank_nicolson_500 = interpolate_to_steps(finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="crank-nicolson")[1],
                                                     finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="crank-nicolson")[0],
                                                     S_steps)

        error_explicit_30.append(np.abs(BS_prices - V_explicit_30[S_steps == S][0]))
        error_implicit_30.append(np.abs(BS_prices - V_implicit_30[S_steps == S][0]))
        error_crank_nicolson_30.append(np.abs(BS_prices - V_crank_nicolson_30[S_steps == S][0]))
        error_explicit_500.append(np.abs(BS_prices - V_explicit_500[S_steps == S][0]))
        error_implicit_500.append(np.abs(BS_prices - V_implicit_500[S_steps == S][0]))
        error_crank_nicolson_500.append(np.abs(BS_prices - V_crank_nicolson_500[S_steps == S][0]))

    errors_explicit_30.append(error_explicit_30)
    errors_implicit_30.append(error_implicit_30)
    errors_crank_nicolson_30.append(error_crank_nicolson_30)
    errors_explicit_500.append(error_explicit_500)
    errors_implicit_500.append(error_implicit_500)
    errors_crank_nicolson_500.append(error_crank_nicolson_500)

# Convert lists to arrays for easier plotting
errors_explicit_30 = np.array(errors_explicit_30)
errors_implicit_30 = np.array(errors_implicit_30)
errors_crank_nicolson_30 = np.array(errors_crank_nicolson_30)
errors_explicit_500 = np.array(errors_explicit_500)
errors_implicit_500 = np.array(errors_implicit_500)
errors_crank_nicolson_500 = np.array(errors_crank_nicolson_500)

# Plotting
plt.figure(figsize=(14, 10))

# M=30
plt.subplot(2, 1, 1)
plt.plot(S_steps, errors_explicit_30.mean(axis=1), marker='o', linestyle='-', label='Explicit (M=30)')
plt.plot(S_steps, errors_implicit_30.mean(axis=1), marker='s', linestyle='--', label='Implicit (M=30)')
plt.plot(S_steps, errors_crank_nicolson_30.mean(axis=1), marker='^', linestyle='-.', label='Crank-Nicolson (M=30)')
plt.title('Error Comparison (M=30)')
plt.xlabel('Stock Price (S)')
plt.ylabel('Average Absolute Error')
plt.legend()

# M=500
plt.subplot(2, 1, 2)
plt.plot(S_steps, errors_explicit_500.mean(axis=1), marker='o', linestyle='-', label='Explicit (M=500)')
plt.plot(S_steps, errors_implicit_500.mean(axis=1), marker='s', linestyle='--', label='Implicit (M=500)')
plt.plot(S_steps, errors_crank_nicolson_500.mean(axis=1), marker='^', linestyle='-.', label='Crank-Nicolson (M=500)')
plt.title('Error Comparison (M=500)')
plt.xlabel('Stock Price (S)')
plt.ylabel('Average Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
