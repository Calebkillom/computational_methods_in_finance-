import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formula for European Put Option
def black_scholes_put(S, E, T, r, sigma):
    d1 = (np.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return E * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Finite Difference Methods (explicit, implicit, and crank-nicolson)
def finite_difference_methods(S, E, T, r, sigma, N, M, method="explicit"):
    S_max = max(S)
    S_min = min(S)
    ΔS = (S_max - S_min) / N
    Δt = T / M
    grid_S = np.linspace(S_min, S_max, N + 1)
    grid_t = np.linspace(0, T, M + 1)
    
    # Initialize the grid for option values
    V = np.zeros((N + 1, M + 1))
    
    # Set terminal conditions for European put option at maturity
    V[:, -1] = np.maximum(E - grid_S, 0)
    
    if method == "explicit":
        # Explicit finite difference method
        for j in range(M - 1, -1, -1):
            for i in range(1, N):
                V[i, j] = V[i, j + 1] + Δt * (
                    0.5 * sigma ** 2 * (grid_S[i] ** 2) * 
                    (V[i + 1, j + 1] - 2 * V[i, j + 1] + V[i - 1, j + 1]) / (ΔS ** 2) - 
                    r * grid_S[i] * (V[i + 1, j + 1] - V[i - 1, j + 1]) / (2 * ΔS))
    
    elif method == "implicit":
        # Implicit finite difference method
        diag_main = 1 + Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2)
        diag_upper = -0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 - r / (2 * ΔS))
        diag_lower = -0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / (2 * ΔS))
        A = np.diag(diag_main) + np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
        for j in range(M - 1, -1, -1):
            V[1:N, j] = np.linalg.solve(A, V[1:N, j + 1])
    
    elif method == "crank-nicolson":
        # Crank-Nicolson method
        theta = 0.5
        diag_main = (1 + Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2)) * (1 - theta)
        diag_upper = (-0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 - r / (2 * ΔS))) * (1 - theta)
        diag_lower = (-0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / (2 * ΔS))) * (1 - theta)
        A = np.diag(diag_main) + np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
        for j in range(M - 1, -1, -1):
            V[1:N, j] = np.linalg.solve(A, V[1:N, j + 1])
    
    return V[:, 0], grid_S  # Return the option values at t=0 (initial condition)

# Parameters
E = 10          # Strike price
r = 0.05        # Risk-free rate
T = 0.5         # Time to maturity
sigma = 0.2     # Volatility
S_min = 5       # Minimum stock price
S_max = 15      # Maximum stock price
N = 100         # Number of space steps
S = np.linspace(S_min, S_max, N+1)  # Stock prices

# Exact Black-Scholes prices for comparison
BS_prices = np.array([black_scholes_put(S_i, E, T, r, sigma) for S_i in S])

# Time steps M=30
M1 = 30
explicit_30, _ = finite_difference_methods(S, E, T, r, sigma, N, M1, method="explicit")
implicit_30, _ = finite_difference_methods(S, E, T, r, sigma, N, M1, method="implicit")
crank_nicolson_30, _ = finite_difference_methods(S, E, T, r, sigma, N, M1, method="crank-nicolson")

# Time steps M=500
M2 = 500
explicit_500, _ = finite_difference_methods(S, E, T, r, sigma, N, M2, method="explicit")
implicit_500, _ = finite_difference_methods(S, E, T, r, sigma, N, M2, method="implicit")
crank_nicolson_500, _ = finite_difference_methods(S, E, T, r, sigma, N, M2, method="crank-nicolson")

# Errors (difference between Black-Scholes and finite difference prices)
error_explicit_30 = np.abs(BS_prices - explicit_30)
error_implicit_30 = np.abs(BS_prices - implicit_30)
error_cn_30 = np.abs(BS_prices - crank_nicolson_30)

error_explicit_500 = np.abs(BS_prices - explicit_500)
error_implicit_500 = np.abs(BS_prices - implicit_500)
error_cn_500 = np.abs(BS_prices - crank_nicolson_500)

# Plot errors for M=30
plt.figure(figsize=(10, 6))
plt.plot(S, error_explicit_30, label="Explicit (M=30)", linestyle="--", color="blue")
plt.plot(S, error_implicit_30, label="Implicit (M=30)", linestyle="--", color="green")
plt.plot(S, error_cn_30, label="Crank-Nicolson (M=30)", linestyle="--", color="red")
plt.title("Error Comparison (M = 30)")
plt.xlabel("Stock Price (S)")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

# Plot errors for M=500
plt.figure(figsize=(10, 6))
plt.plot(S, error_explicit_500, label="Explicit (M=500)", linestyle="--", color="blue")
plt.plot(S, error_implicit_500, label="Implicit (M=500)", linestyle="--", color="green")
plt.plot(S, error_cn_500, label="Crank-Nicolson (M=500)", linestyle="--", color="red")
plt.title("Error Comparison (M = 500)")
plt.xlabel("Stock Price (S)")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

# Compare both M=30 and M=500 on the same plot
plt.figure(figsize=(10, 6))
plt.plot(S, error_explicit_30, label="Explicit (M=30)", linestyle="--", color="blue")
plt.plot(S, error_explicit_500, label="Explicit (M=500)", color="blue")
plt.plot(S, error_implicit_30, label="Implicit (M=30)", linestyle="--", color="green")
plt.plot(S, error_implicit_500, label="Implicit (M=500)", color="green")
plt.plot(S, error_cn_30, label="Crank-Nicolson (M=30)", linestyle="--", color="red")
plt.plot(S, error_cn_500, label="Crank-Nicolson (M=500)", color="red")
plt.title("Error Comparison: M = 30 vs M = 500")
plt.xlabel("Stock Price (S)")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()
