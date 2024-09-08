import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
E = 10          # Strike price
r = 0.05        # Risk-free rate
T = 0.5         # Time to maturity
sigma = 0.2     # Volatility
S_min = 5       # Minimum stock price
S_max = 15      # Maximum stock price
S_steps = np.arange(S_min, S_max + 1, 1)  # Stock price range

# Black-Scholes formula for European Put Option
def black_scholes_put(S, E, T, r, sigma):
    d1 = (np.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return E * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Finite Difference Methods
def finite_difference_methods(S, E, T, r, sigma, N, M, method="explicit"):
    # Space and time grids
    S_max = max(S)
    S_min = min(S)
    ΔS = (S_max - S_min) / N
    Δt = T / M
    grid_S = np.linspace(S_min, S_max, N + 1)
    grid_t = np.linspace(0, T, M + 1)
    
    # Initialize option value matrix with correct shape
    V = np.zeros((N + 1, M + 1))
    V[:, -1] = np.maximum(E - grid_S, 0)  # Payoff at maturity

    if method == "explicit":
        # Explicit Method
        for j in range(M - 1, -1, -1):
            for i in range(1, N):  # Stock price steps
                V[i, j] = V[i, j + 1] + Δt * (0.5 * sigma ** 2 * (grid_S[i] ** 2) * 
                        (V[i + 1, j + 1] - 2 * V[i, j + 1] + V[i - 1, j + 1]) / (ΔS ** 2) - 
                        r * grid_S[i] * (V[i + 1, j + 1] - V[i - 1, j + 1]) / (2 * ΔS))

    elif method == "implicit":
        # Implicit Method
        # Create matrices of the correct size
        diag_main = 1 + Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2)
        diag_upper = -0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 - r / (2 * ΔS))
        diag_lower = -0.5 * Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / (2 * ΔS))

        # Construct the matrix A
        A = np.diag(diag_main) + np.diag(diag_upper[:-1], 1) + np.diag(diag_lower[:-1], -1)

        # Matrix B (identity matrix)
        B = np.diag(1 - Δt * (0.5 * sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2))
        
        for j in range(M - 1, -1, -1):
            V[1:N, j] = np.linalg.solve(A, B @ V[1:N, j + 1])

    elif method == "crank-nicolson":
        # Crank-Nicolson Method
        diag_main = 1 + 0.5 * Δt * (sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / 2)
        diag_upper = -0.25 * Δt * (sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 + r / ΔS)
        diag_lower = -0.25 * Δt * (sigma ** 2 * (np.arange(1, N)) ** 2 / ΔS ** 2 - r / ΔS)
        
        A_CN = np.diag(diag_main) + np.diag(diag_upper[:-1], 1) + np.diag(diag_lower[:-1], -1)
        B_CN = np.diag(diag_main) + np.diag(-diag_upper[:-1], 1) + np.diag(-diag_lower[:-1], -1)
        
        for j in range(M - 1, -1, -1):
            V[1:N, j] = np.linalg.solve(A_CN, B_CN @ V[1:N, j + 1])
    
    return V[:, 0], grid_S  # Return price at current time and the stock price grid


# Calculate exact Black-Scholes prices
BS_prices = black_scholes_put(S_steps, E, T, r, sigma)

# Finite difference prices for different M
V_explicit_30 = finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="explicit")[0]
V_explicit_500 = finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="explicit")[0]

V_implicit_30 = finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="implicit")[0]
V_implicit_500 = finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="implicit")[0]

V_crank_nicolson_30 = finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=30, method="crank-nicolson")[0]
V_crank_nicolson_500 = finite_difference_methods(S_steps, E, T, r, sigma, N=100, M=500, method="crank-nicolson")[0]

# Errors
error_explicit_30 = np.abs(BS_prices - V_explicit_30)
error_explicit_500 = np.abs(BS_prices - V_explicit_500)

error_implicit_30 = np.abs(BS_prices - V_implicit_30)
error_implicit_500 = np.abs(BS_prices - V_implicit_500)

error_crank_nicolson_30 = np.abs(BS_prices - V_crank_nicolson_30)
error_crank_nicolson_500 = np.abs(BS_prices - V_crank_nicolson_500)

# Plotting errors
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(S_steps, error_explicit_30, label='Explicit Method (M=30)')
plt.plot(S_steps, error_explicit_500, label='Explicit Method (M=500)')
plt.title('Error in Explicit Method')
plt.xlabel('Stock Price')
plt.ylabel('Absolute Error')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(S_steps, error_implicit_30, label='Implicit Method (M=30)')
plt.plot(S_steps, error_implicit_500, label='Implicit Method (M=500)')
plt.title('Error in Implicit Method')
plt.xlabel('Stock Price')
plt.ylabel('Absolute Error')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(S_steps, error_crank_nicolson_30, label='Crank-Nicolson Method (M=30)')
plt.plot(S_steps, error_crank_nicolson_500, label='Crank-Nicolson Method (M=500)')
plt.title('Error in Crank-Nicolson Method')
plt.xlabel('Stock Price')
plt.ylabel('Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
