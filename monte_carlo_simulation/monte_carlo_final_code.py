import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

""" Parameters """
S0 = 5
sigma = 0.30
r = 0.06
K = S0
T = 1
m = 1000
dt = T / m

""" Define the payoff function for a European call option """
def european_call_payoff(S, K):
    return np.maximum(S - K, 0)

""" Define the payoff function for a digital call option """
def digital_call_payoff(S, K):
    return 1 * (S > K)  # Payoff is 1 if S > K, otherwise 0

""" Euler scheme for asset path generation """
def euler_scheme(S0, sigma, r, T, m, N):
    dt = T / m
    paths = np.zeros((m+1, N))
    paths[0] = S0
    for t in range(1, m+1):
        Z = np.random.standard_normal(N)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

""" Milstein scheme for asset path generation """
def milstein_scheme(S0, sigma, r, T, m, N):
    dt = T / m
    paths = np.zeros((m+1, N))
    paths[0] = S0
    for t in range(1, m+1):
        Z = np.random.standard_normal(N)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + 0.5 * sigma**2 * (np.sqrt(dt) * Z)**2)
    return paths

""" Black-Scholes formula for European call option """
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

""" Black-Scholes formula for digital call option """
def black_scholes_digital_call(S0, K, T, r, sigma):
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    digital_call_price = np.exp(-r * T) * norm.cdf(d2)
    return digital_call_price

""" Monte Carlo simulation for pricing options """
def monte_carlo_pricing(S0, sigma, r, K, T, m, N, scheme, payoff_function):
    paths = scheme(S0, sigma, r, T, m, N)
    payoffs = payoff_function(paths[-1], K)
    price = np.exp(-r * T) * np.mean(payoffs)
    sample_variance = np.var(payoffs, ddof=1)
    std_error = np.sqrt(sample_variance / N)
    return price, std_error

""" Number of paths to test """
Ns = [100, 1000, 5000, 10000, 50000, 100000]

""" Black-Scholes prices for European and Digital Call options """
bs_call_price = black_scholes_call(S0, K, T, r, sigma)
bs_digital_call_price = black_scholes_digital_call(S0, K, T, r, sigma)

""" Function to calculate and plot differences for both European and Digital options """
def plot_differences(option_type, payoff_function, bs_price, filename):
    euler_differences = []
    milstein_differences = []

    for N in Ns:
        # Euler scheme
        paths_euler = euler_scheme(S0, sigma, r, T, m, N)
        price_euler = np.exp(-r * T) * np.mean(payoff_function(paths_euler[-1], K))
        euler_differences.append(price_euler - bs_price)

        # Milstein scheme
        paths_milstein = milstein_scheme(S0, sigma, r, T, m, N)
        price_milstein = np.exp(-r * T) * np.mean(payoff_function(paths_milstein[-1], K))
        milstein_differences.append(price_milstein - bs_price)

    # Plot the differences
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, euler_differences, marker='o', label='Euler Scheme')
    plt.plot(Ns, milstein_differences, marker='s', label='Milstein Scheme')
    plt.xscale('log')
    plt.title(f"Difference between {option_type} Option Prices and Black-Scholes Price")
    plt.xlabel("Number of Paths (N)")
    plt.ylabel("Price Difference")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(filename, format='pdf')  # Save as a PDF
    plt.show()

""" Plot for European Call Option and save to a PDF """
plot_differences("European Call", european_call_payoff, bs_call_price, "european_call_plot.pdf")

""" Plot for Digital Call Option and save to a PDF """
plot_differences("Digital Call", digital_call_payoff, bs_digital_call_price, "digital_call_plot.pdf")
