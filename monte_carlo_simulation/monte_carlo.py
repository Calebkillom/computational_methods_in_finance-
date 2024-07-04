import numpy as np
from scipy.stats import norm

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
    return K * (S > K)

""" Euler scheme """
def euler_scheme(S0, sigma, r, T, m, N):
    dt = T / m
    paths = np.zeros((m+1, N))
    paths[0] = S0
    for t in range(1, m+1):
        Z = np.random.standard_normal(N)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

""" Milstein scheme """
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
    digital_call_price = K * np.exp(-r * T) * norm.cdf(d2)
    return digital_call_price

""" Monte Carlo simulation """
def monte_carlo_pricing(S0, sigma, r, K, T, m, N, scheme, payoff_function):
    paths = scheme(S0, sigma, r, T, m, N)
    payoffs = payoff_function(paths[-1], K)
    price = np.exp(-r * T) * np.mean(payoffs)
    sample_variance = np.var(payoffs, ddof=1)
    std_error = np.sqrt(sample_variance / N)
    return price, std_error

""" Number of paths """
Ns = [100, 1000, 5000, 10000, 50000, 100000]

""" Compute prices using Euler and Milstein schemes for European call option """
euler_prices = []
milstein_prices = []
euler_errors = []
milstein_errors = []

for N in Ns:
    euler_price, euler_error = monte_carlo_pricing(S0, sigma, r, K, T, m, N, euler_scheme, european_call_payoff)
    milstein_price, milstein_error = monte_carlo_pricing(S0, sigma, r, K, T, m, N, milstein_scheme, european_call_payoff)
    euler_prices.append(euler_price)
    milstein_prices.append(milstein_price)
    euler_errors.append(euler_error)
    milstein_errors.append(milstein_error)

""" Black-Scholes price for European call option """
bs_call_price = black_scholes_call(S0, K, T, r, sigma)

""" Compute prices using Euler and Milstein schemes for digital call option """
euler_digital_prices = []
milstein_digital_prices = []
euler_digital_errors = []
milstein_digital_errors = []

for N in Ns:
    euler_digital_price, euler_digital_error = monte_carlo_pricing(S0, sigma, r, K, T, m, N, euler_scheme, digital_call_payoff)
    milstein_digital_price, milstein_digital_error = monte_carlo_pricing(S0, sigma, r, K, T, m, N, milstein_scheme, digital_call_payoff)
    euler_digital_prices.append(euler_digital_price)
    milstein_digital_prices.append(milstein_digital_price)
    euler_digital_errors.append(euler_digital_error)
    milstein_digital_errors.append(milstein_digital_error)

""" Black-Scholes price for digital call option """
bs_digital_call_price = black_scholes_digital_call(S0, K, T, r, sigma)

""" Print results """
print("Black-Scholes Price for European Call Option:", bs_call_price)
print("\nNumber of Paths | Euler Price | Euler Error | Milstein Price | Milstein Error")
for i, N in enumerate(Ns):
    print(f"{N:<15} | {euler_prices[i]:<11.4f} | {euler_errors[i]:<11.4f} | {milstein_prices[i]:<14.4f} | {milstein_errors[i]:<13.4f}")

print("\nBlack-Scholes Price for Digital Call Option:", bs_digital_call_price)
print("\nNumber of Paths | Euler Digital Price | Euler Digital Error | Milstein Digital Price | Milstein Digital Error")
for i, N in enumerate(Ns):
    print(f"{N:<15} | {euler_digital_prices[i]:<18.4f} | {euler_digital_errors[i]:<17.4f} | {milstein_digital_prices[i]:<20.4f} | {milstein_digital_errors[i]:<19.4f}")
