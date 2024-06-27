import numpy as np
import matplotlib.pyplot as plt

""" Parameters """
X0 = 1.0
mu = 0.06
sigma = 0.2
T = 1.0
num_steps_values = [10, 20, 50, 100, 200, 500, 1000]

""" Exact solution function for GBM """
def exact_solution(X0, mu, sigma, t):
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * np.sqrt(t) * np.random.normal()
    return X0 * np.exp(drift + diffusion)

""" Euler-Maruyama method for GBM """
def euler_maruyama(X0, mu, sigma, T, num_steps):
    dt = T / num_steps
    t_values = np.linspace(0.0, T, num_steps + 1)
    dt_values = np.diff(t_values)  # Array of time increments
    W = np.zeros(num_steps + 1)
    for k in range(num_steps):
        dW = np.sqrt(dt_values[k]) * np.random.normal()
        W[k + 1] = W[k] + dW
    X = np.zeros(num_steps + 1)
    X[0] = X0
    for i in range(num_steps):
        dW = W[i + 1] - W[i]
        X[i + 1] = X[i] + mu * X[i] * dt_values[i] + sigma * X[i] * dW
    return X[-1]

""" Compute errors for different values of Delta t """
errors = []

for num_steps in num_steps_values:
    exact = exact_solution(X0, mu, sigma, T)
    euler_approx = euler_maruyama(X0, mu, sigma, T, num_steps)
    errors.append(np.abs(exact - euler_approx))

""" Compute logarithms """
log_Delta = np.log(T / np.array(num_steps_values))
log_errors = np.log(errors)

""" Perform linear regression to find the coefficients """
coeffs = np.polyfit(log_Delta, log_errors, 1)
fit_line = coeffs[1] + coeffs[0] * log_Delta

""" Plot log-log plot """
plt.figure(figsize=(8, 6))
plt.scatter(log_Delta, log_errors, label='Computed Errors')
plt.plot(log_Delta, fit_line, color='red', linestyle='--',
          label=f'Fit: ln(epsilon) = {coeffs[1]:.5f} + {coeffs[0]:.5f} * ln(Delta)')
plt.xlabel('ln(Delta)')
plt.ylabel('ln(Absolute Error)')
plt.title('Log-log Plot of Absolute Error vs ln(Delta)')
plt.legend()
plt.grid(True)
plt.savefig('log_plot.png')
plt.show()

print(f"Coefficients from linear fit: intercept = {coeffs[1]}, slope = {coeffs[0]}")
