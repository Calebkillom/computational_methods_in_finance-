import pandas as pd
import numpy as np
from scipy.stats import norm

""" Specify the correct file path """
file_path = (
    'C:\\Users\\bebbu\\OneDrive\\Desktop\\MaFin\\'
    'Computational_Methods_in_Finance\\computational_methods_in_finance-\\'
    'options_data.xlsx'
)

"""
Read the Excel file into a DataFrame,
skipping the second row and assuming the first row contains headers
"""
df = pd.read_excel(file_path, header=0, skiprows=[1])

""" Drop the first row (index 0) if it contains 'Strike' and NaN values """
df = df.drop(index=0, errors='ignore')

""" Drop the 'Unnamed: 1' column if it exists """
df = df.drop(columns=['Unnamed: 1'], errors='ignore')

""" Rename columns """
df = df.rename(columns={
    'Unnamed: 0': 'Strike',
    'Unnamed: 2': 'tao=0.25',
    'Unnamed: 3': 'tao=0.5',
    'Unnamed: 4': 'tao=1',
    'Unnamed: 5': 'tao=1.5'
})

""" Computation of the Implied Volatility By Newton's Method """
""" 
Calculate the Vega for Current Time
Calculate  Volatility for Next Period
Repeat Process 1 and 2
"""

""" List to store dictionaries """
data_list = []

""" Iterate through columns and rows """
for column in df.columns[1:]:
    tao_value = float(column.split('=')[1])
    for index, row in df.iterrows():
        strike = row['Strike']
        call_opt_price = row[column]
        
        """ Create dictionary and append to list """
        data_dict = {
            "strike": strike,
            "tao": tao_value,
            "call_opt_price": call_opt_price
        }
        data_list.append(data_dict)

""" Calculating the Black scholes price for the first period """
""" Access the first element """
first_element = data_list[0]
k = first_element["strike"]
tau = first_element["tao"]

""" Given parameters """
S = 100
r = 0.03
q = 0
sigma = 0.25

"""  Black-Scholes formula """
def black_scholes_call(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

""" Calculate the call option price """
call_option_price = black_scholes_call(S, k, tau, r, sigma, q)

""" Calculating the Vega """
def black_scholes_vega(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return vega
call_option_vega = black_scholes_vega(S, k, tau, r, sigma, q)

""" List to store results """
implied_volatility_data = []

""" Iterate through the entries in data_list and calculate implied volatility for each entry """
for i in range(len(data_list)):
    data_dict = data_list[i]
    strike = data_dict["strike"]
    tau = data_dict["tao"]
    market_price = data_dict["call_opt_price"]
    
    """ Calculate Vega for this specific strike and tau """
    vega = black_scholes_vega(S, strike, tau, r, sigma, q)
    
    """ Use initial sigma (current sigma) as 0.15 (or any initial guess you prefer) """
    initial_sigma = 0.15
    
    """ Define Newton-Raphson method for implied volatility (single iteration) """
    def newton_raphson_iv(S, K, T, r, market_price, initial_sigma, q):
        sigma = initial_sigma
        
        """ Calculate theoretical price and vega for the current sigma """
        theoretical_price = black_scholes_call(S, K, T, r, sigma, q)
        vega = black_scholes_vega(S, K, T, r, sigma, q)
        
        """ Update sigma using Newton-Raphson method (single iteration) """
        sigma_new = sigma - (theoretical_price - market_price) / vega
        
        return sigma_new
    
    """ Calculate implied volatility for this strike and tau """
    implied_volatility = newton_raphson_iv(S, strike, tau, r, market_price, initial_sigma, q)
    
    """ Append dictionary with implied volatility, strike, and tao to the list """
    implied_volatility_data.append({
        "implied_volatility": implied_volatility,
        "strike": strike,
        "maturity": tau
    })

""" Print the new array of dictionaries with implied volatilities """
for item in implied_volatility_data:
    print(item)