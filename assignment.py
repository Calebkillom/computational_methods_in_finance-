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

""" Print the modified DataFrame """
print(df)

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
for item in data_list:
    print(item)