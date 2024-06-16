import pandas as pd

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