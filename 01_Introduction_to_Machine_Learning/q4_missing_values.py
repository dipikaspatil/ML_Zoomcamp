import pandas as pd

df = pd.read_csv('car_fuel_efficiency_2026.csv')

print(f"missing-value counts per column \n {df.isnull().sum()}") # missing-value counts per column

print(f"final count of columns that have at least one missing value {(df.isnull().sum() > 0).sum()}") # final count of columns that have at least one missing value