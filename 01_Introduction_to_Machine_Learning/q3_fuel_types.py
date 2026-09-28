import pandas as pd

df = pd.read_csv('car_fuel_efficiency_2026.csv')

print(df['fuel_type'].nunique())