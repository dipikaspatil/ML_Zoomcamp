import pandas as pd

df = pd.read_csv('car_fuel_efficiency_2026.csv')

median_horsepower_before = df["horsepower"].median()

print(f"median horsepower before : {median_horsepower_before}")

mode_horsepower = df['horsepower'].mode()[0]
print(f"mode (most frequent) horsepower: {mode_horsepower}")

df["horsepower"] = df["horsepower"].fillna(mode_horsepower)

median_horsepower_after = df["horsepower"].median()
print(f"median horsepower after : {median_horsepower_after}")