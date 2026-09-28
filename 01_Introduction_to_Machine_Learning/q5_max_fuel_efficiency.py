import pandas as pd

df = pd.read_csv('car_fuel_efficiency_2026.csv')

asia_df = df[df["origin"] == "Asia"]

print(f"Maximum fuel efficiency of cars from Asia : {asia_df["fuel_efficiency_mpg"].max()}")

