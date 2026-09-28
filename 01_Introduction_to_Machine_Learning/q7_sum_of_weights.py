import pandas as pd
import numpy as np

df = pd.read_csv('car_fuel_efficiency_2026.csv')

# Select all the cars from Asia
asia_cars_df = df[df["origin"] == "Asia"]

# Select only columns vehicle_weight and model_year
asia_cars_df = asia_cars_df[["vehicle_weight", "model_year"]]


# Select the first 7 values
# Get the underlying NumPy array. Let's call it X.
X = asia_cars_df.iloc[:7].values
print(f"X: {X}")

# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
XT = X.T
print(f"XT: {XT}")
XTX = XT.dot(X)
print(f"XTX: {XTX}")

# Invert XTX.
XTX_inv = np.linalg.inv(XTX)
print(f"XTX_inv: {XTX_inv}")

# Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = XTX_inv.dot(XT).dot(y)

print(f"w : {w}")
print(f"Sum of all the elements of w : {w.sum()}")