
## Homework 1

🔗 [Homework source](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/01-intro/homework.md) | 📝 [Submit answers](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01)

> Uses the pinned **2026 Car Fuel Efficiency** dataset.

### Setup

Install Python, NumPy, Pandas, Matplotlib, and Seaborn (see the [1.6 notes above](#16-setting-up-the-environment) 

Using a Python virtual environment (`venv`) instead of conda:

```bash
# Create the venv
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### Getting the data

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

### Questions

#### Q1. Pandas version
What version of Pandas did you install?

You can get the version information using the __version__ field:

```python
(venv) >> % python3 q1_pandas_version.py
3.0.6
```

#### Q2. Records count

Read data with Pandas. How many records are in the dataset?

- 5000
- 9000
- 10000 <-- answer
- 15000

```shell
(venv) >> % python3 q2_records_count.py
(10000, 11)
```

#### Q3. Fuel types
How many fuel types are presented in the dataset?

- 1
- 2
- 3 <-- answer
- 4

```shell
(venv) >> % python3 q3_fuel_types.py
3
```

#### Q4. Missing values
How many columns in the dataset have missing values?

- 0
- 1
- 2 <-- answer
- 3
- 4

```shell
(venv) >> % python3 q4_missing_values.py
missing-value counts per column
 model_year               0
origin                   0
fuel_type                0
drivetrain               0
num_doors                0
engine_displacement      0
num_cylinders            0
horsepower             877
vehicle_weight           0
acceleration           264
fuel_efficiency_mpg      0
dtype: int64

final count of columns that have at least one missing value 2
```

#### Q5. Max fuel efficiency
What's the maximum fuel efficiency of cars from Asia?

21.2
31.2
41.2 <-- answer
51.2

```shell
(venv) >> % python3 q5_max_fuel_efficiency.py
Maximum fuel efficiency of cars from Asia : 41.2
```

#### Q6. Median value of horsepower
Find the median value of the horsepower column in the dataset.
Next, calculate the most frequent value of the same horsepower column.
Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
Now, calculate the median value of horsepower once again.
Has it changed?

Yes, it increased
Yes, it decreased <-- answer
No

```shell
(venv) >> % python3 q6_median_value_of_horsepower.py
median horsepower before : 254.0
mode (most frequent) horsepower: 252.0
median horsepower after : 252.0
```

#### Q7. Sum of weights
1. Select all the cars from Asia
2. Select only columns vehicle_weight and model_year
3. Select the first 7 values
4. Get the underlying NumPy array. Let's call it X.
5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
6. Invert XTX.
7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.

What's the sum of all the elements of the result?
Note: You just implemented linear regression. We'll talk about it in the next lesson.

0.0369
0.369 <-- answer
3.69
36.9

```shell
(venv) >> % python3 q7_sum_of_weights.py
X: [[4240 1996]
 [4310 1992]
 [4230 1997]
 [4420 1978]
 [3950 2007]
 [4750 2021]
 [4450 2019]]
XT: [[4240 4310 4230 4420 3950 4750 4450]
 [1996 1992 1997 1978 2007 2021 2019]]
XTX: [[131950500  60750580]
 [ 60750580  28041424]]
XTX_inv: [[ 2.96830537e-06 -6.43071025e-06]
 [-6.43071025e-06  1.39675281e-05]]
w : [0.13644777 0.2327492 ]
Sum of all the elements of w : 0.36919696904925203
```