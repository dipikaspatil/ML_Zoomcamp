# Homework 2

This homework uses the pinned 2026 car fuel-efficiency release in the course repository. The plan and report are available in cohorts/2026/data/.

## Dataset
For this homework, we'll use the 2026 Car Fuel Efficiency dataset. Download it from here.

You can do it with wget:
```shell
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

The goal of this homework is to create a regression model for predicting the car fuel efficiency (column 'fuel_efficiency_mpg').

## Preparing the dataset
Use only the following columns:

- engine_displacement,
- horsepower,
- vehicle_weight,
- model_year,
- fuel_efficiency_mpg

## EDA
Look at the fuel_efficiency_mpg variable. Does it have a long tail?

- No there isn o long tail. So for homework we don't need ot use - np.log1p

## Step 0: Setup

```python
python3 -m venv venv
source venv/bin/activate

# verify
python3 -c "import pandas, numpy, seaborn, matplotlib; print('ok')"

# start jupyter notebook
jupyter notebook

```

## Question 1
There's one column with missing values. What is it?

- 'engine_displacement'
- 'horsepower' <-- answer
- 'vehicle_weight'
- 'model_year'

```python
df.isnull().sum()

# Output
engine_displacement      0
horsepower             877
vehicle_weight           0
model_year               0
fuel_efficiency_mpg      0
dtype: int64
```


## Question 2
Question 2
What's the median (50% percentile) for variable 'horsepower'?

- 204
- 254 <-- answer
- 304
- 354

```python
df.horsepower.median()

# Output
np.float64(254.0)
```

## Prepare and split the dataset
Shuffle the filtered dataset and create the split exactly as in the lecture:

```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

n, n_train, n_val, n_test

# Output
(10000, 6000, 2000, 2000)
```

## Question 3
We need to deal with missing values for the column from Q1.
We have two options: fill it with 0 or with the mean of this variable.
Try both options. For each, train a linear regression model without regularization using the code from the lessons.
For computing the mean, use the training only!
Use the validation dataset to evaluate the models and compare the RMSE of each option.
Round the RMSE scores to 3 decimal digits using round(score, 3). This keeps the imputation difference visible in this release.
Which option gives better RMSE?

Options:

- With mean <-- answer
- With 0
- Both are equally good

```python
# Output
{'zero': np.float64(2.205), 'mean': np.float64(2.202)}
Better: with mean
```

## Question 4
Now let's train a regularized linear regression.
For this question, fill the NAs with 0.
Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
Use RMSE to evaluate the model on the validation dataset.
Round the RMSE scores to 4 decimal digits. This keeps the small but real regularization differences visible instead of turning several choices into a tie.
Which r gives the best RMSE?
If multiple options give the same best RMSE, select the smallest r.

Options:

- 0 <-- answer
- 0.01
- 0.1
- 1
- 5
- 10
- 100

```python
#Output
{0: np.float64(2.2053),
 0.01: np.float64(2.2058),
 0.1: np.float64(2.2241),
 1: np.float64(2.3492),
 5: np.float64(2.4094),
 10: np.float64(2.4195),
 100: np.float64(2.4292)}
```

## Question 5
We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
For each seed, do the train/validation/test split with 60%/20%/20% distribution.
Fill the missing values with 0 and train a model without regularization.
For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
Round the result to 3 decimal digits (round(std, 3))
What's the value of std?

- 0.006
- 0.016
- 0.029 <-- answer
- 0.036

Note: Standard deviation shows how different the values are. If it's low, then all values are approximately the same. If it's high, the values are different. If standard deviation of scores is low, then our model is stable.

```python
# Output
0 2.2392
1 2.2021
2 2.1634
3 2.2044
4 2.2019
5 2.2458
6 2.2697
7 2.1908
8 2.2166
9 2.207

std: 0.029
```

## Question 6
Split the dataset like previously, use seed 9.
Combine train and validation datasets.
Fill the missing values with 0 and train a model with r=0.001.
What's the RMSE on the test dataset?
Options:

- 0.236
- 2.236 <-- answer
- 22.10
- 221.0

```python
# Output
np.float64(2.236)
```