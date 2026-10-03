# 2.1 Car Price Prediction Project

**Chapter 2: Machine Learning for Regression**

- Lesson video: https://www.youtube.com/watch?v=vM3SqPNlStE&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=12
- Slides: https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-21-car-price-prediction-project
- Course notes: https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/02-regression/01-car-price-intro.md
- Dataset (Kaggle): https://www.kaggle.com/CooperUnion/cardataset
- Code and data: https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price

---

## 1. What is this project about?

We build a model that **predicts the price of a car** from its characteristics (make, model, year, engine power, fuel type, transmission, etc.).

Imagine a website where a user lists a car for sale. The site can suggest a fair price automatically. That suggestion comes from our model.

This is a **supervised learning** problem:

- We have historical data where the answer (the price) is already known.
- The model learns the relationship between car characteristics and price.
- Later we use it on new cars where the price is unknown.

Because the thing we predict is a **number** (price in dollars), this is a **regression** problem.

---

## 2. Project plan

| Step | What we do |
|------|-----------|
| 1 | Prepare the data and do Exploratory Data Analysis (EDA) |
| 2 | Use linear regression to predict price |
| 3 | Understand the internals of linear regression |
| 4 | Evaluate the model with RMSE |
| 5 | Feature engineering |
| 6 | Regularization |
| 7 | Use the model to make predictions |

Each step maps to the next lessons of chapter 2.

---

## 3. Key terms explained

**Regression**
A type of supervised learning where the target is a continuous number (price, temperature, salary). Compare with classification, where the target is a category (spam / not spam).

**Linear regression**
A model that predicts the target as a weighted sum of the features. In simple form:

```
price = w0 + w1 * year + w2 * horsepower + w3 * mileage + ...
```

The model learns the weights (`w0`, `w1`, ...) from the data.

**Features (X)**
The input columns used for prediction, such as year, engine HP, number of doors.

**Target (y)**
The value we want to predict. Here it is `MSRP` (Manufacturer's Suggested Retail Price).

**EDA (Exploratory Data Analysis)**
Looking at the data before modelling: checking columns, missing values, distributions, outliers, and relationships. It tells us how to clean the data and what to expect.

**RMSE (Root Mean Squared Error)**
A metric to measure how wrong the model is, in the same units as the target. Formula:

```
RMSE = sqrt( (1/n) * sum( (prediction_i - actual_i)^2 ) )
```

Lower is better. An RMSE of 3000 means predictions are typically off by roughly 3000 dollars.

**Feature engineering**
Creating new, more useful features from existing ones. Example: turn `year` into `age = 2017 - year`.

**Regularization**
A technique that stops the model weights from becoming too large. It reduces overfitting and keeps the model stable, especially when features are correlated.

**Overfitting**
When a model memorizes the training data (including noise) and performs badly on new data.

---

## 4. Setting up the environment

Install the libraries (run once in a terminal):

```
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Start Jupyter:

```
jupyter notebook
```

Create a new notebook and start with the imports:

```python
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

%matplotlib inline
```

What each library does:

- `pandas`: tables (DataFrames), reading CSV files, cleaning data
- `numpy`: fast numerical arrays and math
- `matplotlib` and `seaborn`: plotting
- `%matplotlib inline`: show plots directly inside the notebook

---

## 5. Getting the dataset

Download the CSV file directly:

```
wget https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv
```

If `wget` is not available (for example on Windows), load it straight from the URL in Python:

```python
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
df = pd.read_csv(url)
```

Or, if you downloaded the file next to your notebook:

```python
df = pd.read_csv("data.csv")
```

---

## 6. First look at the data

```python
# Number of rows and columns
df.shape
```

```python
# First 5 rows
df.head()
```

```python
# Column names, data types, and non-null counts
df.info()
```

Expected columns in this dataset include:

- `Make`, `Model`, `Year`
- `Engine Fuel Type`, `Engine HP`, `Engine Cylinders`
- `Transmission Type`, `Driven_Wheels`, `Number of Doors`
- `Market Category`, `Vehicle Size`, `Vehicle Style`
- `highway MPG`, `city mpg`, `Popularity`
- `MSRP` (the target)

---

## 7. Key takeaways

- The goal is to predict car price (`MSRP`), which makes this a **regression** problem.
- The dataset comes from a Kaggle competition and has car characteristics as features.
- The plan: prepare data and EDA, linear regression, how it works internally, RMSE, feature engineering, regularization, and using the model.
- Tools used: `pandas`, `numpy`, `matplotlib`, `seaborn`, and later `scikit-learn`.

---

## 8. Navigation

- Previous: Chapter 1 intro
- Next: 2.2 Data preparation

# 2.2 Data Preparation

**Chapter 2: Machine Learning for Regression**

- Lesson video: https://www.youtube.com/watch?v=Kd74oR4QWGM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=13
- Course notes: https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/02-regression/02-data-preparation.md
- Dataset: https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price

---

## 1. Goal of this lesson

Before any modelling we need clean, consistent data. In this lesson we:

1. Download the car price dataset
2. Load it with pandas
3. Make the **column names** consistent (lowercase, no spaces)
4. Find the **string columns**
5. Make the **string values** consistent (lowercase, no spaces)

Inconsistent names and values cause bugs and make the code harder to write, so this small cleanup pays off for the rest of the chapter.

---

## 2. Terms explained

**DataFrame**
The main pandas table structure: rows and named columns, like a spreadsheet or SQL table in memory.

**Series**
A single column of a DataFrame (a labelled 1-D array).

**Index**
The labels of a Series or DataFrame. `df.columns` is also an Index object, holding the column names.

**dtype (data type)**
The type of values stored in a column, such as `int64`, `float64`, or `object`.

**object dtype**
How pandas stores strings. When a column loaded from a CSV has dtype `object`, it is effectively a text column.

**`.str` accessor**
A pandas namespace that lets you apply string functions (`lower`, `replace`, `strip`, ...) to a whole column at once, without writing a loop.

**Method chaining**
Calling several methods one after another on the result of the previous one, for example `.str.lower().str.replace(...)`.

**Boolean mask**
A True/False series used to filter data. Example: `df.dtypes == 'object'` gives True for every text column.

---

## 3. Getting the data

The dataset originates from Kaggle, but a copy is stored in the `mlbookcamp-code` repository. Download it with `wget`:

```python
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
!wget $data
```

Notes:

- The `!` at the start runs a shell command from inside a Jupyter notebook.
- `$data` passes the Python variable into the shell command.
- The file `data.csv` is saved in the current working directory.
- If `wget` is not available (for example on Windows), skip the download and read directly from the URL: `pd.read_csv(data)`.

---

## 4. Loading the data

```python
import pandas as pd

df = pd.read_csv('data.csv')
```

`read_csv` reads the file and returns a DataFrame. Always look at the first rows right after loading:

```python
df.head()
```

You will see columns such as make, model, year, engine details, and **MSRP** (Manufacturer's Suggested Retail Price). MSRP is our **target**, the value we want to predict.

---

## 5. Making column names consistent

### The problem

The raw column names are inconsistent:

- Some have capital letters: `Make`, `Model`
- Some have spaces: `Engine Fuel Type`
- Some use underscores: `Driven_Wheels`

Spaces are the worst. You cannot use dot notation with them:

```python
# This is a syntax error
df.Transmission Type

# You are forced to use brackets instead
df['Transmission Type']
```

With clean names like `transmission_type`, you can write `df.transmission_type`.

### The fix

`df.columns` is an Index, and an Index supports the `.str` accessor, so we can transform all names at once:

```python
df.columns.str.lower()
```

Chain a second step to replace spaces with underscores:

```python
df.columns.str.lower().str.replace(' ', '_')
```

Result:

```
Index(['make', 'model', 'year', 'engine_fuel_type', 'engine_hp',
       'engine_cylinders', 'transmission_type', 'driven_wheels',
       'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style',
       'highway_mpg', 'city_mpg', 'popularity', 'msrp'],
      dtype='object')
```

### Important: save the result

The expression above only **returns** a new Index. The DataFrame is unchanged until you assign it back:

```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

---

## 6. Finding the string columns

Values have the same inconsistency problem (for example `BMW` vs `bmw`, `MANUAL` vs `manual`). To clean them we first need to know which columns contain text, because string methods fail on numbers.

Check the data types:

```python
df.dtypes
```

Filter to the text columns using a boolean mask:

```python
df.dtypes[df.dtypes == 'object']
```

This returns a Series where the index holds the column names and the values are all `object`. We only need the names, so take the index and convert it to a list:

```python
strings = list(df.dtypes[df.dtypes == 'object'].index)
strings
```

Result:

```
['make',
 'model',
 'engine_fuel_type',
 'transmission_type',
 'driven_wheels',
 'market_category',
 'vehicle_size',
 'vehicle_style']
```

---

## 7. Normalizing the string values

Loop over the string columns and apply the same lowercase and underscore transformation. Assign the result back to the DataFrame:

```python
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```

Check the result:

```python
df.head()
```

Now everything is lowercase and spaces inside values (for example `premium unleaded (required)`) become underscores.

---

## 8. Full code of the lesson in one place

```python
import pandas as pd

# 1. Load
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
df = pd.read_csv(data)

# 2. Clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 3. Find text columns
strings = list(df.dtypes[df.dtypes == 'object'].index)

# 4. Clean text values
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.head()
```

---

## 9. Pandas cheat sheet from this lesson

| Code | What it does |
|------|--------------|
| `pd.read_csv(path)` | Read a CSV file into a DataFrame |
| `df.head()` | Show the first 5 rows |
| `df.columns` | Get the column names |
| `df.columns.str.lower()` | Lowercase all column names |
| `df.columns.str.replace(' ', '_')` | Replace spaces with underscores |
| `df.dtypes` | Get the data type of every column |
| `df.index` | Get the row labels of the DataFrame |

---

## 10. Key takeaways

- Clean names and values early. It prevents annoying bugs later.
- Use lowercase and underscores everywhere (snake_case) so you can use dot notation.
- The `.str` accessor applies string operations to a whole column without loops.
- String operations return a new object. You must assign it back (`df[col] = ...`) to keep the change.
- In pandas, text columns have dtype `object`.
- Selecting columns by dtype is done with a boolean mask on `df.dtypes`.

---

## 11. Navigation

- Previous: 2.1 Car price prediction project
- Next: 2.3 Exploratory data analysis (EDA)
