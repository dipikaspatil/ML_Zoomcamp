# Homework 3

This homework uses the pinned 2026 lead-scoring release in the course repository. The plan and report are available in cohorts/2026/data/.

In this homework, we will use the 2026 lead-scoring dataset.

```shell
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv
```

In this dataset our desired target for classification task will be converted variable - has the client signed up to the platform or not.

## Step 0: Setup

```python
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# verify
python3 -c "import pandas, numpy, seaborn, matplotlib; print('ok')"

# start jupyter notebook
jupyter notebook

# Note connect to zoomcam_venv kernel at jupyter

```

## Data preparation
- Check if the missing values are presented in the features.
- If there are missing values:
    - For categorical features, replace them with 'NA'
    - For numerical features, replace with with 0.0


```python
# missing values
df.isnull().sum()

# Output 
lead_source                 148
industry                    240
employment_status           194
location                    208
annual_income               369
number_of_courses_viewed      0
interaction_count             0
lead_score                   35
converted                     0
dtype: int64

# split the columns by type
categorical = list(df.select_dtypes(include=['object', 'string']).columns)
numerical = [c for c in df.select_dtypes(include='number').columns if c != 'converted']

print("categorical:", categorical)
print("numerical:  ", numerical)

# Output
categorical: ['lead_source', 'industry', 'employment_status', 'location']
numerical:   ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

# fill missing values
df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)

df.isnull().sum().sum()

# Output
np.int64(0)
```

Note: all questions and answers are part of homework.ipynb