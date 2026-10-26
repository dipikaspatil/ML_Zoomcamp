# Homework 4

This homework uses the pinned 2026 lead-scoring release in the course repository. The plan and report are available in cohorts/2026/data/.

In this homework, we will use the 2026 lead-scoring dataset. Download it from here - https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv

In this dataset our desired target for classification task will be converted variable - has the client signed up to the platform or not.

## Step 0: Setup
```shell
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# verify
python3 -c "import pandas, numpy, seaborn, matplotlib; print('ok')"

# start jupyter notebook
homework.md

# Note connect to zoomcam_venv kernel at jupyter
```

## Data preparation

- Check if the missing values are presented in the features.
- If there are missing values:
    - For categorical features, replace them with 'NA'
    - For numerical features, replace with with 0.0

Split the data into 3 parts with these exact calls:

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)
```

Note: all questions and answers are part of homework.ipynb
