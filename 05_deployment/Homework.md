# Homework 5: Deploying Machine Learning Models

This homework uses a frozen lead-scoring release for the 2026 cohort. The artifact, dataset checksum, training script, API, and container definition are in this directory. No prior-cohort model or generated answer set is used.

The model was trained from course_lead_scoring_2026.csv (https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/course_lead_scoring_2026.csv) with the following observable features:

| **Feature**                | **Type**    | **Missing-value rule** |
|----------------------------|-------------|------------------------|
| `lead_source`              | categorical | `NA`                   |
| `industry`                 | categorical | `NA`                   |
| `employment_status`        | categorical | `NA`                   |
| `location`                 | categorical | `NA`                   |
| `number_of_courses_viewed` | integer     | Training-set median    |
| `annual_income`            | number      | Training-set median    |
| `interaction_count`        | integer     | Training-set median    |
| `lead_score`               | number      | Training-set median    |

We use a `DictVectorizer` followed by `LogisticRegression(solver="liblinear", C=1.0)`. The training script records the data checksum and imputation values in `model_metadata.json`. `lead_score` is available before conversion and `converted` is the target.

## Setup

Files copied from cohorts/2026/homework/05-deployment in the course repo:

pipeline.bin, model_metadata.json, predict.py, Dockerfile, uv.lock, pyproject.toml, smoke_test.py, q6_test.py, feature_defaults.json, model.py

```shell
uv --version
ls
uv sync --locked
uv run python smoke_test.py
# Output
{'conversion_probability': 0.769799, 'conversion': True}

sha256sum pipeline.bin
# Output
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin
```

## Question 1 — environment check
Run `uv --version` and record the version in your notes. This is a setup check, not a graded answer: your local uv version is not a property of the model.

```shell
05_deployment % uv --version
uv 0.11.7 (9d177269e 2026-04-15 aarch64-apple-darwin)
```

## Question 2 — locked dependency
Open pyproject.toml and uv.lock. Which Scikit-Learn version is part of the 2026 reference environment?

- 1.6.1
- 1.7.2 <-- answer
- 1.8.0
- 2.0.0

Do not regenerate the lockfile for the graded run; use uv sync --locked.

## Question 3 — load the model
Write a Python script that loads pipeline.bin with pickle and calls predict_proba for this lead:

```shell
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```
Report the probability of conversion rounded to three decimal places. The grader accepts an absolute error of at most 0.005; do not choose the nearest answer from a list.

```python
# --- Imports ---------------------------------------------------------------
import pickle                          # Python's built-in tool for saving and loading objects

from model import normalize_record     # helper from the homework's model.py:
                                       # fills missing values and fixes number types


# --- 1. Load the saved model ----------------------------------------------
# pipeline.bin holds ONE object: a scikit-learn Pipeline made of
#   step 1: DictVectorizer      (turns a dict into a numeric feature row)
#   step 2: LogisticRegression  (turns that row into a probability)
# Because both are inside one object, we do not need a separate "dv" like in the lessons.
with open("pipeline.bin", "rb") as f_in:    # "rb" = read binary (pickle files are not text)
    pipeline = pickle.load(f_in)            # unpickle: rebuild the object from the file
                                            # (only load files you trust: unpickling can run code)

# Sanity checks: what did we actually load?
print(type(pipeline))                       # expect: sklearn.pipeline.Pipeline
print([name for name, _ in pipeline.steps]) # names of the steps, in order


# --- 2. Check which categories the model knows ----------------------------
# DictVectorizer turns each text value into its own 0/1 column, named "feature=value",
# for example "lead_source=paid_ads".
# If we spell a value differently from training, NO column matches, scikit-learn does
# not complain, and the prediction is silently wrong. So we look at the real names.
dv = pipeline.steps[0][1]                   # steps is a list of (name, object) pairs;
                                            # [0] = first step, [1] = the object itself
cat_prefixes = ("lead_source", "industry", "employment_status", "location")
print([n for n in dv.get_feature_names_out()          # all column names the model uses
       if n.startswith(cat_prefixes)])               # keep only the text-category ones


# --- 3. The lead from the question ----------------------------------------
# Keys must match the model's feature names exactly.
# Text values use lowercase with underscores, as in the training data.
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}


# --- 4. Apply the same cleaning the real service uses ----------------------
# normalize_record: empty text becomes "NA", empty numbers become the training default,
# and counts become int / other numbers become float.
# Our lead has no missing values, so it should come back unchanged.
record = normalize_record(lead)
print(record)


# --- 5. Predict ------------------------------------------------------------
# predict_proba expects a LIST of records, so we wrap our one dict in [ ].
# It returns a table with two columns: [P(no conversion), P(conversion)].
#   [0, 1] = row 0 (our only lead), column 1 (probability of converting)
proba = pipeline.predict_proba([record])[0, 1]
print(proba)                                # full precision
print(round(float(proba), 3))               # 3 decimals, as the homework asks
                                            # float() first avoids the NumPy rounding quirk
```

Run - `uv run python q3.py`

```shell
# Output
05_deployment % uv run python q3.py
<class 'sklearn.pipeline.Pipeline'>
['dictvectorizer', 'logisticregression']
['employment_status=NA', 'employment_status=employed', 'employment_status=self_employed', 'employment_status=student', 'employment_status=unemployed', 'industry=NA', 'industry=education', 'industry=finance', 'industry=healthcare', 'industry=manufacturing', 'industry=retail', 'industry=technology', 'lead_source=NA', 'lead_source=events', 'lead_source=organic_search', 'lead_source=paid_ads', 'lead_source=referral', 'lead_source=social_media', 'location=NA', 'location=africa', 'location=asia', 'location=europe', 'location=north_america', 'location=south_america']
{'lead_source': 'paid_ads', 'industry': 'technology', 'employment_status': 'employed', 'location': 'north_america', 'number_of_courses_viewed': 2, 'annual_income': 79276.0, 'interaction_count': 4, 'lead_score': 0.41}
0.5329944186432507
0.533 # <-- answer
```

## Question 4 — serve the model
Start the provided API:

```shell
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

The API has GET /health and POST /predict. Send this second lead:

```shell
{
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}
```

For example:

```shell
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
```

Report conversion_probability rounded to three decimal places. The same 0.005 absolute tolerance applies. The response must contain a probability in [0, 1] and a boolean conversion field.

### Terminal 1: start the service and leave it running:
```shell
uv run uvicorn predict:app --port 9696 # we can see Uvicorn running on http://127.0.0.1:9696.

# Output
05_deployment % uv run uvicorn predict:app --port 9696
INFO:     Started server process [70527]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:9696 (Press CTRL+C to quit)
```

### Terminal 2: open a second terminal tab in the same folder. First check the health endpoint:
```shell
curl -s http://localhost:9696/health

# Output
05_deployment % curl -s http://localhost:9696/health
{"status":"ok","model_sha256":"1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1","feature_defaults_sha256":"1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"}%
```

Then send the lead. The values use the same spelling as the model’s categories above:

```shell
curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_source": "organic_search", "industry": "technology", "employment_status": "employed", "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0, "interaction_count": 7, "lead_score": 0.74}'

# Output
05_deployment % curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_source": "organic_search", "industry": "technology", "employment_status": "employed", "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0, "interaction_count": 7, "lead_score": 0.74}'
{"conversion_probability":0.769799,"conversion":true}%
```
----------------------------------------------------------------------------------------------------

### Container
The checked-in Dockerfile is the canonical 2026 container configuration. It uses Python 3.11.15-slim-bookworm, uv 0.10.11, and the checked-in uv.lock. It copies the frozen artifact and starts the API on port 9696.

Build and run it:

docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5

---------------------------------------------------------------------------------------------------

## Question 5 — look at the container configuration

Which Python base-image tag is declared in the Dockerfile?

- python:3.9-slim-bullseye
- python:3.11.15-slim-bookworm <-- answer
- python:3.12-slim-bookworm
- python:3.13.10-slim-bookworm
This question checks a version-controlled declaration. The displayed size of a local Docker image is not graded because it varies by platform, architecture, Docker version, and cached layers.

## Question 6 — run the container
Run the same request from Question 4 against the container. You can use the provided smoke client:

python q6_test.py
Report conversion_probability rounded to three decimal places. The expected result is the same reference inference as Question 4, with the same 0.005 absolute tolerance. Also verify that:

curl -s http://localhost:9696/health
returns {"status":"ok", ...} and the model checksum recorded in model_metadata.json.

### steps

```shell
printf "3.11.15\n" > .python-version

docker build -t zoomcamp-model:2026-hw5 .

# Build and run. Docker Desktop must be open, and the local uvicorn from Q4 must be stopped so port 9696 is free:
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5

# Output
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9696 (Press CTRL+C to quit)

# In a second terminal, use the same two checks as before:

curl -s http://localhost:9696/health

curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_source": "organic_search", "industry": "technology", "employment_status": "employed", "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0, "interaction_count": 7, "lead_score": 0.74}'


# Output 
05_deployment % curl -s http://localhost:9696/health
{"status":"ok","model_sha256":"1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1","feature_defaults_sha256":"1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"}%

05_deployment % curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_source": "organic_search", "industry": "technology", "employment_status": "employed", "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0, "interaction_count": 7, "lead_score": 0.74}'
{"conversion_probability":0.769799,"conversion":true}%
```
