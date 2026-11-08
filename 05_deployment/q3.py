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