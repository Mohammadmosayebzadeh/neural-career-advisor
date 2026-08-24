"""
predict.py
==========
Minimal example: load the trained salary model and predict a salary.

Requirements in the same folder:
    salary_predictor.pkl   the full pipeline (preprocessing + model)
    salary_utils.py        the custom transformers the pipeline references

No preprocessing is needed here: encoding, imputation and the YearsCode
parsing all happen inside the loaded pipeline.
"""

import joblib
import pandas as pd

import salary_utils  # noqa: F401  (must be importable so the pickle can resolve)

# 1) Load the trained pipeline
pipeline = joblib.load("salary_predictor.pkl")

# 2) Build a RAW DataFrame - exactly the survey format, no cleaning
sample = pd.DataFrame([{
    "Country": "Germany",
    "YearsCode": "10",                                # "Less than 1 year" also works
    "DevType": "Developer, full-stack",
    "LanguageHaveWorkedWith": "Python;SQL;JavaScript",
    "Year": 2026,
}])

# 3) Predict
prediction = pipeline.predict(sample)[0]
print(f"Predicted annual salary: {prediction:,.0f} USD")
