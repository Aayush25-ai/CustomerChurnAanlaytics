"""
predict.py
-----------
STEP 4 of the pipeline.

Purpose:
    Load the saved preprocessor + trained model, and use them to predict
    churn for brand-new/unseen customer data (a single customer dict,
    or a whole CSV of new customers).

Why reuse the saved preprocessor instead of writing new encoding logic?
    The model was trained on data transformed a specific way (same
    scaling parameters, same one-hot categories). If new data isn't
    transformed identically, predictions will be wrong or the code
    will crash on unseen categories. Reusing the exact fitted
    preprocessor guarantees consistency between training and inference.

Run this file directly for a demo prediction on one example customer:
    python src/predict.py
"""

import os
import joblib
import pandas as pd

PROCESSED_DIR = "models"
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.pkl")
BEST_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model.pkl")


def load_artifacts():
    """Load the fitted preprocessor and trained model from disk."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    saved = joblib.load(BEST_MODEL_PATH)
    return preprocessor, saved["model"], saved["model_name"]


def predict_single(customer: dict, preprocessor, model) -> dict:
    """
    Predict churn for a single customer.

    Args:
        customer: dict of raw feature values, e.g.
            {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35, "TotalCharges": 350.5
            }
        preprocessor: the fitted ColumnTransformer from data_preprocessing.py
        model: the trained classifier from train_model.py

    Returns:
        dict with churn_prediction (Yes/No) and churn_probability (float)
    """
    # Wrap the single dict in a DataFrame so it matches the shape the
    # preprocessor expects (it was fit on a DataFrame of many rows).
    df = pd.DataFrame([customer])

    X_transformed = preprocessor.transform(df)

    probability = model.predict_proba(X_transformed)[0, 1]
    prediction = "Yes" if probability >= 0.5 else "No"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(probability), 4),
    }


def predict_batch(csv_path: str, preprocessor, model) -> pd.DataFrame:
    """
    Predict churn for a batch of new customers stored in a CSV file
    (same columns as the training data, minus the 'Churn' target column).
    Returns the original DataFrame with two new columns appended:
    'churn_prediction' and 'churn_probability'.
    """
    df = pd.read_csv(csv_path)

    # Defensive check: drop customerID / Churn if present, since the
    # preprocessor was never fit on those columns as features.
    df_features = df.drop(columns=["customerID", "Churn"], errors="ignore")

    X_transformed = preprocessor.transform(df_features)
    probabilities = model.predict_proba(X_transformed)[:, 1]

    df["churn_probability"] = probabilities.round(4)
    df["churn_prediction"] = df["churn_probability"].apply(lambda p: "Yes" if p >= 0.5 else "No")
    return df


if __name__ == "__main__":
    preprocessor, model, model_name = load_artifacts()
    print(f"[INFO] Loaded model: {model_name}")

    # Example customer -- edit these values to test different profiles
    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.5,
    }

    result = predict_single(example_customer, preprocessor, model)
    print(f"[PREDICTION] Churn: {result['churn_prediction']} "
          f"(probability = {result['churn_probability']})")
