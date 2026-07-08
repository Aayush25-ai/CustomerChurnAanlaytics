"""
data_preprocessing.py
----------------------
STEP 1 of the pipeline.

Purpose:
    Load the raw Telco Customer Churn CSV, clean it, engineer/encode
    features, split into train/test sets, and save a reusable
    preprocessing pipeline (so the exact same transformations can later
    be applied to new/unseen customer data during prediction).

Why a separate preprocessing script?
    Keeping preprocessing separate from model training means:
      1. You can re-run/debug cleaning steps without retraining a model.
      2. The exact transformations get saved and reused at prediction
         time -> no "training/serving skew" (a very common real-world bug).

Run this file directly:
    python src/data_preprocessing.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------------------
# CONFIG — centralised paths & constants so nothing is hard-coded twice
# ---------------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "telco_churn.csv")
PROCESSED_DIR = "models"
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.pkl")
TRAIN_TEST_PATH = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42          # fixed seed -> reproducible results every run
TEST_SIZE = 0.2            # 80% train / 20% test split


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Kaggle (see data/README.md) and place it there."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known data-quality issues specific to this dataset:
      - 'TotalCharges' is read as text because a few rows contain blank
        strings instead of numbers (usually for brand-new customers with
        tenure = 0). We convert it to numeric and fill missing values.
      - Drop 'customerID' since it's just a unique identifier with no
        predictive value (keeping it would let the model "memorize" IDs).
    """
    df = df.copy()

    # customerID has zero predictive value -> drop it
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # TotalCharges sometimes contains blank strings -> coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Rows where TotalCharges became NaN are almost always tenure == 0
    # customers (they haven't been billed yet) -> fill with 0
    df["TotalCharges"].fillna(0, inplace=True)

    # Standardise the target column to binary integers: Yes -> 1, No -> 0
    # (Models need numeric targets, and this also makes metrics like
    # ROC-AUC straightforward to compute.)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    print("[INFO] Cleaned data: fixed TotalCharges, dropped customerID, "
          "encoded target as 0/1")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate the DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Scales numeric columns (mean=0, std=1) so models like Logistic
        Regression aren't biased by features on different scales
        (e.g. tenure in months vs TotalCharges in hundreds/thousands).
      - One-hot encodes categorical columns (turns e.g. 'Contract' with
        3 categories into 3 binary columns) so models can use them
        mathematically.

    Using ColumnTransformer + Pipeline (instead of manual pd.get_dummies)
    means this exact transformation can be saved and replayed later on
    brand-new customer data without re-deriving column definitions.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"[INFO] Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load
    df = load_data(RAW_DATA_PATH)

    # 2. Clean
    df = clean_data(df)

    # 3. Split into features (X) and target (y) BEFORE any scaling/encoding
    #    to avoid leaking target information into the transformations.
    X, y = split_features_target(df)

    # 4. Train/test split.
    #    stratify=y ensures both splits keep the same churn/no-churn ratio
    #    as the original data -- important because churn datasets are
    #    usually imbalanced (~27% churn here).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 5. Build and FIT the preprocessor on TRAINING data only.
    #    Fitting on test data too would leak information from the test
    #    set into training (a subtle but serious mistake).
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 6. Save the fitted preprocessor so predict.py can reuse the exact
    #    same scaling/encoding on brand-new customer data.
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"[INFO] Saved fitted preprocessor -> {PREPROCESSOR_PATH}")

    # 7. Save the processed train/test arrays so train_model.py doesn't
    #    need to redo this work every time.
    joblib.dump(
        {
            "X_train": X_train_processed,
            "X_test": X_test_processed,
            "y_train": y_train.values,
            "y_test": y_test.values,
        },
        TRAIN_TEST_PATH,
    )
    print(f"[INFO] Saved processed train/test data -> {TRAIN_TEST_PATH}")


if __name__ == "__main__":
    main()
