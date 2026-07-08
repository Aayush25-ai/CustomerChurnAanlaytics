import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
PROCESSED_DIR = "models"
TRAIN_TEST_PATH = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
BEST_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model.pkl")
RANDOM_STATE = 42


def load_processed_data():
    """Load the train/test arrays saved by data_preprocessing.py."""
    data = joblib.load(TRAIN_TEST_PATH)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def balance_classes(X_train, y_train):
    """
    Churn datasets are typically imbalanced (far more 'stayed' customers
    than 'churned' ones). If we train on imbalanced data as-is, the
    model can get high accuracy just by predicting "no churn" every
    time -- which is useless for the business.

    SMOTE (Synthetic Minority Over-sampling Technique) creates
    synthetic examples of the minority class (churners) so the model
    sees a balanced 50/50 split during training. This is applied ONLY
    to the training set -- never to the test set, so evaluation still
    reflects real-world class proportions.
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Before SMOTE: {np.bincount(y_train)} | "
          f"After SMOTE: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression: a simple, fast, highly interpretable baseline.
    GridSearchCV tries multiple values of the regularization strength
    'C' and picks the one with the best cross-validated ROC-AUC.
    """
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid,
        scoring="roc_auc",
        cv=5,
    )
    grid.fit(X_train, y_train)
    print(f"[INFO] LogisticRegression best params: {grid.best_params_}, "
          f"CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    """
    Random Forest: an ensemble of decision trees. Usually captures
    non-linear relationships and feature interactions better than
    Logistic Regression, at some cost to interpretability.
    """
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"[INFO] RandomForest best params: {grid.best_params_}, "
          f"CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgboost(X_train, y_train):
    """
    XGBoost: gradient-boosted trees, often the strongest performer on
    tabular data like this. Slightly slower to tune but usually worth it.
    """
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    }
    grid = GridSearchCV(
        XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"[INFO] XGBoost best params: {grid.best_params_}, "
          f"CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    # Balance the training set only (never touch the test set).
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    # Train each candidate model.
    models = {
        "LogisticRegression": train_logistic_regression(X_train_bal, y_train_bal),
        "RandomForest": train_random_forest(X_train_bal, y_train_bal),
        "XGBoost": train_xgboost(X_train_bal, y_train_bal),
    }

    # Evaluate every model on the untouched, real-world-proportioned
    # test set, and keep whichever scores highest on ROC-AUC.
    # ROC-AUC is a good metric for imbalanced classification because it
    # measures ranking quality across all classification thresholds,
    # not just accuracy at a single (often misleading) threshold.
    best_name, best_model, best_score = None, None, -1
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_proba)
        print(f"[RESULT] {name} Test ROC-AUC: {score:.4f}")
        if score > best_score:
            best_name, best_model, best_score = name, model, score

    print(f"\n[BEST MODEL] {best_name} with Test ROC-AUC = {best_score:.4f}")

    # Persist the winning model for later use in evaluate.py / predict.py
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name}, BEST_MODEL_PATH)
    print(f"[INFO] Saved best model -> {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
