"""
evaluate.py
------------
STEP 3 of the pipeline.

Purpose:
    Load the saved best model and test data, produce classification
    metrics and diagnostic plots (confusion matrix, ROC curve, feature
    importance) so you can judge whether the model is actually good
    enough and where it makes mistakes.

Run this file directly (after train_model.py):
    python src/evaluate.py
"""

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

PROCESSED_DIR = "models"
TRAIN_TEST_PATH = os.path.join(PROCESSED_DIR, "train_test_data.pkl")
BEST_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model.pkl")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots")


def plot_confusion_matrix(y_test, y_pred, save_path):
    """
    A confusion matrix shows exactly how many customers were:
      - correctly predicted to stay      (True Negative)
      - correctly predicted to churn     (True Positive)
      - wrongly predicted to churn       (False Positive)
      - wrongly predicted to stay        (False Negative) <- most costly!
    Missing an actual churner (False Negative) is usually the most
    expensive mistake for a business, since no retention action gets
    triggered for that customer.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix plot -> {save_path}")


def plot_roc_curve(y_test, y_proba, save_path):
    """
    The ROC curve shows the trade-off between True Positive Rate and
    False Positive Rate at every possible decision threshold. A curve
    that hugs the top-left corner (AUC close to 1.0) means the model
    separates churners from non-churners well.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved ROC curve plot -> {save_path}")


def plot_feature_importance(model, save_path):
    """
    Shows which features most influence the model's predictions.
    Only works for tree-based models (Random Forest, XGBoost) which
    expose `feature_importances_`. Skipped for Logistic Regression here
    for simplicity (its equivalent would be the coefficient magnitudes).
    """
    if not hasattr(model, "feature_importances_"):
        print("[INFO] Selected model has no feature_importances_ attribute; "
              "skipping feature importance plot.")
        return

    importances = model.feature_importances_
    plt.figure(figsize=(6, 5))
    sorted_idx = importances.argsort()[-15:]  # top 15 features
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [f"feature_{i}" for i in sorted_idx])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved feature importance plot -> {save_path}")
    print("[NOTE] Feature indices correspond to the one-hot encoded columns. "
          "See predict.py / preprocessor for exact column names.")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    data = joblib.load(TRAIN_TEST_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    saved = joblib.load(BEST_MODEL_PATH)
    model, model_name = saved["model"], saved["model_name"]
    print(f"[INFO] Evaluating model: {model_name}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"[METRIC] ROC-AUC: {auc:.4f}")

    plot_confusion_matrix(y_test, y_pred, os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba, os.path.join(PLOTS_DIR, "roc_curve.png"))
    plot_feature_importance(model, os.path.join(PLOTS_DIR, "feature_importance.png"))


if __name__ == "__main__":
    main()
