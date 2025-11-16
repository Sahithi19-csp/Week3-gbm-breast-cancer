# src/evaluate.py

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

from config import FIG_DIR, RESULT_DIR, MODEL_DIR
import pandas as pd

def evaluate_model():
    model = joblib.load(f"{MODEL_DIR}/gbm_model.pkl")
    X_test, y_test = joblib.load(f"{MODEL_DIR}/test_split.pkl")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\n=== Test Set Performance ===")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc)

    # Save summary
    summary = pd.DataFrame([{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc
    }])
    summary.to_csv(f"{RESULT_DIR}/metrics.csv", index=False)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix - Test Set")
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png")
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Test Set")
    plt.savefig(f"{FIG_DIR}/roc_curve.png")
    plt.close()

    # Feature Importances
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    feat_names = X_test.columns
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    fi.to_csv(f"{RESULT_DIR}/feature_importances.csv")

    print(f"\nSaved results to {RESULT_DIR} and plots to {FIG_DIR}")

if __name__ == "__main__":
    evaluate_model()
