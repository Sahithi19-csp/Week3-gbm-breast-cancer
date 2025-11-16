# src/train.py

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

from config import MODEL_DIR, RANDOM_STATE, TEST_SIZE

def train_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Preprocessing
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
        ]
    )

    # Model
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", gbm)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # CV evaluation on training set
    print("\nRunning 5-fold Stratified CV...")
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring)
    for metric in scoring:
        print(f"{metric}: {cv_scores[f'test_{metric}'].mean():.4f}")

    # Fit final model
    print("\nTraining final model...")
    pipeline.fit(X_train, y_train)

    # Save model + split
    joblib.dump(pipeline, f"{MODEL_DIR}/gbm_model.pkl")
    joblib.dump((X_test, y_test), f"{MODEL_DIR}/test_split.pkl")

    print(f"Model saved to {MODEL_DIR}/gbm_model.pkl")

if __name__ == "__main__":
    train_model()
