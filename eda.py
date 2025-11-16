# src/eda.py

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from config import FIG_DIR

def run_eda():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print("=== EDA ===")
    print(X.head())
    print("\nShape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())

    # Histogram
    plt.figure(figsize=(6,4))
    X["mean radius"].hist(bins=30)
    plt.title("Histogram: Mean Radius")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hist_mean_radius.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(6,4))
    X.boxplot(column=["mean radius", "mean texture"], rot=45)
    plt.title("Boxplot: Mean Features")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/boxplot_radius_texture.png")
    plt.close()

    print(f"Saved EDA figures to {FIG_DIR}/")
    

if __name__ == "__main__":
    run_eda()
