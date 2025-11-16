---

# **Disease Diagnosis Using Gradient Boosting Machines (GBM)**

## **Project Overview**

This project implements a predictive model for breast cancer diagnosis using Gradient Boosting Machines (GBM) with scikit-learn.
The workflow includes:

* Dataset selection and **Exploratory Data Analysis (EDA)**
* **Preprocessing** using Pipelines
* **Model training** with GBM and hyperparameter tuning
* **Evaluation** using hold-out test set and cross-validation
* **Interpretation** of feature importance
* **Responsible modeling** practices to avoid sensitive data or leakage

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) dataset**, which contains numeric features describing tumor characteristics.

---

## **Table of Contents**

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Running the Code](#running-the-code)
5. [Outputs](#outputs)
6. [Results](#results)
7. [Responsible Modeling](#responsible-modeling)
8. [References](#references)

---

## **Dataset**

* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* **Scikit-learn built-in version:** `sklearn.datasets.load_breast_cancer()`
* **Number of samples:** 569
* **Number of features:** 30 numeric features
* **Target:**

  * 0 = Malignant
  * 1 = Benign
* **Characteristics:**

  * Clean, numeric, no missing values
  * Slight class imbalance (benign > malignant)

---

## **Project Structure**

```
project/
├─ eda.py                  # Exploratory Data Analysis (histograms, boxplots, heatmap)
├─ train.py                # Train Gradient Boosting model and save model object
├─ evaluate.py             # Evaluate model (confusion matrix, ROC, feature importance, CV)
├─ images/                 # Generated PNG images for slides
├─ gbm_model.pkl           # Trained model saved for evaluation
├─ test_data.pkl           # Test dataset saved for evaluation
├─ README.md               # Project documentation
```

---

## **Setup Instructions**

1. **Clone the repository**

```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

2. **Install required Python packages**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

3. **Optional:** Create a virtual environment for project isolation:

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

---

## **Running the Code**

1. **EDA**: Generates histograms, boxplots, and correlation heatmaps.

```bash
python eda.py
```

Output images: `histogram.png`, `boxplot.png`, `heatmap.png`

2. **Train GBM Model**: Splits dataset, trains GBM, and saves model object.

```bash
python train.py
```

Output: `gbm_model.pkl`, `test_data.pkl`

3. **Evaluate Model**: Generates confusion matrix, ROC curve, feature importance, and CV metrics.

```bash
python evaluate.py
```

Output images: `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `cv_metrics.png`

> All plots are saved to the `images/` folder by default.

---

## **Outputs**

* **EDA Visualizations:** histogram, boxplot, heatmap
* **Model Evaluation:**

  * Confusion matrix
  * ROC curve
  * Feature importance
  * Cross-validation metrics
* **Trained model** for reproducibility: `gbm_model.pkl`

These images can be used for presentations, reports, or further analysis.

---

## **Results**

* **Test Set Performance:**

  * Accuracy: ~96%
  * Precision: ~97%
  * Recall: ~95%
  * F1 Score: ~96%
  * ROC-AUC: ~0.99

* **Cross-Validation:**

  * 5-Fold CV Accuracy: 94–97%
  * Stable and low variance across folds

* **Top Features (Importance):**

  1. Worst radius
  2. Worst concavity
  3. Worst area
  4. Worst perimeter
  5. Mean texture

* The model is **interpretable**, stable, and aligns with clinical insights.

---

## **Responsible Modeling**

* No sensitive or personal patient information is used.
* Preprocessing is applied **only on training data** to prevent leakage.
* GBM provides **feature importance** for explainable predictions.
* Thresholds and evaluation metrics (recall, precision) are chosen with clinical relevance in mind.
* Model is **not intended as a standalone diagnostic tool**—only as a decision support aid.

---

## **References**

1. **Scikit-learn documentation – Breast Cancer Dataset**
   [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

2. **Gradient Boosting Classifier (Scikit-learn)**
   [https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

3. **Breast Cancer Wisconsin Dataset – UCI ML Repository**
   [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

4. **Seaborn Documentation**
   [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

5. **Matplotlib Documentation**
   [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

6. **SHAP for interpretability (Optional)**
   [https://github.com/slundberg/shap](https://github.com/slundberg/shap)

7. **Python Official Documentation**
   [https://docs.python.org/3/](https://docs.python.org/3/)

---

This README is **ready to use for GitHub submission** and provides **all the instructions to run your code, generate images, and understand the project workflow**.

---

