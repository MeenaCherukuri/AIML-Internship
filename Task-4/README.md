# 🧠 Logistic Regression Classifier – Breast Cancer Prediction

A machine learning project to classify whether a tumor is **malignant or benign** using **Logistic Regression**.

---

## 📌 Objective

Build a **binary classifier** using **Logistic Regression** to predict breast cancer diagnosis and evaluate it using multiple metrics.

---

## 🗂️ Dataset

- **Name**: Breast Cancer Wisconsin Diagnostic Dataset  
- **Source**: Scikit-learn built-in datasets  
- **Target**: `0` = Malignant, `1` = Benign  
- **Features**: 30 numerical attributes such as radius, texture, perimeter, area, etc.

---

## ⚙️ Tools Used

- `Python`
- `Pandas`, `NumPy` – Data manipulation
- `Scikit-learn` – Model building and evaluation
- `Matplotlib`, `Seaborn` – Visualization

---

## 🛠️ Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/logistic-regression-breast-cancer.git
   cd logistic-regression-breast-cancer
2. Create and activate a virtual environment (optional):
    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

🚀 Usage:

Run the classification pipeline:
python logistic_regression_classifier.py

📊 Visuals

🔹 Confusion Matrix

Shows how many predictions are true/false positives/negatives.

🔹 ROC Curve

Illustrates the model's diagnostic ability across different thresholds.
https://github.com/MeenaCherukuri/AIML-Internship/blob/3d2266701266b4b3eabd16115a03fde395bfa0e0/Task-4/roc_curve.png
## ✅ Evaluation Metrics

| **Metric**       | **Value (Example)** |
|------------------|---------------------|
| **Accuracy**     | 96.49%              |
| **Precision**    | 96%                 |
| **Recall**       | 97%                 |
| **ROC-AUC Score**| 0.99                |

> 🔧 You can tune the classification **threshold** (e.g., `0.3`) to adjust the **trade-off between precision and recall** depending on the problem domain (e.g., prioritizing recall in medical diagnosis).

---

## 📐 Sigmoid Function

The **sigmoid function** is used in logistic regression to convert a linear combination of features into a probability:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- The output of the sigmoid function lies between **0 and 1**.
- It allows logistic regression to model **probabilities** of class membership.
## 📎 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to use, modify, and distribute for personal or commercial purposes.

---

## 🙌 Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) – For providing robust machine learning tools and datasets.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) – For making the Breast Cancer dataset publicly available.






