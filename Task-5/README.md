# 🧠 Task 5: Decision Trees and Random Forests

## 📋 Objective

Learn how to use **tree-based models** for **classification and regression**, specifically:

- Train and visualize a **Decision Tree Classifier**
- Analyze **overfitting** and control tree depth
- Train a **Random Forest** and compare performance
- Interpret **feature importances**
- Evaluate models using **cross-validation**

---

## 🛠 Tools & Libraries

- Python 3.x
- [Scikit-learn](https://scikit-learn.org/)
- [Graphviz](https://graphviz.org/) (for visualization)
- Pandas, Matplotlib, NumPy

---  

## 📦 Dataset

We use the **Heart Disease Dataset**, a popular binary classification dataset.

- Target column: `target` (1 = presence of heart disease, 0 = absence)
- [Click here to download dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
  *(Alternative: Load from Kaggle or other public sources)*
   [Kaggle dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 🚀 Getting Started

### 1. Clone the Repository
    ```bash
    git clone https://github.com/your-username/tree-models-task5.git
    cd tree-models-task5

### 2. Install Requirements
    ```bash
    pip install -r requirements.txt
### 3. Run the Notebook
 Open the Jupyter notebook:
    ```bash
    jupyter notebook task5_decision_tree_random_forest.ipynb
 Or run the Python script (if applicable):
 
    ```bash
    python task5_decision_tree_random_forest.py
## 🧪 Experiments & Results

### ✅ Decision Tree

- Model trained using `DecisionTreeClassifier`
- Tree visualized using **Graphviz**
- Controlled for overfitting using `max_depth`
- Optimal depth typically falls between **3–6**

---

### 🌲 Random Forest

- Trained using `RandomForestClassifier`
- Achieved **higher accuracy** than a single decision tree
- Able to provide **feature importances**

---

### 📊 Cross-Validation (CV)

| Model           | CV Accuracy (5-fold) |
|-----------------|----------------------|
| Decision Tree   | ~0.79 (varies)       |
| Random Forest   | ~0.84 (varies)       |

> 🔍 **Note:** Random Forests are generally more robust due to ensemble learning.

---

### 📌 Feature Importance

Top influential features (example):

- `cp` — Chest pain type  
- `thalach` — Maximum heart rate achieved  
- `oldpeak` — ST depression induced by exercise  
- `ca` — Number of major vessels colored by fluoroscopy

---

### 🧹 Notes

- 📦 **Graphviz** must be installed separately (system-level install).
- 📁 Ensure the dataset file (`heart.csv`) is placed in the project directory.
- 🛠️ For advanced tuning, consider using `GridSearchCV`.


