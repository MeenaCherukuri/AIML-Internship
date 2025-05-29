# 🧠 Task 3: Linear Regression

### 🎯 Objective
Understand and implement simple and multiple linear regression to predict house prices.

---

## 📦 Dataset
**House Price Prediction Dataset**  
(Download link: [*Click here to download dataset*](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction))

---

## 🛠️ Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🔍 Workflow

### 1️⃣ Load & Preprocess Data
- Loaded CSV dataset into a DataFrame
- Selected only numeric columns
- Removed null values

### 2️⃣ Feature & Target Selection
- Simple Linear Regression: `GrLivArea` → `SalePrice`
- (Extendable to multiple features like `OverallQual`, `GarageArea`)

### 3️⃣ Train-Test Split
Used `train_test_split` from `sklearn.model_selection` with an 80-20 split.

### 4️⃣ Model Training
Trained a **Linear Regression** model using:

### 5️⃣ Evaluation Metrics
Used three standard regression metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² Score

### 6️⃣ Visualization
- Actual vs Predicted values
- Regression line overlay on scatterplot

## 📈 Results & Interpretation
- R² Score tells how much variance in SalePrice is explained by GrLivArea.
- Slope (coefficient) shows how much SalePrice increases per unit of GrLivArea.
- Visualization confirms linear relationship.

