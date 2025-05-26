# Step-by-Step: Data Cleaning & Preprocessing in Machine Learning(Titanic Dataset)🚀
This project demonstrates step-by-step data preprocessing techniques applied to the **Titanic Dataset** to prepare it for Machine Learning models.

### ✅ Step 1: Import Dataset and Explore Basic Info
📁 **Dataset**-We use the classic Titanic dataset containing information about passengers, such as age, sex, class, fare, and survival status.

### ✅ Step 2: Handle Missing Values (Mean/Median/Imputation)
- Fill missing values in Age using mean.
- Drop Cabin column due to high number of nulls.
- Drop rows with missing Embarked

### ✅ Step 3: Convert Categorical Features into Numeric (Encoding)
<Encode Sex: 'male' → 1, 'female' → 0
Encode Embarked: 'S' → 0, 'C' → 1, 'Q' → 2

### ✅ Step 4: Normalize/Standardize Numerical Features
Using StandardScaler from sklearn to scale Age and Fare.

### ✅ Step 5: Visualize Outliers with Boxplots and Remove Them
* Plot boxplots for Age and Fare.
* Remove outliers using the IQR method.
### ✅ Step 6:🧾 Final Output
- Cleaned dataset ready for ML model training.
- No missing values.
- Outliers removed.
- All features properly encoded and scaled.
## 📌 Objectives
1. Import the dataset and explore basic structure.
2. Handle missing values using imputation techniques.
3. Encode categorical variables into numeric format.
4. Normalize/standardize numerical features.
5. Visualize and remove outliers using boxplots and IQR.
## 🛠 Tools Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
## 📂 How to Use
1. Download the Titanic dataset (titanic.csv)
2. Run the script in any Python environment (Jupyter Notebook, VS Code, etc.)
3. Use df for model building (e.g., logistic regression, decision trees)
