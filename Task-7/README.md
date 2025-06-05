# 🧠 Support Vector Machines (SVM) for Binary Classification

## 📌 Objective

Apply **Support Vector Machines (SVM)** to perform both **linear** and **non-linear** binary classification using the **Breast Cancer dataset**. Gain hands-on experience with:

- Margin maximization  
- Kernel trick (RBF kernel)  
- Hyperparameter tuning  
- Decision boundary visualization  
- Cross-validation  

---

## 🚀 Tools & Libraries Used

- Python 3  
- NumPy  
- Matplotlib  
- Scikit-learn  
- PCA (for visualization)  
- Google Colab (Recommended)  

---

## 📁 Dataset

We use the **Breast Cancer dataset** from `sklearn.datasets`, which contains:

- 🧬 30 numerical features  
- 🟢 2 classes: Malignant (0) & Benign (1)  
- 👩‍⚕️ 569 samples  

> No need to download externally. It's built into `sklearn`.

---

## 🧪 What You'll Learn

- 🧭 How SVM finds optimal decision boundaries  
- 🌀 Using **RBF kernel** for non-linear separation  
- 🎯 Hyperparameter tuning with Grid Search (`C`, `gamma`)  
- 📊 Evaluation using confusion matrix & classification report  
- 🧵 PCA to reduce dimensions for 2D visualization  

---

## 📌 Task Checklist

| Task                                 | Status  |
|--------------------------------------|---------|
| Load and prepare the dataset         | ✅ Done |
| Train Linear and RBF Kernel SVMs     | ✅ Done |
| Visualize decision boundaries        | ✅ Done |
| Tune hyperparameters using GridSearchCV | ✅ Done |
| Evaluate performance using CV        | ✅ Done |

---

## 🧾 Code Workflow

### 1. 📦 Import Libraries & Load Dataset

To begin the project, essential libraries are imported. These include tools for:

- Loading datasets (`sklearn.datasets`)
- Data preprocessing and model building (`sklearn.preprocessing`, `sklearn.svm`, etc.)
- Evaluation and visualization

The **Breast Cancer Wisconsin Diagnostic Dataset** is loaded using `load_breast_cancer()` from Scikit-learn. This built-in dataset is ideal for binary classification tasks and contains:

- **569 samples**
- **30 numerical features** (e.g., mean radius, texture, smoothness)
- **Target variable** with 2 classes:  
  - `0` for **malignant**
  - `1` for **benign**

This dataset serves as the input for training and evaluating the SVM models.


### 2. 🔄 Preprocessing & Dimensionality Reduction

Before training the SVM models, the data must be preprocessed to ensure that features are on the same scale. This is done using **standardization**, which scales the features to have zero mean and unit variance. This step is crucial for SVMs as they are sensitive to feature scaling.

To enable visualization of the decision boundaries in a 2D plot, we use **Principal Component Analysis (PCA)**. PCA reduces the dataset from 30 dimensions down to 2 principal components while preserving the maximum possible variance. This makes it easier to visualize how the SVM separates classes in a lower-dimensional space.

---

### 3. 🤖 Training Linear and RBF SVM Models

Two types of SVM models are trained:

- **Linear Kernel SVM**: Suitable when data is linearly separable. It finds the hyperplane that maximizes the margin between classes.
  
- **RBF (Radial Basis Function) Kernel SVM**: Suitable for non-linear classification problems. It uses the **kernel trick** to transform data into higher dimensions where a linear separator can be found.

Both models are trained using the preprocessed and 2D-reduced dataset.

---

### 4. 🧭 Visualizing Decision Boundaries

To better understand how SVM models classify the data, decision boundaries are visualized:

- A grid of points is created across the feature space.
- The trained SVM model predicts the class label for each point.
- The result is displayed as a colored region representing the decision boundary.
- The actual data points are overlaid on this region to show how well the model separates the classes.

This visualization is done for both the linear and RBF kernel SVMs to compare how each handles the data.

---

### 5. 🎯 Hyperparameter Tuning

SVM performance can significantly depend on the choice of hyperparameters:

- **C**: Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification error.
- **Gamma**: Defines how far the influence of a single training example reaches (used in RBF kernel).

To find the best combination of these parameters, **GridSearchCV** is used. It performs an exhaustive search over a specified parameter grid using cross-validation to evaluate performance.

---

### 6. 📈 Cross-Validation Accuracy

To assess the generalization performance of the model, **k-fold cross-validation** is performed:

- The dataset is divided into *k* equal parts (commonly k=5).
- The model is trained on *k-1* parts and validated on the remaining part.
- This process is repeated *k* times, and the results are averaged.

Cross-validation ensures that the model performs well across different subsets of the data and is not overfitting to the training set.
## 📊 Results Summary

- ✅ **Linear SVM** performed well with separable classes.  
- ✅ **RBF Kernel SVM** captured non-linear trends better.  
- 📈 **Best parameters** via GridSearchCV yielded improved accuracy.  
- 🔄 **Cross-validation** showed consistent generalization performance.  

---

## 🎨 Visual Output

- 🟪 **Decision boundaries** of Linear vs RBF SVMs (plotted with PCA)  
- 🟢 **Colored scatter plots** by class  

> The visualization helps understand how different kernels affect the decision boundaries in a 2D space.

---

## 💡 Conclusion

- **SVM** is powerful for both linear and non-linear classification tasks.  
- The **kernel trick** allows SVMs to operate in higher-dimensional spaces without explicitly transforming the data.  
- Proper **feature scaling** and **hyperparameter tuning** are essential for optimal performance.  
- **PCA** is highly useful for visualizing high-dimensional data in 2D while retaining important variance.

---

## ▶️ Run on Google Colab

> 📌 Simply copy the provided code into [Google Colab](https://colab.research.google.com/) and run each cell step-by-step to explore the full SVM pipeline interactively.

---

## 📚 References

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)  
- [Support Vector Machine Intuition – TDS Article](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)  
- [Dataset Info – Breast Cancer Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)  
