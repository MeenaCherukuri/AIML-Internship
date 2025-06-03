# TASK - 6 : 🧠 K-Nearest Neighbors (KNN) Classification

## 📌 Objective
The goal of this project is to implement the **K-Nearest Neighbors (KNN)** algorithm to solve a classification problem using the **Iris dataset**. Through this task, you will gain a practical understanding of instance-based learning, Euclidean distance, and how to choose the optimal value of **K** for classification performance.

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy

---

## 📂 Dataset

We use the **Iris dataset**, which contains 150 samples of iris flowers from three different species (*setosa, versicolor, virginica*), each with four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The dataset is available in `scikit-learn`.

---

## 📚 Learning Outcomes

- Understanding **instance-based learning** (lazy learning).
- Applying **Euclidean distance** in classification.
- Selecting optimal **K** value.
- Evaluating classification performance.
- Visualizing **decision boundaries** in 2D.

---

## 🚀 How to Run the Project

1. Clone the repository or copy the code to your local environment.
2. Ensure the required Python libraries are installed:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn

# KNN - Step-by-Step Guide

## 🧪 Step-by-Step Implementation

### 1. Load and Explore Dataset
We load the **Iris dataset** using `sklearn.datasets.load_iris()` and inspect the features and target labels.

### 2. Normalize Features
KNN uses **distance metrics**, so feature scaling is crucial. We use `StandardScaler` to normalize the data.

### 3. Train/Test Split
Split the dataset into **training (70%)** and **testing (30%)** sets using `train_test_split`.

### 4. Train KNN Classifier
Train `KNeighborsClassifier` from **Scikit-learn** using different values of **K** and evaluate their performance using **accuracy scores**.

### 5. Evaluate the Model
Use metrics like:

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

Visualize the **confusion matrix** using a **Seaborn heatmap** for better interpretability.

### 6. Visualize Decision Boundaries
To visualize **decision boundaries**:

- Reduce features to the **first two** for 2D plotting.
- Create a **meshgrid** that spans the feature space.
- Plot a **contour map** showing the classification regions based on predicted labels.

---

## 📌 Notes

- Feature scaling is essential for KNN as it is distance-based.
- The best value of **K** should be selected by evaluating accuracy over a range of K values.
- Visualization of decision boundaries helps in understanding how the model separates different classes visually in feature space.

---

## ✅ Requirements

Make sure you have the following Python libraries installed:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn

## 📉 Confusion Matrix

|                      | Predicted Setosa | Predicted Versicolor | Predicted Virginica |
|----------------------|------------------|-----------------------|---------------------|
| **Actual Setosa**    | 16               | 0                     | 0                   |
| **Actual Versicolor**| 0                | 13                    | 2                   |
| **Actual Virginica** | 0                | 1                     | 13                  |

---

## 🗺️ Decision Boundary Plot (K=3)

![Decision Boundary](<!-- Add screenshot path if available -->)

---

## 🧠 Concepts Behind KNN

- **Lazy Learner**:  
  KNN doesn't build a model during training. It stores the entire training dataset and makes decisions during prediction based on distances.

- **Euclidean Distance**:  
  KNN uses this metric to calculate the distance between data points.

- **K Selection**:
  - **Small K**: More sensitive to noise in the dataset.
  - **Large K**: Can smooth out important patterns and cause underfitting.

---

## 📌 Conclusion

K-Nearest Neighbors is a **simple yet effective** classification algorithm. Through this project, we learned:

- The **importance of feature scaling** in distance-based models.
- How to determine the **optimal value of K**.
- How **decision boundaries** help visualize model performance in lower dimensions.

---

## 📎 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Iris Dataset on Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)

---
