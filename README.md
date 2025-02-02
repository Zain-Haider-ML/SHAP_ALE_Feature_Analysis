# **Fuel Economy Prediction: Machine Learning & Model Explainability**

## **Overview**
This project analyzes a **fuel economy dataset** using **Neural Networks (NN) and XGBoost** to predict vehicle fuel efficiency. The goal is to build robust models while ensuring interpretability using SHAP (SHapley Additive exPlanations) and ALE (Accumulated Local Effects) plots. The analysis also includes **statistical correlation tests** to examine feature relationships.  

## **Key Features**
- **Data Preprocessing:** Handling missing values, checking feature distributions, and exploring relationships.
- **Model Training:**
  - **Neural Networks (TensorFlow/Keras)**
  - **XGBoost (Extreme Gradient Boosting)**
- **Overfitting Prevention:** Evaluating performance using a separate test dataset.
- **Explainability Techniques:**
  - **SHAP Values:** Understanding how each feature contributes to the prediction.
  - **ALE Plots:** Exploring non-linear feature effects without interaction biases.
- **Statistical Analysis:** Applying correlation tests to identify significant relationships among variables.

---

## **Dataset Information**
The dataset used in this project is **`fueleconomy.csv`**, which contains detailed information on various vehicles, including:
- **Vehicle Specifications:** Engine size, horsepower, weight, fuel type, etc.
- **Fuel Economy Metrics:** Miles per gallon (MPG) and other efficiency indicators.

### **Data Preprocessing Steps**
- **Handling Missing Data:** Identifying and addressing missing values.
- **Feature Engineering:** Creating new features or transforming existing ones.
- **Exploratory Data Analysis (EDA):** Statistical summary, visualizations, and correlation analysis.

---

## **Installation & Dependencies**
Ensure you have Python installed and set up a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn tensorflow keras xgboost shap alepython
```

---

## **Modeling Approach**
### **1. Neural Network (NN)**
- Implemented using **TensorFlow/Keras**.
- Architecture: Includes dense layers with activation functions like ReLU.
- Optimizer: Adam, with Mean Squared Error (MSE) as the loss function.
- Regularization: Normalization layer to prevent overfitting.

### **2. XGBoost**
- A powerful gradient boosting model used for regression tasks.
- Tuned hyperparameters to optimize performance.
- Importance plots used to analyze feature contributions.

---

## **Model Evaluation**
To assess model performance, we use:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

The dataset is split into training and test sets to evaluate generalization ability.

---

## **Explainability Methods**
### **1. SHAP (SHapley Additive exPlanations)**
- Used to interpret model predictions by measuring feature importance.
- Provides global and local interpretability.

### **2. ALE (Accumulated Local Effects)**
- Shows how individual features influence the model’s output.
- Unlike Partial Dependence Plots (PDPs), ALE accounts for correlated features.

---

## **Statistical Analysis**
- **Feature Correlation:** Pearson and Spearman correlation coefficients.
- **Hypothesis Testing:** Statistical significance tests for feature importance.

---

## **Conclusion**

This project demonstrates the effectiveness of Neural Networks and XGBoost in predicting vehicle fuel efficiency. Through statistical analysis and model explainability techniques like SHAP and ALE, I also got valuable insights into the most influential features driving predictions. The findings can help automotive manufacturers, policymakers, and consumers make informed decisions about fuel-efficient vehicles. Future work could explore hyperparameter tuning, additional feature engineering, and alternative modeling approaches to enhance prediction accuracy further.

---

