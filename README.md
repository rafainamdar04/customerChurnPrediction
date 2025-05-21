# Customer Churn Prediction

Predict whether telecom customers will churn (leave the service) using machine learning, with a Streamlit app for interactive visualization and dashboard.

---

## 🚀 Project Overview

This project uses a telecom customer dataset with a mix of categorical and numerical features to predict customer churn. The solution includes machine learning models and a Streamlit app that displays churn predictions and a dashboard with visualizations.

---

## 📂 Dataset Summary

- Contains 21 columns including the target variable `Churn`  
- Mix of categorical and numerical features  

---

## 🛠 Features & Workflow

- Data cleaning & preprocessing  
- Exploratory Data Analysis with interactive visualizations in Streamlit  
- Encoding categorical variables with LabelEncoder  
- Data balancing using SMOTE  
- Model training & cross-validation with Decision Tree, Random Forest, and XGBoost  
- Random Forest Classifier performed best and was chosen as the final model  
- Model evaluation with accuracy, confusion matrix, and classification report  
- Save model and encoders for prediction use  

---

## 💻 Installation & Running the App

```bash
git clone <repo-url>
cd customer-churn-prediction
python -m venv customerChurn

# Activate virtual environment
# On Linux/macOS
source customerChurn/bin/activate

# On Windows PowerShell
.\customerChurn\Scripts\Activate.ps1

# On Windows CMD
customerChurn\Scripts\activate.bat

pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
