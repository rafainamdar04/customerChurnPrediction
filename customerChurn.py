import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('CustomerChurn.csv')
df = df.drop(columns=['customerID'])

print(df['gender'].unique())
print(df['Partner'].unique())
print(df['Dependents'].unique())

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in df:
    if col not in numerical_features:
        print(col, df[col].unique())

df['TotalCharges'] = df['TotalCharges'].replace({" ": '0.0'}).astype(float)

print(df['Churn'].value_counts())

def plot_histogram(df, column_name):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=column_name, kde=True)
    plt.title(f'Distribution of {column_name}')
    plt.axvline(df[column_name].mean(), color='red', linestyle='dashed', label='mean')
    plt.axvline(df[column_name].median(), color='blue', linestyle='-', label='median')
    plt.legend()
    plt.show()

plot_histogram(df, 'tenure')
plot_histogram(df, 'MonthlyCharges')
plot_histogram(df, 'TotalCharges')

def plot_boxplot(df, column_name):
    plt.figure(figsize=(10, 5))
    sns.boxplot(y=df[column_name])
    plt.title(f'Boxplot of {column_name}')
    plt.ylabel(column_name)
    plt.show()

plot_boxplot(df, 'tenure')
plot_boxplot(df, 'MonthlyCharges')
plot_boxplot(df, 'TotalCharges')

plt.figure(figsize=(8, 4))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

object_columns = df.select_dtypes(include='object').columns.to_list()
object_columns = ['SeniorCitizen'] + object_columns

for col in object_columns:
    plt.figure(figsize=(5, 3))
    sns.countplot(data=df, x=col)
    plt.title(f'Countplot of {col}')
    plt.show()

df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
object_columns = df.select_dtypes(include='object').columns

encoders = {}
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

x = df.drop(columns=['Churn'])
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

cv_scores = {}
for model_name, model in models.items():
    print(f"Training {model_name} with default parameters")
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring='accuracy')
    cv_scores[model_name] = scores
    print(f"{model_name} cross validation accuracy: {np.mean(scores):.2f}")
    print("-" * 70)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train_smote, y_train_smote)

y_test_pred = rfc.predict(x_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

model_data = {'model': rfc, 'feature_names': x.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

rfc_model = model_data['model']
feature_names = model_data['feature_names']

input_data = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "Churn": "No"
}

input_data_df = pd.DataFrame([input_data])
with open('encoders.pkl', "rb") as f:
    encoders = pickle.load(f)

input_data_df = input_data_df.drop(columns=["customerID", "Churn"])

for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

prediction = rfc_model.predict(input_data_df)
pred_prob = rfc_model.predict_proba(input_data_df)

print(prediction)
print(f'Prediction: {"churn" if prediction[0] == 1 else "no churn"}')
print(f"prediction probability: {pred_prob}")
