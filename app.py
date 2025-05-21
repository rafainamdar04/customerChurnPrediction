import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

@st.cache_data
def load_data():
    df = pd.read_csv('CustomerChurn.csv').drop(columns=['customerID'])
    df['TotalCharges'] = df['TotalCharges'].replace({" ": '0.0'}).astype(float)
    return df

@st.cache_data
def load_model_encoders():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data['model'], encoders

df = load_data()
model, encoders = load_model_encoders()

st.title("Customer Churn Prediction & Dashboard")


st.sidebar.header("Customer Input Features")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=29.85)
    TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=29.85)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    return pd.DataFrame([data])

input_df = user_input()

# Encode input using saved encoders
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# ------------------------------------
# Tabs for prediction and dashboard
# ------------------------------------
tab1, tab2 = st.tabs(["Prediction", "Data Visualization Dashboard"])

with tab1:
    st.header("Churn Prediction")
    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0][prediction]

        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        st.info(f"Prediction Probability: {pred_prob:.2f}")

with tab2:
    st.header("Exploratory Data Analysis (EDA)")

    # Histograms for numerical features
    st.subheader("Distribution of Numerical Features")
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(num_features):
        sns.histplot(df[col], kde=True, ax=axs[i], color='skyblue')
        axs[i].axvline(df[col].mean(), color='red', linestyle='dashed', label='Mean')
        axs[i].axvline(df[col].median(), color='green', linestyle='solid', label='Median')
        axs[i].set_title(f'{col} Distribution')
        axs[i].legend()
    st.pyplot(fig)

    # Boxplots for numerical features
    st.subheader("Boxplots of Numerical Features")
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(num_features):
        sns.boxplot(y=df[col], ax=axs2[i], color='lightgreen')
        axs2[i].set_title(f'{col} Boxplot')
    st.pyplot(fig2)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

    # Countplots for categorical features
    st.subheader("Countplots for Categorical Features")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    # Adding 'SeniorCitizen' as categorical (0/1)
    cat_cols = ['SeniorCitizen'] + [c for c in cat_cols if c != 'SeniorCitizen']

    # Use columns in a grid layout for neatness
    cols = st.columns(3)
    for i, col_name in enumerate(cat_cols):
        with cols[i % 3]:
            st.write(f"**{col_name}**")
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x=col_name, palette='pastel', ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            st.pyplot(fig4)
