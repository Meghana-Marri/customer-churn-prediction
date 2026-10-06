import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# -----------------------------
# Load Model and Scaler
# -----------------------------

model = joblib.load("churn_logistic_model.pkl")
scaler = joblib.load("churn_scaler.pkl")


# -----------------------------
# Training Information
# -----------------------------

categorical_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

feature_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]


# -----------------------------
# App Title
# -----------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to churn."
)


# -----------------------------
# Customer Details
# -----------------------------

st.header("Customer Information")


col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

with col2:

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=140.0
    )


# -----------------------------
# Prediction
# -----------------------------

st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    customer_df = pd.DataFrame([customer])


    # Convert categorical columns to the same categories
    # used during model training

    category_values = {
        "gender": ["Female", "Male"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": [
            "No",
            "Yes",
            "No phone service"
        ],
        "InternetService": [
            "DSL",
            "Fiber optic",
            "No"
        ],
        "OnlineSecurity": [
            "No",
            "Yes",
            "No internet service"
        ],
        "OnlineBackup": [
            "No",
            "Yes",
            "No internet service"
        ],
        "DeviceProtection": [
            "No",
            "Yes",
            "No internet service"
        ],
        "TechSupport": [
            "No",
            "Yes",
            "No internet service"
        ],
        "StreamingTV": [
            "No",
            "Yes",
            "No internet service"
        ],
        "StreamingMovies": [
            "No",
            "Yes",
            "No internet service"
        ],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year"
        ],
        "PaperlessBilling": [
            "No",
            "Yes"
        ],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    }


    for col in categorical_cols:

        customer_df[col] = pd.Categorical(
            customer_df[col],
            categories=category_values[col]
        )


    # One-hot encoding

    customer_df = pd.get_dummies(
        customer_df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )


    # Make sure columns exactly match training data

    customer_df = customer_df.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scale features

    customer_scaled = scaler.transform(customer_df)


    # Get probability

    probability = model.predict_proba(
        customer_scaled
    )[0][1]


    # Threshold selected during model evaluation

    threshold = 0.30

    prediction = 1 if probability >= threshold else 0


    # -----------------------------
    # Display Result
    # -----------------------------

    st.header("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )


    if prediction == 1:

        st.error(
            "🔴 High Risk: Customer is likely to churn."
        )

        st.write(
            "The model predicts that this customer "
            "has a high probability of leaving the service."
        )

    else:

        st.success(
            "🟢 Low Risk: Customer is unlikely to churn."
        )

        st.write(
            "The model predicts that this customer "
            "is likely to stay."
        )