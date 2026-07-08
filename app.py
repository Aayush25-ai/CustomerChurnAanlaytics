"""
app.py
-------
STEP 5 (optional) — an interactive Streamlit web app.

Purpose:
    Let anyone enter a customer's details in a simple form and instantly
    see the churn prediction + probability, without touching any code.
    This is a common way to demo an ML pipeline to non-technical
    stakeholders.

Run with:
    streamlit run app/app.py

Note: run this from the project ROOT folder (not inside app/) so the
relative paths to models/ resolve correctly, OR adjust MODEL_DIR below.
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Allow importing from src/ when running `streamlit run app/app.py`
# from the project root.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")


@st.cache_resource
def load_artifacts():
    """
    Cached so the model/preprocessor are loaded from disk only once per
    session, not on every user interaction (Streamlit reruns the whole
    script top-to-bottom on every widget change).
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    saved = joblib.load(MODEL_PATH)
    return preprocessor, saved["model"], saved["model_name"]


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
    st.title("📉 Customer Churn Predictor")
    st.write(
        "Enter a customer's details below to predict whether they are "
        "likely to churn."
    )

    preprocessor, model, model_name = load_artifacts()
    st.caption(f"Model in use: **{model_name}**")

    # --- Input form -----------------------------------------------------
    # Grouped into columns purely for a cleaner layout.
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

    if st.button("Predict Churn"):
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
            "TotalCharges": total_charges,
        }

        df = pd.DataFrame([customer])
        X_transformed = preprocessor.transform(df)
        probability = model.predict_proba(X_transformed)[0, 1]
        prediction = "Yes" if probability >= 0.5 else "No"

        st.divider()
        if prediction == "Yes":
            st.error(f"⚠️ This customer is LIKELY TO CHURN "
                      f"(probability: {probability:.1%})")
        else:
            st.success(f"✅ This customer is likely to STAY "
                       f"(churn probability: {probability:.1%})")


if __name__ == "__main__":
    main()
