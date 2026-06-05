import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap

logistic_model = load("artifacts/logistic_model.pkl")
rf_model = load("artifacts/rf_model.pkl")
xgb_model = load("artifacts/xgb_model.pkl")

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])
preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

with tab1:

    st.title("Customer Churn Prediction App")

    st.write("Enter customer details to predict churn probability")


    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )


    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])

    tenure = st.slider("Tenure (months)",0,72)

    phoneservice = st.selectbox("Phone Service", ["Yes","No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    onlinesecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    streamingtv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    )

    monthly = st.number_input("Monthly Charges",0.0,200.0)
    total = st.number_input("Total Charges",0.0,10000.0)


    data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phoneservice],
    "MultipleLines":[multiplelines],
    "InternetService":[internet],
    "OnlineSecurity":[onlinesecurity],
    "OnlineBackup":[onlinebackup],
    "DeviceProtection":[deviceprotection],
    "TechSupport":[techsupport],
    "StreamingTV":[streamingtv],
    "StreamingMovies":[streamingmovies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
    })

    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    if st.button("Predict Churn"):

        # Prediction
        prob = model.predict_proba(data)[0][1]

        st.metric(
            "Churn Probability",
            f"{prob*100:.1f}%"
        )

        st.progress(float(prob))

        # Risk Category
        if prob > 0.6:
            st.error("High Risk Customer")
        elif prob > 0.3:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")

        st.write(
            f"Model Used: **{model_choice}**"
        )

        # SHAP Section
        st.subheader(
            "Prediction Explanation (SHAP)"
        )

        X_transformed = preprocessor.transform(data)

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        # Select correct model for SHAP
        if model_choice == "Random Forest":

            explainer = shap.Explainer(
                rf_model.named_steps["model"]
            )

        elif model_choice == "XGBoost":

            explainer = shap.Explainer(
                xgb_model.named_steps["model"]
            )

        else:

            st.info(
                "SHAP explanations are available for Random Forest and XGBoost models."
            )

            explainer = None

        if explainer is not None:

            shap_values = explainer(
                X_transformed_df
            )

            fig = plt.figure()

            if len(shap_values.values.shape) == 3:
                shap.plots.waterfall(
                    shap_values[0, :, 1],
                    show=False
                )
            else:
                shap.plots.waterfall(
                    shap_values[0],
                    show=False
                )

            st.pyplot(fig)

