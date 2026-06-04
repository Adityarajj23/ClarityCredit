import streamlit as st
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ── Constants (must match train_model.py) ─────────────────────────────────────
BALANCE_INCOME_OFFSET = 150000.0   # same `c` used during training

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

# ── Preprocessing — mirrors train_model.py exactly ────────────────────────────
def preprocess_data(data, scaler):
    """
    data: raw DataFrame with columns:
      Gender, Married, Dependents, Education, Self_Employed,
      ApplicantIncome(₹), CoapplicantIncome(₹),
      LoanAmount(₹), Loan_Amount_Term(months),
      Credit_History, Property_Area
    Returns the 4-feature scaled DataFrame the model expects.
    """
    # Feature engineering — EMI is calculated from LoanAmount / Loan_Amount_Term
    data['Total_Income']    = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['EMI']             = data['LoanAmount'] / data['Loan_Amount_Term']
    data['Balance_Income']  = data['Total_Income'] - data['EMI']

    # Log transforms
    data['Total_Income_log']   = np.log(data['Total_Income'])
    data['EMI_log']            = np.log(data['EMI'])
    data['Balance_Income_log'] = np.log(data['Balance_Income'] + BALANCE_INCOME_OFFSET)

    # Drop raw columns we no longer need
    data = data.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Total_Income', 'EMI', 'Balance_Income'], axis=1)

    # Scale only the log-transformed numerical columns
    numerical_columns_log = ['Total_Income_log', 'EMI_log', 'Balance_Income_log']
    log_df = pd.DataFrame(
        data[numerical_columns_log].values,
        columns=numerical_columns_log
    )
    data[numerical_columns_log] = scaler.transform(log_df)

    # Final 4 features the model was trained on
    feature_column = ['Credit_History', 'Total_Income_log', 'EMI_log', 'Balance_Income_log']
    return data[feature_column]


# ── App ────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Loan Approval Predictor — India",
        page_icon="₹",
        layout="wide"
    )

    st.title("₹ Loan Approval Prediction — India")
    st.caption(
        "Fill in the applicant's details across the three tabs below. "
        "The Random Forest model will predict loan approval and explain the decision using SHAP values."
    )
    st.divider()

    # ── Tabbed input form ──────────────────────────────────────────────────────
    st.subheader("📋 Application Details")
    tab1, tab2, tab3 = st.tabs(["👤 Profile Information", "💰 Income & Location", "📊 Loan Requirements"])

    with tab1:
        st.markdown("**Applicant Demographics**")
        col_a, col_b = st.columns(2)
        with col_a:
            Gender      = st.selectbox("Gender", ["Male", "Female"])
            Married     = st.selectbox("Marital Status", ["Yes", "No"])
        with col_b:
            Dependents  = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            Education   = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    with tab2:
        st.markdown("**Financial Context & Location**")
        col_c, col_d = st.columns(2)
        with col_c:
            ApplicantIncome    = st.number_input(
                "Monthly Applicant Income (₹)",
                min_value=1000, max_value=50_00_000, value=40_000, step=1000,
                help="Net monthly take-home income of the primary applicant."
            )
            CoapplicantIncome  = st.number_input(
                "Monthly Co-applicant Income (₹)",
                min_value=0, max_value=50_00_000, value=0, step=1000,
                help="Monthly income of co-applicant (spouse/guarantor). Enter 0 if none."
            )
        with col_d:
            Property_Area = st.selectbox(
                "Property Area Type",
                ["Urban", "Semiurban", "Rural"],
                help="Location type of the property to be purchased."
            )

    with tab3:
        st.markdown("**Loan Agreement & Credit Standing**")
        col_e, col_f = st.columns(2)
        with col_e:
            LoanAmount_Lakhs = st.number_input(
                "Loan Amount requested (₹ Lakhs)",
                min_value=0.5, max_value=500.0, value=15.0, step=0.5,
                help="Total loan amount requested in Lakhs."
            )
            Loan_Amount_Term = st.number_input(
                "Loan Term (Months)",
                min_value=12, max_value=480, value=360, step=12,
                help="Repayment term of the loan in months (e.g., 360 for 30 years)."
            )
        with col_f:
            Credit_History = st.selectbox(
                "Credit History Standing",
                ["Good (repaid all debts)", "Bad (missed payments)"],
                help="Whether the applicant has a satisfactory repayment record."
            )

    st.divider()

    # ── Derived live metrics ───────────────────────────────────────────────────
    total_income  = ApplicantIncome + CoapplicantIncome
    loan_amount_inr = LoanAmount_Lakhs * 1_00_000
    calculated_emi = loan_amount_inr / Loan_Amount_Term
    balance_after = total_income - calculated_emi

    m1, m2, m3 = st.columns(3)
    m1.metric("Combined Monthly Income", f"₹{total_income:,.0f}")
    m2.metric("Monthly EMI (Estimated)", f"₹{calculated_emi:,.0f}")
    m3.metric(
        "Balance After EMI", f"₹{balance_after:,.0f}",
        delta="Surplus" if balance_after > 0 else "Deficit",
        delta_color="normal" if balance_after > 0 else "inverse"
    )

    st.divider()

    # ── Map UI values → model-compatible numeric values ────────────────────────
    data = {
        "Gender":           1 if Gender == "Male" else 0,
        "Married":          1 if Married == "Yes" else 0,
        "Dependents":       3 if Dependents == "3+" else int(Dependents),
        "Education":        0 if Education == "Graduate" else 1,
        "Self_Employed":    1 if Self_Employed == "Yes" else 0,
        "ApplicantIncome":  float(ApplicantIncome),
        "CoapplicantIncome":float(CoapplicantIncome),
        "LoanAmount":       float(loan_amount_inr),  # in ₹
        "Loan_Amount_Term": float(Loan_Amount_Term),
        "Credit_History":   1.0 if Credit_History == "Good (repaid all debts)" else 0.0,
        # Alphabetical LabelEncoder order: Rural=0, Semiurban=1, Urban=2
        "Property_Area":    ["Rural", "Semiurban", "Urban"].index(Property_Area),
    }
    user_input = pd.DataFrame(data, index=[0])

    # ── Predict button ─────────────────────────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1, 1.5])
    with col_btn2:
        submit = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

    if submit:
        if total_income <= 0:
            st.error("Combined income must be greater than ₹0.")
            return

        model  = load_model()
        scaler = load_scaler()

        processed_data = preprocess_data(user_input.copy(), scaler)
        prediction     = model.predict(processed_data)[0]
        proba          = model.predict_proba(processed_data)[0]

        st.divider()

        # ── Result ─────────────────────────────────────────────────────────────
        if prediction == 1:
            st.success(
                f"### ✅ APPROVED — Approval probability: {proba[1]*100:.1f}%\n\n"
                f"The model predicts this application **meets the credit criteria**. "
                f"An estimated EMI of **₹{calculated_emi:,.0f}/month** appears manageable "
                f"against a combined income of **₹{total_income:,.0f}/month**."
            )
        else:
            st.error(
                f"### ❌ DECLINED — Approval probability: {proba[1]*100:.1f}%\n\n"
                f"The model predicts this application **does not meet the credit criteria**. "
                "Common reasons: poor credit history, high EMI-to-income ratio, or low disposable balance."
            )

        # ── SHAP Explanation ───────────────────────────────────────────────────
        st.subheader("Model Explainability (SHAP Values)")
        st.caption(
            "SHAP (SHapley Additive exPlanations) decomposes the prediction — "
            "showing how each feature pushed the approval probability **up (red/pink)** "
            "or **down (blue)** from the model's baseline."
        )

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(processed_data)

        sv   = shap_values.values[0, :, 1]   # class 1 = Approved
        bv   = shap_values.base_values[0, 1]
        dv   = shap_values.data[0]

        fig, ax = plt.subplots(figsize=(10, 4))
        plt.sca(ax)
        shap.waterfall_plot(
            shap.Explanation(
                values        = sv,
                base_values   = bv,
                data          = dv,
                feature_names = ["Credit History", "Total Income (log)", "EMI (log)", "Balance Income (log)"]
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Feature definitions ────────────────────────────────────────────────
        st.subheader("Feature Definitions & Interpretation")
        st.markdown(
            "- **Credit History**: Most critical binary factor. A good history strongly pulls the decision toward approval.\n"
            "- **Total Income** *(log-scaled)*: Combined monthly income of applicant + co-applicant in ₹. Higher income increases repayment capacity.\n"
            "- **EMI** *(log-scaled)*: Monthly instalment = Loan Amount ÷ Repayment Term. Lower EMI reduces financial strain.\n"
            "- **Balance Income** *(log-scaled)*: Disposable income remaining after paying EMI. Higher surplus is highly favourable.\n\n"
            "> *Note: Demographic fields (Gender, Marital Status, etc.) are collected for audit purposes but are intentionally excluded "
            "from the final prediction model to ensure fairness and reduce demographic bias.*"
        )

        # ── Raw feature breakdown ──────────────────────────────────────────────
        st.subheader("Your Input — Derived Metrics")
        st.table(pd.DataFrame({
            "Feature": [
                "Credit History",
                "Total Monthly Income",
                "Monthly EMI",
                "Balance Income after EMI",
            ],
            "Value": [
                "Good" if Credit_History.startswith("Good") else "Bad",
                f"₹{total_income:,.0f}",
                f"₹{calculated_emi:,.0f}",
                f"₹{balance_after:,.0f}",
            ],
            "SHAP Impact on Approval": [
                f"{sv[0]:+.4f}",
                f"{sv[1]:+.4f}",
                f"{sv[2]:+.4f}",
                f"{sv[3]:+.4f}",
            ]
        }))


if __name__ == "__main__":
    main()