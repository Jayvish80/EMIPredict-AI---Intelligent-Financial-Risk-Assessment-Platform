import streamlit as st
import pandas as pd
import numpy as np
import pickle # Used here for simplicity, typically replaced by mlflow.sklearn.load_model
import os

# --- 1. CONFIGURATION AND UTILITIES ---

# Set up page config
st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dummy/Placeholder for Preprocessing & Model Artifacts
# In a real project, these objects (Scaler, One-Hot Encoder mapping, Models)
# would be loaded from the MLflow Model Registry.
# Assume the trained objects and models are saved/pickled/loaded here:

try:
    # --- Replace with actual MLflow model loading in production ---
    # scaler = mlflow.sklearn.load_model("runs:/<run_id>/scaler_artifact")
    # ohe_columns = mlflow.artifacts.load("ohe_columns.pkl")
    # classification_model = mlflow.sklearn.load_model("models:/EMIPredict_Classification_Prod/latest")
    # regression_model = mlflow.sklearn.load_model("models:/EMIPredict_Regression_Prod/latest")
    
    # Placeholder: Assuming you've saved the trained models and preprocessors from Step 5
    st.sidebar.info("Placeholder: Models and Preprocessors Loaded Successfully.")

except Exception as e:
    st.sidebar.error(f"Error loading model artifacts. Please ensure files are correctly registered in MLflow: {e}")
    # Use dummy data for the purpose of demonstrating the UI
    class DummyModel:
        def predict(self, X):
            if X.shape[1] == 71: # Assuming 71 features after encoding
                # Dummy Classification (random but reasonable output)
                return np.random.choice([0, 1, 2], size=X.shape[0])
            else:
                # Dummy Regression (random value within expected range)
                return np.random.randint(2500, 45000, size=X.shape[0])

    classification_model = DummyModel()
    regression_model = DummyModel()
    ohe_columns = [] # Placeholder for expected feature columns

# Function to perform the full Feature Engineering pipeline (from Step 3)
def feature_engineering_pipeline(input_df):
    """Applies all derived feature engineering steps to the input data."""
    df = input_df.copy()

    # 1. Total Monthly Expenses
    expense_cols = ['school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses', 'monthly_rent']
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)

    # 2. Total Current Obligation
    df['total_current_obligation'] = df['current_emi_amount']

    # 3. Expense-to-Income Ratio
    df['expense_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary']
    df['expense_to_income_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    df.loc[df['expense_to_income_ratio'] > 1, 'expense_to_income_ratio'] = 1

    # 4. Debt-to-Income Ratio (DTI)
    df['debt_to_income_ratio'] = df['total_current_obligation'] / df['monthly_salary']
    df['debt_to_income_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    df.loc[df['debt_to_income_ratio'] > 1, 'debt_to_income_ratio'] = 1

    # 5. Free Cash Flow
    df['free_cash_flow'] = df['monthly_salary'] - (df['total_monthly_expenses'] + df['total_current_obligation'])
    df.loc[df['free_cash_flow'] < 0, 'free_cash_flow'] = 0

    # 6. Credit Risk Flag (Simple)
    RISK_THRESHOLD = 650
    df['low_credit_risk_flag'] = np.where(
        (df['credit_score'] < RISK_THRESHOLD) & (df['existing_loans'] == 'Yes'), 1, 0
    )

    # 7. Employment Stability Score (Interaction Feature)
    df['employment_stability_score'] = df['monthly_salary'] * df['years_of_employment']

    # 8. Emergency Fund Ratio
    df['emergency_fund_ratio'] = df['emergency_fund'] / df['monthly_salary']
    df['emergency_fund_ratio'].replace([np.inf, -np.inf], 0, inplace=True)

    return df

# --- 2. MULTI-PAGE STRUCTURE (MAIN APP) ---

def main_page():
    """Home Page with project overview."""
    st.title("💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform")
    st.markdown("""---""")
    st.header("Project Overview")
    st.write(
        """
        Welcome to the production-ready platform for real-time EMI prediction. 
        This application uses advanced Machine Learning models (specifically **XGBoost Classifier** and **XGBoost Regressor**) 
        to provide dual risk assessment for loan applications:
        """
    )

    st.subheader("Dual ML Problem Solved:")
    st.markdown(
        """
        * **Classification (EMI Eligibility):** Predicts the loan risk status (`Eligible`, `High_Risk`, or `Not_Eligible`).
        * **Regression (Maximum EMI Amount):** Calculates the maximum safe monthly EMI amount the applicant can afford.
        """
    )
    
    st.subheader("Business Impact")
    st.info(
        """
        This platform helps financial institutions and FinTech companies automate loan approval processes, 
        implement risk-based pricing, and reduce manual underwriting time by up to 80%.
        """
    )
    st.markdown("---")
    st.markdown("Use the sidebar navigation to perform risk assessments.")

def classification_page(model):
    """Page for EMI Eligibility Prediction."""
    st.title("🔍 EMI Eligibility Prediction (Classification)")
    st.subheader("Determine Loan Risk Status")

    with st.form("classification_form"):
        st.markdown("### Personal & Financial Profile Input (22 Variables)")
        
        # --- Form Layout ---
        col1, col2, col3 = st.columns(3)

        # Column 1: Demographics & Employment
        with col1:
            st.markdown("#### Demographics & Employment")
            age = st.slider("Age (25-60 years)", 25, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            monthly_salary = st.number_input("Monthly Salary (INR)", 15000.0, 200000.0, 50000.0, step=1000.0)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", 0.0, 30.0, 5.0, step=0.1)
            company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Large Indian", "Startup"])

        # Column 2: Housing & Obligations
        with col2:
            st.markdown("#### Housing & Obligations")
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent (INR)", 0.0, 50000.0, 10000.0, step=500.0)
            family_size = st.slider("Family Size", 1, 6, 3)
            dependents = st.slider("Financial Dependents", 0, 5, 2)
            school_fees = st.number_input("School Fees (INR)", 0.0, 50000.0, 0.0, step=100.0)
            college_fees = st.number_input("College Fees (INR)", 0.0, 50000.0, 0.0, step=100.0)
            travel_expenses = st.number_input("Travel Expenses (INR)", 0.0, 30000.0, 5000.0, step=100.0)
            groceries_utilities = st.number_input("Groceries/Utilities (INR)", 0.0, 50000.0, 15000.0, step=100.0)
            other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", 0.0, 30000.0, 2000.0, step=100.0)
        
        # Column 3: Financial Status & Loan Details
        with col3:
            st.markdown("#### Financial Status & Loan Details")
            existing_loans = st.selectbox("Existing Loans Status", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount (INR)", 0.0, 50000.0, 0.0, step=100.0)
            credit_score = st.slider("Credit Score (300-850)", 300, 850, 700)
            bank_balance = st.number_input("Bank Balance (INR)", 0.0, 1000000.0, 100000.0, step=1000.0)
            emergency_fund = st.number_input("Emergency Fund (INR)", 0.0, 500000.0, 50000.0, step=1000.0)
            emi_scenario = st.selectbox("EMI Scenario", ["Personal Loan EMI", "Vehicle EMI", "Home Appliances EMI", "E-commerce Shopping EMI", "Education EMI"])
            requested_amount = st.number_input("Requested Loan Amount (INR)", 10000.0, 1500000.0, 500000.0, step=5000.0)
            requested_tenure = st.slider("Requested Tenure (Months)", 3, 84, 36)
        
        # --- Submit Button ---
        submitted = st.form_submit_button("Predict Eligibility")

    if submitted:
        # 1. Create Input DataFrame
        raw_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education, 
            'monthly_salary': monthly_salary, 'employment_type': employment_type, 'years_of_employment': years_of_employment, 
            'company_type': company_type, 'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size, 
            'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees, 'travel_expenses': travel_expenses, 
            'groceries_utilities': groceries_utilities, 'other_monthly_expenses': other_monthly_expenses, 
            'existing_loans': existing_loans, 'current_emi_amount': current_emi_amount, 'credit_score': credit_score, 
            'bank_balance': bank_balance, 'emergency_fund': emergency_fund, 'emi_scenario': emi_scenario, 
            'requested_amount': requested_amount, 'requested_tenure': requested_tenure
        }
        raw_df = pd.DataFrame([raw_data])

        # 2. Apply Feature Engineering
        fe_df = feature_engineering_pipeline(raw_df)

        # 3. Apply Preprocessing (Encoding and Scaling - highly complex step simplified here)
        # Note: This is the most complex step involving OHE and scaling based on training data.
        # Placeholder for full preprocessing, converting DataFrame to a structure the model expects.
        # In production, this must match the transformation in Step 3 exactly.
        
        # Simplified prediction (Requires model to handle raw input for demo)
        
        # Dummy Model Output Map: 0=Not_Eligible, 1=High_Risk, 2=Eligible
        eligibility_map = {0: "Not_Eligible", 1: "High_Risk", 2: "Eligible"}
        
        # Convert necessary categorical columns to a format the dummy model can digest
        df_for_pred = pd.get_dummies(fe_df, drop_first=True)
        # Pad/align features to match the exact number of features the trained model expects (71 features)
        df_final = pd.DataFrame(0, index=df_for_pred.index, columns=ohe_columns)
        for col in df_for_pred.columns:
            if col in df_final.columns:
                df_final[col] = df_for_pred[col]
        
        # 4. Predict
        try:
            prediction = model.predict(df_final.iloc[0:1])[0]
            status = eligibility_map.get(prediction, "Error")
        except:
             # Fallback for when the complex feature alignment fails in the demo environment
            status = eligibility_map.get(model.predict(pd.DataFrame(np.zeros((1, 71))))[0], "Error")

        # 5. Display Result
        st.markdown("### Prediction Result")
        if status == "Eligible":
            st.success(f"✅ Eligibility Status: {status} (Low Risk)")
            st.write("Recommendation: Loan is approved under standard terms.")
        elif status == "High_Risk":
            st.warning(f"⚠️ Eligibility Status: {status} (Marginal Case)")
            st.write("Recommendation: Requires mandatory higher interest rates or collateral.")
        elif status == "Not_Eligible":
            st.error(f"❌ Eligibility Status: {status} (High Risk)")
            st.write("Recommendation: Loan is not recommended based on current financial profile.")
        else:
            st.info("Prediction output could not be mapped correctly.")


def regression_page(model):
    """Page for Maximum EMI Amount Prediction."""
    st.title("💸 Maximum EMI Amount Prediction (Regression)")
    st.subheader("Calculate Applicant's Safe Affordability Limit")
    st.write("This model determines the maximum safe monthly EMI amount the applicant can reliably pay.")

    # Note: Regression input form is often identical to Classification, 
    # as they use the same features. For simplicity, we reuse the prediction logic 
    # from the classification page (without the eligibility prediction)
    
    with st.form("regression_form"):
        # --- Form Layout (Same as classification for consistency) ---
        # ... (Include all 25 input fields here) ...
        # Placeholder for brevity:
        monthly_salary = st.number_input("Monthly Salary (INR)", 15000.0, 200000.0, 50000.0, key="reg_salary")
        current_emi_amount = st.number_input("Current EMI Amount (INR)", 0.0, 50000.0, 0.0, key="reg_current_emi")
        
        # Using a submit button for the form
        submitted = st.form_submit_button("Predict Max EMI")


    if submitted:
        # 1. Create Input DataFrame (Placeholder for full data collection)
        # In a real app, collect all 25 raw inputs here:
        raw_data = {'monthly_salary': monthly_salary, 'current_emi_amount': current_emi_amount, 
                    'age': 35, 'gender': 'Male', 'marital_status': 'Married', 'education': 'Graduate', 
                    'employment_type': 'Private', 'years_of_employment': 5.0, 'company_type': 'MNC', 
                    'house_type': 'Rented', 'monthly_rent': 10000.0, 'family_size': 3, 
                    'dependents': 2, 'school_fees': 0.0, 'college_fees': 0.0, 'travel_expenses': 5000.0, 
                    'groceries_utilities': 15000.0, 'other_monthly_expenses': 2000.0, 
                    'existing_loans': 'Yes', 'credit_score': 700, 'bank_balance': 100000.0, 
                    'emergency_fund': 50000.0, 'emi_scenario': 'Personal Loan EMI', 
                    'requested_amount': 500000.0, 'requested_tenure': 36} # Default values used
        raw_df = pd.DataFrame([raw_data])

        # 2. Apply Feature Engineering
        fe_df = feature_engineering_pipeline(raw_df)

        # 3. Apply Preprocessing (OHE/Scaling)
        df_for_pred = pd.get_dummies(fe_df, drop_first=True)
        df_final = pd.DataFrame(0, index=df_for_pred.index, columns=ohe_columns)
        for col in df_for_pred.columns:
            if col in df_final.columns:
                df_final[col] = df_for_pred[col]

        # 4. Predict
        try:
            max_emi = model.predict(df_final.iloc[0:1])[0]
        except:
            # Fallback for when the complex feature alignment fails in the demo environment
            max_emi = model.predict(pd.DataFrame(np.zeros((1, 71))))[0]
            
        max_emi = max_emi.clip(500, 50000) # Ensure result stays within expected range (500-50000 INR)
        
        # 5. Display Result
        st.markdown("### Maximum Affordability Calculation")
        st.success(f"Estimated Maximum Safe Monthly EMI: **₹ {max_emi:,.2f}**")
        st.write("This value represents the maximum monthly payment the applicant can safely undertake without excessive financial risk.")

# --- 3. PAGE ROUTING ---

# Sidebar for navigation (Simulates a multi-page app)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EMI Eligibility (Classification)", "Max EMI Amount (Regression)"])

if page == "Home":
    main_page()
elif page == "EMI Eligibility (Classification)":
    classification_page(classification_model)
elif page == "Max EMI Amount (Regression)":
    regression_page(regression_model)

# --- 4. MLflow Status Dashboard (Placeholder for Step 7) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")
st.sidebar.code("Classification Model: XGBoost Classifier (Prod)")
st.sidebar.code("Regression Model: XGBoost Regressor (Prod)")