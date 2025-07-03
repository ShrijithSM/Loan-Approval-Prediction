import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('loan_approval_classifier_rf_tuned.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'loan_approval_classifier_rf_tuned.pkl' is in the current directory.")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏦 Loan Approval Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar for information
    with st.sidebar:
        st.markdown("### 📋 About This App")
        st.markdown("""
        This application uses a **Random Forest Classifier** to predict loan approval based on various applicant features.
        
        **Features considered:**
        - Personal Information
        - Financial Details
        - Loan Information
        - Credit History
        """)
        
        st.markdown("### 📊 Model Performance")
        st.info("The model has been trained and tuned for optimal performance using cross-validation.")
        
        st.markdown("### 🔍 How to Use")
        st.markdown("""
        1. Fill in all the required information
        2. Click 'Predict Loan Approval'
        3. View your prediction result
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Applicant Information</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("loan_application_form"):
            # Personal Information Section
            st.markdown("#### 👤 Personal Information")
            col_personal1, col_personal2 = st.columns(2)
            
            with col_personal1:
                gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
                married = st.selectbox("Marital Status", ["No", "Yes"], help="Are you married?")
                dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"], help="Number of people dependent on you")
            
            with col_personal2:
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], help="Your highest education level")
                self_employed = st.selectbox("Self Employed", ["No", "Yes"], help="Are you self-employed?")
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], help="Location of the property")
            
            # Financial Information Section
            st.markdown("#### 💰 Financial Information")
            col_financial1, col_financial2 = st.columns(2)
            
            with col_financial1:
                applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=1000, 
                                                 help="Your monthly income in rupees")
                coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=1000,
                                                   help="Co-applicant's monthly income (if any)")
            
            with col_financial2:
                loan_amount = st.number_input("Loan Amount (₹ in thousands)", min_value=0, value=100, step=10,
                                            help="Loan amount requested in thousands")
                loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=12, max_value=480, value=360, step=12,
                                                 help="Loan repayment period in months")
            
            # Credit History Section
            st.markdown("#### 📈 Credit Information")
            credit_history = st.selectbox("Credit History", [1, 0], 
                                        format_func=lambda x: "Good Credit History" if x == 1 else "Poor/No Credit History",
                                        help="1 = Good credit history, 0 = Poor/No credit history")
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = prepare_input_data(
                    gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount,
                    loan_amount_term, credit_history, property_area
                )
                
                # Make prediction
                prediction, probability = make_prediction(model, input_data)
                
                # Display results in the second column
                with col2:
                    display_prediction_results(prediction, probability, input_data)
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Fill the form and click "Predict Loan Approval" to see results here.</div>', 
                   unsafe_allow_html=True)

def prepare_input_data(gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_amount_term, credit_history, property_area):
    """Prepare input data for prediction"""
    
    # Convert dependents to numeric (handle 3+ case)
    dependents_numeric = 3.0 if dependents == "3+" else float(dependents)
    
    # Create feature engineering
    total_income = applicant_income + coapplicant_income
    loan_amount_to_income_ratio = loan_amount * 1000 / (total_income + 1e-6)  # Convert loan amount back to actual value
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents_numeric],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
        'TotalIncome': [total_income],
        'LoanAmount_to_Income_Ratio': [loan_amount_to_income_ratio]
    })
    
    return input_data

def make_prediction(model, input_data):
    """Make prediction using the trained model"""
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def display_prediction_results(prediction, probability, input_data):
    """Display prediction results with styling"""
    if prediction is not None:
        # Main prediction result
        if prediction == 1:
            st.markdown(
                '<div class="prediction-box approved">✅ LOAN APPROVED!</div>',
                unsafe_allow_html=True
            )
            st.success("Congratulations! Your loan application is likely to be approved.")
        else:
            st.markdown(
                '<div class="prediction-box rejected">❌ LOAN REJECTED</div>',
                unsafe_allow_html=True
            )
            st.error("Unfortunately, your loan application is likely to be rejected.")
        
        # Probability scores
        st.markdown("#### 📊 Confidence Scores")
        approval_prob = probability[1] * 100
        rejection_prob = probability[0] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Approval Probability", f"{approval_prob:.1f}%")
        with col2:
            st.metric("Rejection Probability", f"{rejection_prob:.1f}%")
        
        # Progress bars for visual representation
        st.markdown("#### 📈 Probability Breakdown")
        st.progress(approval_prob / 100)
        st.caption(f"Approval: {approval_prob:.1f}%")
        
        # Additional insights
        st.markdown("#### 💡 Key Insights")
        
        # Calculate some insights based on input
        total_income = input_data['TotalIncome'].iloc[0]
        loan_amount = input_data['LoanAmount'].iloc[0] * 1000  # Convert back to actual amount
        loan_to_income_ratio = input_data['LoanAmount_to_Income_Ratio'].iloc[0]
        
        insights = []
        
        if total_income > 10000:
            insights.append("✅ Good income level")
        else:
            insights.append("⚠️ Consider increasing income sources")
        
        if loan_to_income_ratio < 0.3:
            insights.append("✅ Healthy loan-to-income ratio")
        else:
            insights.append("⚠️ High loan-to-income ratio")
        
        if input_data['Credit_History'].iloc[0] == 1:
            insights.append("✅ Good credit history")
        else:
            insights.append("❌ Poor credit history - major factor")
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Recommendations
        if prediction == 0:
            st.markdown("#### 🎯 Recommendations")
            recommendations = [
                "Improve your credit score",
                "Increase your income or add a co-applicant",
                "Reduce the loan amount requested",
                "Consider a longer loan term to reduce EMI burden"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

# Additional features section
def show_additional_features():
    """Show additional features and information"""
    st.markdown("---")
    st.markdown("### 📚 Additional Information")
    
    with st.expander("🔍 Understanding the Prediction"):
        st.markdown("""
        **Factors that positively influence loan approval:**
        - Higher income (applicant + co-applicant)
        - Good credit history
        - Lower loan amount relative to income
        - Graduate education level
        - Stable employment
        
        **Factors that may negatively influence approval:**
        - Poor or no credit history
        - High loan-to-income ratio
        - Very low income
        - Too many dependents relative to income
        """)
    
    with st.expander("📊 Model Information"):
        st.markdown("""
        **Model Details:**
        - Algorithm: Random Forest Classifier
        - Training: Cross-validated and hyperparameter tuned
        - Features: 13 engineered features
        - Performance: Optimized for accuracy and reliability
        
        **Note:** This prediction is based on historical data patterns and should be used as a guide only.
        """)

if __name__ == "__main__":
    main()
    show_additional_features()
