import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING - DARK THEME
# ==========================================
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    
    [data-testid="stHeader"] {
        background-color: #0e1117;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: #161b22 !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border: 2px solid #58a6ff;
    }
    
    .stMetric label {
        color: #79c0ff !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        color: #c9d1d9 !important;
    }
    
    /* Prediction boxes */
    .prediction-high {
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
        color: #ffffff;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(248,81,73,0.3);
    }
    
    .prediction-low {
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #238636 0%, #3fb950 100%);
        color: #ffffff;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(63,185,80,0.3);
    }
    
    /* Info boxes */
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #161b22;
        border-left: 5px solid #58a6ff;
        color: #c9d1d9;
    }
    
    .info-box h3 {
        color: #79c0ff !important;
        font-weight: bold !important;
        margin-top: 0 !important;
    }
    
    .info-box p {
        color: #c9d1d9 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* Section titles */
    .section-title {
        color: #79c0ff;
        font-size: 26px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #c9d1d9 !important;
        font-weight: bold !important;
        font-size: 15px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #58a6ff !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="tab-list"] {
        border-bottom: 2px solid #30363d !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox, .stSlider {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    
    .stTextInput input::placeholder {
        color: #6e7681 !important;
    }
    
    /* Labels */
    label {
        color: #79c0ff !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 12px 30px !important;
        border-radius: 6px !important;
        border: 1px solid #3fb950 !important;
        box-shadow: 0 4px 8px rgba(63,185,80,0.3) !important;
    }
    
    .stButton > button:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 6px 12px rgba(63,185,80,0.4) !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #79c0ff !important;
    }
    
    /* Paragraphs */
    p, li {
        color: #c9d1d9 !important;
    }
    
    /* Selectbox dropdown */
    [data-baseweb="select"] {
        background-color: #0d1117 !important;
    }
    
    /* Slider */
    .stSlider [data-testid="stThumb"] {
        background-color: #58a6ff !important;
    }
    
    .stSlider [data-testid="stTickBar"] {
        background-color: #30363d !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #010409;
    }
    
    /* Radio buttons */
    .stRadio {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_model.pkl")
        return model
    except:
        st.error("⚠️ Model file (churn_model.pkl) not found. Please ensure the model is saved in the project directory.")
        return None

model = load_model()

# ==========================================
# HEADER
# ==========================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #79c0ff;'>🔮 Telecom Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #c9d1d9; font-size: 16px; font-weight: bold;'>Predict whether a customer will churn using machine learning</p>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color: #79c0ff; text-align: center; font-size: 20px;'>📍 Navigation Menu</h2>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Select a page:",
        ["🏠 Home", "🎯 Make Prediction", "📊 About Dataset", "ℹ️ Help & Info"],
        label_visibility="collapsed"
    )

# ==========================================
# PAGE: HOME
# ==========================================
if page == "🏠 Home":
    st.markdown("<h2 class='section-title'>Welcome to Churn Prediction System</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3>📈 About</h3>
        <p>This application predicts customer churn using advanced machine learning models trained on Telecom customer data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h3>🎯 Objective</h3>
        <p>Identify customers at risk of churning so proactive measures can be taken to retain them.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-box'>
        <h3>✨ Model Performance</h3>
        <p>Achieves high accuracy with multiple ML models including Gradient Boosting & Random Forest.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2 class='section-title'>🔑 Key Factors Influencing Churn</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff;'>
        <h4 style='color: #79c0ff;'>📅 Tenure</h4>
        <p style='color: #c9d1d9; font-weight: 500;'>Customers with shorter tenure are more likely to churn</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff; margin-top: 10px;'>
        <h4 style='color: #79c0ff;'>💰 Monthly Charges</h4>
        <p style='color: #c9d1d9; font-weight: 500;'>Higher charges correlate with increased churn risk</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff; margin-top: 10px;'>
        <h4 style='color: #79c0ff;'>📋 Contract Type</h4>
        <p style='color: #c9d1d9; font-weight: 500;'>Month-to-month contracts have higher churn rates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff;'>
        <h4 style='color: #79c0ff;'>🌐 Internet Service</h4>
        <p style='color: #c9d1d9; font-weight: 500;'>Fiber optic users show higher churn tendency</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff; margin-top: 10px;'>
        <h4 style='color: #79c0ff;'>🛡️ Tech Support</h4>
        <p style='color: #c9d1d9; font-weight: 500;'>Lack of support increases churn probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff; margin-top: 10px;'>
        <h4 style='color: #79c0ff;'>👤 Personal Factors</h4>
        <p style='color: #c9d1d9; font-weight: 500;'>Customers without partner/dependents churn more often</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2 class='section-title'>🚀 Get Started</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #161b22; padding: 20px; border-radius: 8px; border-left: 5px solid #58a6ff;'>
    <p style='color: #79c0ff; font-weight: bold; font-size: 16px;'>Click on <strong>🎯 Make Prediction</strong> in the sidebar to predict churn for a customer.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE: MAKE PREDICTION
# ==========================================
elif page == "🎯 Make Prediction":
    if model is None:
        st.error("Cannot proceed without model. Please load the model file first.")
    else:
        st.markdown("<h2 class='section-title'>Customer Information</h2>", unsafe_allow_html=True)
        
        # Organize inputs in logical sections
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Customer", "📞 Services", "💳 Billing", "📊 Usage"])
        
        # ==========================================
        # TAB 1: CUSTOMER DEMOGRAPHICS
        # ==========================================
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
                partner = st.selectbox("Has Partner", ["Yes", "No"], key="partner")
            
            with col2:
                dependents = st.selectbox("Has Dependents", ["Yes", "No"], key="dependents")
                tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long has the customer been with us?")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        
        # ==========================================
        # TAB 2: SERVICES
        # ==========================================
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
            
            with col2:
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="lines")
        
        # ==========================================
        # TAB 3: BILLING & CONTRACT
        # ==========================================
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
            
            with col2:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    key="payment"
                )
        
        # ==========================================
        # TAB 4: USAGE & CHARGES
        # ==========================================
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0, 
                                           help="Average monthly bill amount")
            
            with col2:
                st.metric("Estimated Total Charges ($)", f"{tenure * monthly_charges:.2f}")
        
        # ==========================================
        # PREDICTION BUTTON
        # ==========================================
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True, 
                                      help="Click to generate prediction")
        
        # ==========================================
        # MAKE PREDICTION
        # ==========================================
        if predict_button:
            try:
                # Ordinal encode Contract (as done in training notebook)
                contract_encoding = {
                    "Month-to-month": 0,
                    "One year": 1,
                    "Two year": 2
                }
                
                # Create input dictionary with proper encoding
                input_data = {
                    "tenure": tenure,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": tenure * monthly_charges,
                    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                    "Partner": partner,
                    "Dependents": dependents,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract_encoding[contract],  # Encode as numeric
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "gender": gender
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Make prediction - the pipeline handles all preprocessing
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                st.markdown("<h2 class='section-title'>🎯 Prediction Results</h2>", unsafe_allow_html=True)
                
                # Main prediction result
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if pred == 1:
                        st.markdown(
                            f"""
                            <div class='prediction-high'>
                            ⚠️ HIGH CHURN RISK<br>
                            Probability: {prob*100:.1f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        risk_level = "High"
                        recommendation = "Recommend immediate retention actions"
                    else:
                        st.markdown(
                            f"""
                            <div class='prediction-low'>
                            ✅ LOW CHURN RISK<br>
                            Probability: {prob*100:.1f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        risk_level = "Low"
                        recommendation = "Customer appears stable"
                
                # Detailed metrics
                st.markdown("<h3 style='color: #79c0ff; font-weight: bold; margin-top: 20px;'>Detailed Analysis</h3>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Risk Level", risk_level, delta="Critical" if pred == 1 else "Stable")
                
                with col2:
                    st.metric("Churn Probability", f"{prob*100:.1f}%")
                
                with col3:
                    st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")
                
                with col4:
                    st.metric("Retention Score", f"{(1-prob)*100:.1f}%")
                
                # Risk factors
                st.markdown("<h3 style='color: #79c0ff; font-weight: bold; margin-top: 20px;'>Risk Factors</h3>", unsafe_allow_html=True)
                
                risk_factors = []
                
                if tenure < 12:
                    risk_factors.append("🔴 Very short tenure (high-risk period)")
                elif tenure < 24:
                    risk_factors.append("🟡 Short tenure (moderate-risk period)")
                
                if internet_service == "Fiber optic":
                    risk_factors.append("🔴 Fiber optic service (higher churn rate)")
                
                if contract == "Month-to-month":
                    risk_factors.append("🔴 Month-to-month contract (low commitment)")
                
                if tech_support == "No" or tech_support == "No internet service":
                    risk_factors.append("🟡 No tech support (increased vulnerability)")
                
                if monthly_charges > 100:
                    risk_factors.append("🟡 High monthly charges")
                
                if partner == "No" and dependents == "No":
                    risk_factors.append("🟡 No family commitment (higher mobility)")
                
                if not risk_factors:
                    st.markdown("""
                    <div style='background-color: #0f2818; padding: 15px; border-radius: 8px; border-left: 5px solid #3fb950;'>
                    <p style='color: #3fb950; font-weight: bold; margin: 0;'>✅ No major risk factors identified. Customer appears stable.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for factor in risk_factors:
                        st.markdown(f"""
                        <div style='background-color: #3d1f1a; padding: 12px; border-radius: 6px; border-left: 4px solid #f85149; margin-bottom: 10px;'>
                        <p style='color: #f85149; font-weight: 500; margin: 0;'>{factor}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("<h3 style='color: #79c0ff; font-weight: bold; margin-top: 20px;'>💡 Recommendations</h3>", unsafe_allow_html=True)
                
                recommendations = []
                
                if pred == 1:
                    recommendations.append("⭐ Offer loyalty discount or upgrade package")
                    recommendations.append("📞 Reach out with personalized service review")
                    
                    if tech_support == "No":
                        recommendations.append("🛡️ Add tech support to improve satisfaction")
                    
                    if contract == "Month-to-month":
                        recommendations.append("📋 Encourage upgrade to longer-term contract with incentives")
                    
                    if tenure < 6:
                        recommendations.append("👋 Provide enhanced onboarding and support")
                
                else:
                    recommendations.append("✅ Continue current service strategy")
                    recommendations.append("📊 Monitor usage patterns for early warning signs")
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div style='background-color: #161b22; padding: 12px; border-radius: 6px; border-left: 4px solid #58a6ff; margin-bottom: 10px;'>
                    <p style='color: #79c0ff; font-weight: 500; margin: 0;'><strong>{i}.</strong> {rec}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export summary
                st.markdown("---")
                summary = f"""
                **Prediction Summary:**
                - Customer Risk: {'High Churn Risk' if pred == 1 else 'Low Churn Risk'}
                - Churn Probability: {prob*100:.1f}%
                - Recommendation: {recommendation}
                - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                st.markdown("""
                <div style='background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff;'>
                <p style='color: #79c0ff; font-weight: 500; margin: 0;'><strong>📊 Prediction Summary:</strong></p>
                <p style='color: #c9d1d9; margin: 5px 0;'>✓ Customer Risk: """ + ('High Churn Risk ⚠️' if pred == 1 else 'Low Churn Risk ✅') + f"""</p>
                <p style='color: #c9d1d9; margin: 5px 0;'>✓ Churn Probability: {prob*100:.1f}%</p>
                <p style='color: #c9d1d9; margin: 5px 0;'>✓ Recommendation: {recommendation}</p>
                <p style='color: #c9d1d9; margin: 5px 0;'>✓ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please ensure all fields are filled correctly.")

# ==========================================
# PAGE: ABOUT DATASET
# ==========================================
elif page == "📊 About Dataset":
    st.markdown("<h2 class='section-title'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", "7,032")
    with col2:
        st.metric("Total Features", "19")
    with col3:
        st.metric("Churn Rate", "~26.5%")
    
    st.markdown("<h3 class='section-title'>Feature Categories</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #161b22; padding: 20px; border-radius: 8px; border-left: 5px solid #58a6ff;'>
        <h4 style='color: #79c0ff;'>📊 Demographic Features:</h4>
        <ul style='color: #c9d1d9; font-weight: 500;'>
        <li>Gender</li>
        <li>SeniorCitizen</li>
        <li>Partner</li>
        <li>Dependents</li>
        </ul>
        
        <h4 style='color: #79c0ff; margin-top: 15px;'>📞 Service Features:</h4>
        <ul style='color: #c9d1d9; font-weight: 500;'>
        <li>PhoneService</li>
        <li>MultipleLines</li>
        <li>InternetService</li>
        <li>OnlineSecurity</li>
        <li>OnlineBackup</li>
        <li>DeviceProtection</li>
        <li>TechSupport</li>
        <li>StreamingTV</li>
        <li>StreamingMovies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #161b22; padding: 20px; border-radius: 8px; border-left: 5px solid #58a6ff;'>
        <h4 style='color: #79c0ff;'>💳 Account Features:</h4>
        <ul style='color: #c9d1d9; font-weight: 500;'>
        <li>Contract</li>
        <li>PaperlessBilling</li>
        <li>PaymentMethod</li>
        </ul>
        
        <h4 style='color: #79c0ff; margin-top: 15px;'>📈 Usage Features:</h4>
        <ul style='color: #c9d1d9; font-weight: 500;'>
        <li>Tenure (months)</li>
        <li>MonthlyCharges ($)</li>
        <li>TotalCharges ($)</li>
        </ul>
        
        <h4 style='color: #79c0ff; margin-top: 15px;'>🎯 Target Variable:</h4>
        <ul style='color: #c9d1d9; font-weight: 500;'>
        <li>Churn (Yes/No)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h3 class='section-title'>Data Quality</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #0f2818; padding: 20px; border-radius: 8px;'>
    <p style='color: #3fb950; font-weight: bold; font-size: 16px; margin: 5px 0;'>✅ No duplicate records found</p>
    <p style='color: #3fb950; font-weight: bold; font-size: 16px; margin: 5px 0;'>✅ Missing values handled appropriately</p>
    <p style='color: #3fb950; font-weight: bold; font-size: 16px; margin: 5px 0;'>✅ Outliers within normal ranges</p>
    <p style='color: #3fb950; font-weight: bold; font-size: 16px; margin: 5px 0;'>✅ Balanced preprocessing applied</p>
    <p style='color: #3fb950; font-weight: bold; font-size: 16px; margin: 5px 0;'>✅ SMOTE used to handle class imbalance</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE: HELP & INFO
# ==========================================
elif page == "ℹ️ Help & Info":
    st.markdown("<h2 class='section-title'>How It Works</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Process Flow
    
    1. **Enter Customer Data** - Fill in customer information across different tabs
    2. **Submit for Prediction** - Click the predict button to generate results
    3. **Get Insights** - Receive detailed analysis with risk factors and recommendations
    
    ### 🤖 Model Information
    
    - **Algorithm:** Gradient Boosting Classifier
    - **Training Method:** 5-fold cross-validation with class balancing (SMOTE)
    - **Primary Metric:** ROC-AUC Score
    - **Performance:** ~84% accuracy on test data
    
    ### 📊 Key Metrics Explained
    
    - **Churn Probability:** Likelihood (0-100%) that customer will churn
    - **Risk Level:** Classification as High or Low risk
    - **Confidence:** How confident the model is in its prediction
    - **Retention Score:** Opposite of churn probability (1 - churn probability)
    
    ### 💡 Tips for Best Results
    
    - Provide accurate customer information
    - Use actual billing amounts and tenure
    - Select appropriate service packages
    - Review recommendations for retention strategies
    """)
    
    st.markdown("<h2 class='section-title'>About This Project</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    **Capstone Project:** Telecom Customer Churn Prediction
    
    **Developer:** Aatequa Ansari
    
    **Institution:** E&ICT IITG
    
    **Objective:** Develop a machine learning system to predict customer churn and enable proactive retention strategies.
    
    **Technologies Used:**
    - Python, Pandas, NumPy
    - Scikit-learn, XGBoost
    - Streamlit (for interactive UI)
    - Machine Learning (Classification)
    """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #999; font-size: 12px;'>© 2024 Telecom Churn Prediction System | Capstone Project</p>", 
               unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #c9d1d9; font-weight: bold; font-size: 12px;'>Built with ❤️ by Aatequa Ansari | Powered by Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)