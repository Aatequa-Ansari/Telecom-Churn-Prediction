# 🔮 Telecom Customer Churn Prediction

## 📋 Project Overview

A machine learning-powered web application that predicts whether a telecom customer will churn (leave the service). This capstone project uses advanced classification algorithms to identify at-risk customers and provide actionable retention strategies.

**Developer:** Aatequa Ansari  
**Institution:** E&ICT IITG  
**Project Type:** Capstone Project  
**Last Updated:** 2026
---

## 🎯 Objective

The primary objective of this project is to:
- Develop a predictive model to identify customers at risk of churning
- Enable telecom companies to take proactive retention measures
- Provide data-driven insights into customer churn behavior
- Build an intuitive interface for non-technical stakeholders to make predictions

---

## ✨ Key Features

### 🏠 Home Page
- Welcome dashboard with project overview
- Key factors influencing churn (Tenure, Monthly Charges, Contract Type, Internet Service, Tech Support, Personal Factors)
- Quick navigation to prediction tool

### 🎯 Make Prediction
- **Customer Demographics Tab:** Gender, Senior Citizen status, Partner/Dependents, Tenure, Phone Service
- **Services Tab:** Internet Service type, Online Security, Backup, Device Protection, Tech Support, Streaming services, Multiple Lines
- **Billing Tab:** Contract Type, Paperless Billing, Payment Method
- **Usage Tab:** Monthly and Total Charges
- **Real-time Prediction:** Instant ML model prediction with confidence scores
- **Detailed Analysis:**
  - Risk Level (High/Low)
  - Churn Probability (%)
  - Confidence Score
  - Retention Score

### 📊 Risk Assessment
- Identifies specific risk factors for each customer
- Provides tailored recommendations for retention
- Color-coded risk indicators (🔴 High, 🟡 Moderate, 🟢 Low)

### 📊 Dataset Information
- Complete feature documentation
- Data quality metrics
- Dataset statistics and characteristics

### ℹ️ Help & Information
- Model information and performance metrics
- How-to guide for using the application
- Metric explanations

---

## 📊 Dataset Details

### Dataset Statistics
- **Total Records:** 7,032 customer records
- **Total Features:** 19 variables
- **Target Variable:** Churn (Binary: Yes/No)
- **Churn Rate:** ~26.5% (imbalanced dataset)
- **Data Quality:** High - No duplicates, handled missing values

### Feature Categories

#### 👤 Demographic Features (4)
- `Gender` - Customer gender (Male/Female)
- `SeniorCitizen` - Senior citizen status (0/1)
- `Partner` - Has partner (Yes/No)
- `Dependents` - Has dependents (Yes/No)

#### 📞 Service Features (9)
- `PhoneService` - Phone service subscription (Yes/No)
- `MultipleLines` - Multiple phone lines (Yes/No/No phone service)
- `InternetService` - Internet service type (DSL/Fiber optic/No)
- `OnlineSecurity` - Online security add-on (Yes/No/No internet service)
- `OnlineBackup` - Online backup service (Yes/No/No internet service)
- `DeviceProtection` - Device protection plan (Yes/No/No internet service)
- `TechSupport` - Tech support service (Yes/No/No internet service)
- `StreamingTV` - Streaming TV service (Yes/No/No internet service)
- `StreamingMovies` - Streaming movies service (Yes/No/No internet service)

#### 💳 Account Features (3)
- `Contract` - Contract type (Month-to-month/One year/Two year)
- `PaperlessBilling` - Paperless billing (Yes/No)
- `PaymentMethod` - Payment method (Electronic check/Mailed check/Bank transfer/Credit card)

#### 📈 Usage & Billing Features (3)
- `Tenure` - Months as customer (0-72)
- `MonthlyCharges` - Monthly bill amount ($)
- `TotalCharges` - Total charges to date ($)

---

## 🔍 Key Findings

### Top Churn Risk Factors

1. **Contract Type** 📋
   - Month-to-month contracts have significantly higher churn rates
   - Two-year contracts show lowest churn tendency

2. **Tenure** 📅
   - New customers (< 12 months) have extremely high churn risk
   - Churn decreases significantly after 12-month threshold
   - Customers with 24+ months tenure show very low churn

3. **Internet Service Type** 🌐
   - Fiber optic users show highest churn rate
   - DSL service customers have moderate churn
   - Customers without internet service show very low churn

4. **Tech Support** 🛡️
   - Lack of tech support correlates with high churn
   - Customers with tech support are much more likely to stay

5. **Monthly Charges** 💰
   - Higher monthly charges increase churn probability
   - Customers paying >$100/month at higher risk
   - Sweet spot for retention: $50-90/month

6. **Personal Factors** 👤
   - Customers without partners/dependents churn more frequently
   - Family commitment correlates with retention

---

## 🤖 Model Information

### Model Architecture
- **Primary Algorithm:** Logistic Regression (Best performer: 80.24% accuracy)
- **Secondary Algorithm:** Gradient Boosting (80.10% accuracy)
- **Training Approach:** 5-fold Cross-validation with SMOTE
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Preprocessing:** StandardScaler for numeric features, OneHotEncoder for categorical

### Model Performance

#### Best Model: Logistic Regression
- **Accuracy:** 80.24%
- **Precision:** 64.37%
- **Recall:** 57.49%
- **F1-Score:** 60.73%

#### Alternative Model: Gradient Boosting
- **Accuracy:** 80.10%
- **Precision:** 65.16%
- **Recall:** 54.01%
- **F1-Score:** 59.06%

#### Model Comparison (Test Set Performance)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 80.24% | 64.37% | 57.49% | 60.73% |
| Gradient Boosting | 80.10% | 65.16% | 54.01% | 59.06% |
| XGBoost | 77.54% | 58.95% | 51.07% | 54.73% |
| Random Forest | 77.97% | 60.13% | 50.80% | 55.07% |
| KNN | 76.26% | 55.13% | 57.49% | 56.28% |
| Decision Tree | 72.78% | 48.85% | 51.07% | 49.93% |

### Model Output
- **Churn Probability:** Predicted probability of churn (0-100%)
- **Confidence Score:** Model confidence in the prediction
- **Risk Classification:** Binary classification (Churn/No Churn)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone/Download the project**
   ```bash
   cd path/to/Telecome_Customer_Capstone_Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   
   **Option A: Using requirements.txt (Recommended)**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Option B: Manual installation**
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib seaborn imbalanced-learn
   ```

4. **Ensure model file is present**
   - Place `churn_model.pkl` in the project directory
   - Place `Telco_Customer_Churn.csv` in the project directory (for reference)

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Telecome_Customer_Capstone_Project/
├── app.py                          # Main Streamlit application
├── Aatequa_Ansari.ipynb           # Jupyter notebook with model development
├── Telco_Customer_Churn.csv       # Dataset
├── churn_model.pkl                # Trained ML model
├── MODEL_DEVELOPMENT_DOCUMENT.md            # Detailed analysis document
└── README.md                       # This file
```

---

## 💻 Technologies Used

### Data Science & ML
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **SMOTE** - Class imbalance handling

### Visualization & UI
- **Streamlit** - Interactive web application framework
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical visualizations

### Utilities
- **Joblib** - Model serialization
- **Python datetime** - Timestamp generation

---

## 🎨 UI/UX Features

### Dark Theme
- Professional dark color scheme (#0e1117 background)
- Reduced eye strain for extended use
- Accent colors: Blue (#58a6ff), Green (#3fb950), Red (#f85149)

### Responsive Design
- Multi-column layouts that adapt to screen size
- Tab-based organization for cleaner interface
- Organized input fields in logical sections

### Visual Feedback
- Color-coded risk indicators (High=Red, Low=Green)
- Progress indication through styled metrics
- Gradient backgrounds for emphasis
- Box shadows for depth

---

## 📊 How to Use the Application

### Step 1: Navigate to Make Prediction
Click on **"🎯 Make Prediction"** in the sidebar navigation menu

### Step 2: Enter Customer Information
Fill in customer details across 4 tabs:
- **👤 Customer:** Demographics and basic info
- **📞 Services:** Service subscriptions and add-ons
- **💳 Billing:** Contract and payment information
- **📈 Usage:** Charges and tenure

### Step 3: Click Predict
Press the **"🔮 Predict Churn Risk"** button

### Step 4: Review Results
Examine:
- Risk level and churn probability
- Identified risk factors
- Tailored recommendations
- Retention strategies

### Step 5: Take Action
Use insights to develop targeted retention strategies

---

## 💡 Retention Recommendations

### For High-Risk Customers
- ⭐ Offer loyalty discounts or upgrade packages
- 📞 Reach out with personalized service review
- 🛡️ Add tech support to improve satisfaction
- 📋 Encourage upgrade to longer-term contract with incentives
- 👋 Provide enhanced onboarding and support

### For Low-Risk Customers
- ✅ Continue current service strategy
- 📊 Monitor usage patterns for early warning signs
- 📈 Explore upsell opportunities

---

## 📈 Business Impact

### Potential Benefits
- **Reduced Churn Rate:** Identify and retain at-risk customers
- **Increased Revenue:** Fewer lost customers and higher lifetime value
- **Targeted Marketing:** Focus retention efforts on high-risk segments
- **Resource Optimization:** Allocate retention budget efficiently
- **Data-Driven Decisions:** Support strategy with ML insights

### ROI Potential
- Retaining 10% of predicted churners could save millions in annual revenue
- Personalized retention strategies increase effectiveness vs. generic campaigns

---

## 🔒 Data Privacy & Security

- No data is stored on the server
- Predictions are made in real-time without persistence
- No external API calls or data transmission
- GDPR-compliant architecture (no personal data retention)

---

## 📝 Model Limitations

1. **Training Data Dependency:** Model performance based on historical Telco data
2. **Feature Scope:** Limited to available customer attributes
3. **Temporal Dynamics:** Trained on static snapshots; doesn't capture time-series trends
4. **Generalization:** May need retraining for different regions/markets
5. **External Factors:** Cannot account for unforeseen market changes

---

## 🚀 Future Enhancements

- [ ] Time-series model for trend analysis
- [ ] LIME/SHAP for model interpretability
- [ ] Batch prediction capability
- [ ] Dashboard with historical predictions
- [ ] API endpoint for integration
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Real-time model monitoring
- [ ] A/B testing framework for retention strategies

---

## 📚 References & Resources

### Dataset
- Telecom Customer Churn Dataset
- Source: Kaggle / Telecom Provider

### ML Techniques
- Gradient Boosting: https://xgboost.readthedocs.io/
- SMOTE: https://imbalanced-learn.org/
- Cross-validation: https://scikit-learn.org/

### Framework
- Streamlit Documentation: https://streamlit.io/
- Scikit-learn: https://scikit-learn.org/

---

## 📞 Contact & Support

**Developer:** Aatequa Ansari  
**Email:** aatequaansari12@gmail.com  
**Institution:** E&ICT IITG  
**Project:** Capstone Project - Telecom Customer Churn Prediction

---

## 📄 License

© 2026 Aatequa Ansari. Capstone Project - E&ICT IITG.  
All rights reserved.

---

## 🙏 Acknowledgments

- E&ICT IITG for project guidance and resources
- Telecom dataset providers
- Open-source community for ML libraries (scikit-learn, XGBoost, Streamlit)
- All contributors and reviewers

---

## 📊 Summary

This project demonstrates the practical application of machine learning in business analytics. By leveraging advanced classification algorithms and providing an intuitive interface, it enables telecom companies to make data-driven decisions about customer retention. The combination of predictive modeling, interactive UI, and actionable insights makes this a comprehensive solution for churn prediction and customer retention strategy.

**Built with ❤️ by Aatequa Ansari | Powered by Python, Streamlit & Machine Learning**
