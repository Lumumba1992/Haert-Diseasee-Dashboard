# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, auc, 
                             RocCurveDisplay, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; padding:10px 0;">
        <h2>CardioPredict AI</h2>
        <h4>Center for Data Analytics and Modeling - Chuka University</h4>
        <p><b>Authors:</b> Prof. Dennis K. Muriithi | Lumumba W. Victor</p>
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)
# ==========================================
# CUSTOM CSS: INCREASE SIDEBAR FONT SIZE
# ==========================================
st.markdown(
    """
    <style>
    .stSidebar .stRadio [data-baseweb="radio"] label { font-size: 35px !important; }
    .stSidebar .stRadio > div > label { font-size: 30px !important; }
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown h3 { font-size: 25px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 1. CLINICAL RISK STRATIFICATION ENGINE
# ==========================================
def get_clinical_recommendation(prob):
    """
    Maps predicted probability to clinical risk tiers and generates 
    evidence-based medical recommendations based on standard cardiology guidelines.
    """
    if prob <= 0.30:
        return (
            "🟢 LOW RISK (0% - 30%)", 
            "success",
            "**Clinical Action Plan: Routine Care & Primary Prevention**\n\n"
            "- **Diagnostics:** No immediate invasive or advanced testing required.\n"
            "- **Intervention:** Focus on lifestyle maintenance (diet, exercise, smoking cessation).\n"
            "- **Follow-up:** Re-evaluate standard cardiovascular risk factors in 1–3 years, or sooner if new symptoms (e.g., atypical chest pain) develop."
        )
    elif prob <= 0.50:
        return (
            "🟡 MODERATE RISK (31% - 50%)", 
            "info",
            "**Clinical Action Plan: Preventive Action & Non-Invasive Testing**\n\n"
            "- **Diagnostics:** Consider scheduling a non-invasive functional stress test or echocardiogram to rule out ischemia.\n"
            "- **Intervention:** Aggressively manage modifiable risk factors (optimize BP, initiate/statin therapy if indicated, control blood sugar).\n"
            "- **Follow-up:** Schedule a clinical review in 3–6 months to monitor symptom progression."
        )
    elif prob <= 0.70:
        return (
            "🟠 HIGH RISK (51% - 70%)", 
            "warning",
            "**Clinical Action Plan: Urgent Cardiology Evaluation**\n\n"
            "- **Diagnostics:** Prompt referral to cardiology. Consider advanced non-invasive imaging (e.g., CCTA) or diagnostic stress imaging.\n"
            "- **Intervention:** Evaluate the need for guideline-directed medical therapy (GDMT) including antiplatelets, beta-blockers, and high-intensity statins.\n"
            "- **Follow-up:** Urgent outpatient follow-up. If patient is highly symptomatic (e.g., limiting angina), consider acute admission."
        )
    else: # 71% - 100%
        return (
            "🔴 VERY HIGH / CRITICAL RISK (71% - 100%)", 
            "error",
            "**Clinical Action Plan: Immediate Intervention & Invasive Diagnostics**\n\n"
            "- **Diagnostics:** Immediate cardiology consultation required. High suspicion of severe/multi-vessel disease.\n"
            "- **Intervention:** Prepare for potential invasive coronary angiography. Evaluate for revascularization (PCI/Stent or CABG).\n"
            "- **Follow-up:** If patient presents with unstable angina or equivalent symptoms, admit for inpatient observation and immediate workup."
        )

# ==========================================
# 2. LOAD PRE-TRAINED MODELS (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    preprocessor = rf_model.named_steps['preprocessor']
    return preprocessor, rf_model, xgb_model

preprocessor, rf_model, xgb_model = load_models()

# ==========================================
# 3. LOAD DATA FOR EVALUATION
# ==========================================
@st.cache_data
def get_evaluation_data():
    df = pd.read_csv('Heart.csv', na_values=['NA', '?', ''])
    df['AHD'] = df['AHD'].map({'No': 0, 'Yes': 1})
    X = df.drop(columns=['AHD', 'HD'])
    y = df['AHD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    return X, y, X_test, y_test

X_base, y_base, X_test, y_test = get_evaluation_data()
feature_names = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca', 'Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🩺 CardioPredict AI")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to:", [
    "🏠 Dashboard Overview",
    "📊 Model Performance & Charts",
    "🩺 New Patient Prediction",
    "🔍 Explainability (SHAP)"
])

# ==========================================
# 5. PAGE RENDERING
# ==========================================

if page == "🏠 Dashboard Overview":
    st.title("🏠 CardioPredict AI Dashboard")
    st.markdown("""
    ### Clinical Objective
    Predict the presence of **Atherosclerotic Heart Disease (AHD)** to enable early intervention and risk stratification.
    
    - **Target Variable**: 'AHD' (Binary: 0 = No Disease, 1 = Disease Present)
    - **Clinical Priority**: Minimizing **False Negatives** is critical. A missed cardiac risk can lead to delayed treatment and acute events. Therefore, **Recall/Sensitivity** is prioritized.
    - **Technology**: This dashboard deploys ensemble learning models (Random Forest & XGBoost) with built-in Explainable AI (SHAP) and **Automated Clinical Risk Stratification** to ensure transparent, trustworthy clinical decision support.
    """)
    st.info("💡 **Tip**: Navigate to the 'New Patient Prediction' tab to test the inference engine and view automated clinical action plans.")

elif page == "📊 Model Performance & Charts":
    st.title("📊 Model Performance & Visualizations")
    st.markdown("Evaluation on the held-out 20% Test Set (61 patients)")
    
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    def get_metrics(y_true, y_pred, y_prob):
        from sklearn.metrics import precision_recall_curve
        p, r, _ = precision_recall_curve(y_true, y_prob)
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_prob),
            'PR-AUC': auc(r, p)
        }
    
    rf_metrics = get_metrics(y_test, rf_pred, rf_prob)
    xgb_metrics = get_metrics(y_test, xgb_pred, xgb_prob)
    
    metrics_df = pd.DataFrame({'Random Forest': rf_metrics, 'XGBoost': xgb_metrics}).round(4)
    st.dataframe(metrics_df.style.highlight_max(axis=1, color='lightgreen'), use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROC-AUC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        rf_prob = np.asarray(rf_prob).ravel()
        xgb_prob = np.asarray(xgb_prob).ravel()
        RocCurveDisplay.from_predictions(y_test, rf_prob, name="Random Forest", ax=ax_roc)
        RocCurveDisplay.from_predictions(y_test, xgb_prob, name="XGBoost", ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title("Receiver Operating Characteristic")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        
    with col2:
        st.subheader("🎯 Confusion Matrix (Random Forest)")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=ax_cm, cmap='Blues', values_format='d')
        ax_cm.set_title('Random Forest Confusion Matrix')
        st.pyplot(fig_cm)

    st.markdown("---")
    st.subheader("🧬 Feature Importance (XGBoost)")
    importances = xgb_model.named_steps['xgb'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    ax_imp.barh(feat_imp.index, feat_imp.values, color='#2ca02c', edgecolor='black')
    ax_imp.set_title('Top Clinical Predictors (Gain Importance)')
    ax_imp.set_xlabel('Importance Score')
    ax_imp.grid(axis='x', alpha=0.3)
    st.pyplot(fig_imp)

elif page == "🩺 New Patient Prediction":
    st.title("🩺 New Patient Prediction & Risk Stratification")
    st.markdown("Enter the patient's clinical metrics below to generate a risk assessment and automated clinical action plan.")
    
    with st.form("patient_form"):
        st.subheader("Patient Demographics & Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=29, max_value=77, value=55)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            chest_pain = st.selectbox("Chest Pain Type", ["typical", "nontypical", "nonanginal", "asymptomatic"])
        with col2:
            rest_bp = st.number_input("Resting BP (mmHg)", min_value=90, max_value=200, value=130)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=126, max_value=564, value=250)
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        with col3:
            max_hr = st.number_input("Max Heart Rate (bpm)", min_value=71, max_value=202, value=150)
            ex_ang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
            
        st.subheader("Diagnostic Test Results")
        col4, col5, col6 = st.columns(3)
        with col4:
            rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal (0)", "ST-T abnormality (1)", "LV hypertrophy (2)"][x])
            slope = st.selectbox("ST Segment Slope", [1, 2, 3], format_func=lambda x: ["Upsloping (1)", "Flat (2)", "Downsloping (3)"][x-1])
        with col5:
            ca = st.number_input("Major Vessels (Ca)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia (Thal)", ["normal", "fixed", "reversable"])
        with col6:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            model_choice = st.selectbox("Select Prediction Model", ["Random Forest", "XGBoost"])
            
        submitted = st.form_submit_button("🔮 Generate Prediction & Action Plan", type="primary")

    if submitted:
        new_patient = pd.DataFrame([{
            'Age': age, 'Sex': sex, 'ChestPain': chest_pain, 'RestBP': rest_bp,
            'Chol': chol, 'Fbs': fbs, 'RestECG': rest_ecg, 'MaxHR': max_hr,
            'ExAng': ex_ang, 'Oldpeak': oldpeak, 'Slope': slope, 'Ca': ca, 'Thal': thal
        }])
        
        model = rf_model if model_choice == "Random Forest" else xgb_model
        pred_class = model.predict(new_patient)[0]
        pred_prob = model.predict_proba(new_patient)[0][1]
        
        # --- NEW: RISK STRATIFICATION & RECOMMENDATIONS ---
        risk_level, alert_type, recommendation = get_clinical_recommendation(pred_prob)
        
        st.markdown("---")
        st.subheader("📋 Prediction & Risk Stratification Results")
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            # Dynamic Alert Box based on Risk Tier
            if alert_type == "success": st.success(f"### {risk_level}")
            elif alert_type == "info": st.info(f"### {risk_level}")
            elif alert_type == "warning": st.warning(f"### {risk_level}")
            elif alert_type == "error": st.error(f"### {risk_level}")
            
            st.metric("Predicted Probability", f"{pred_prob:.1%}")
            st.caption(f"**Model Used:** {model_choice}")
            
        with col_res2:
            st.subheader("🩺 Automated Clinical Action Plan")
            st.markdown(recommendation)
            
        st.markdown("---")
        st.warning("⚠️ **Clinical Disclaimer:** This AI tool is designed for Clinical Decision Support (CDS) only. It does not replace professional medical judgment. Always correlate AI predictions with the patient's full clinical history, physical examination, and institutional protocols.")
            
        # Store for SHAP tab
        st.session_state['last_patient'] = new_patient
        st.session_state['last_model'] = model
        st.session_state['last_prob'] = pred_prob
        st.session_state['last_pred'] = pred_class
        st.session_state['last_risk'] = risk_level

elif page == "🔍 Explainability (SHAP)":
    st.title("🔍 Explainable AI (SHAP)")
    
    if 'last_patient' not in st.session_state:
        st.warning("⚠️ Please go to the 'New Patient Prediction' tab and generate a prediction first to see the SHAP explanation.")
    else:
        patient = st.session_state['last_patient']
        model = st.session_state['last_model']
        prob = st.session_state['last_prob']
        risk_level = st.session_state['last_risk']
        
        st.markdown(f"### Explaining Prediction for Current Patient")
        st.markdown(f"**Risk Tier:** {risk_level} | **Probability:** {prob:.1%}")
        
        with st.spinner("Computing SHAP values..."):
            X_trans = preprocessor.transform(patient)
            base_model = model.named_steps['rf'] if 'rf' in model.named_steps else model.named_steps['xgb']
            
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_trans)
            
            if isinstance(shap_values, list):
                shap_vals_pos = shap_values[1]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            elif len(shap_values.shape) == 3:
                shap_vals_pos = shap_values[:, :, 1]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:
                shap_vals_pos = shap_values
                base_val = explainer.expected_value
                
            exp = shap.Explanation(
                values=shap_vals_pos[0],
                base_values=float(base_val),
                data=X_trans[0],
                feature_names=feature_names
            )
            
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title("How the model reached its decision", fontsize=14)
            st.pyplot(fig)
            
        st.markdown("""
        ### 🧠 How to read this plot:
        - **Base Value**: The average predicted risk across all patients in the training data.
        - **Red Bars**: Features that **increased** the patient's risk of heart disease (pushing them into a higher risk tier).
        - **Blue Bars**: Features that **decreased** the patient's risk.
        - **Final Value (f(x))**: The model's final raw output, which translates to the probability and risk tier shown on the prediction tab.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & SHAP")

## %
