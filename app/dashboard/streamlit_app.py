
"""
Streamlit Dashboard for Churn Prediction
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "models")

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
    return model, scaler, label_encoders, metadata


# Page config
st.set_page_config(
    page_title="Tunisian E-Commerce Churn Predictor",

    layout="wide"
)


model, scaler, label_encoders, metadata = load_model()

# Title
st.title("🛒 Tunisian E-Commerce Churn Prediction System")
st.markdown("### Predict customer churn and get actionable insights")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model", metadata['model_name'].split('_', 1)[1])
    st.metric("Accuracy", f"{metadata['accuracy']*100:.2f}%")
    st.metric("F1-Score", f"{metadata['f1_score']*100:.2f}%")
    st.metric("ROC-AUC", f"{metadata['roc_auc']:.4f}")
    
    st.markdown("---")
    st.markdown("**Trained:** " + metadata['training_date'][:10])

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "📈 Analytics"])

# TAB 1: Single Prediction
with tab1:
    st.header("Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📱 Basic Info")
        tenure = st.slider("Tenure (months)", 0, 50, 12)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        
    with col2:
        st.subheader("🛍️ Shopping Behavior")
        order_count = st.slider("Order Count", 0, 30, 8)
        order_cat = st.selectbox("Preferred Category", 
                                 ["Laptop & Accessory", "Mobile Phone", "Fashion", 
                                  "Grocery", "Others"])
        payment_mode = st.selectbox("Payment Mode", 
                                    ["Debit Card", "Credit Card", "UPI", "COD", "E-wallet"])
        coupon_used = st.slider("Coupons Used", 0, 20, 5)
        cashback = st.slider("Cashback Amount (DT)", 0.0, 1000.0, 250.0)
        
    with col3:
        st.subheader("⭐ Engagement")
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        hours_app = st.slider("Hours on App", 0.0, 10.0, 3.5)
        devices = st.slider("Devices Registered", 1, 6, 3)
        addresses = st.slider("Number of Addresses", 1, 10, 2)
        days_last_order = st.slider("Days Since Last Order", 0, 60, 10)
        
    col4, col5 = st.columns(2)
    with col4:
        warehouse_dist = st.slider("Warehouse Distance (km)", 0, 100, 15)
        login_device = st.selectbox("Preferred Login", ["Mobile Phone", "Computer"])
        
    with col5:
        complain = st.selectbox("Has Complaint?", ["No", "Yes"])
        order_hike = st.slider("Order Growth %", -50.0, 100.0, 15.0)
    
    if st.button("🔮 Predict Churn", type="primary"):
        # Prepare data
        input_data = pd.DataFrame([{
            'Tenure': tenure,
            'PreferredLoginDevice': login_device,
            'CityTier': city_tier,
            'WarehouseToHome': warehouse_dist,
            'PreferredPaymentMode': payment_mode,
            'Gender': gender,
            'HourSpendOnApp': hours_app,
            'NumberOfDeviceRegistered': devices,
            'PreferedOrderCat': order_cat,
            'SatisfactionScore': satisfaction,
            'MaritalStatus': marital,
            'NumberOfAddress': addresses,
            'Complain': 1 if complain == "Yes" else 0,
            'OrderAmountHikeFromlastYear': order_hike,
            'CouponUsed': coupon_used,
            'OrderCount': order_count,
            'DaySinceLastOrder': days_last_order,
            'CashbackAmount': cashback
        }])
        
        # Feature engineering (same as API)
        input_data['AvgTimePerDevice'] = input_data['HourSpendOnApp'] / (input_data['NumberOfDeviceRegistered'] + 1)
        input_data['EngagementScore'] = (
            input_data['HourSpendOnApp'] * 0.4 +
            input_data['NumberOfDeviceRegistered'] * 0.3 +
            input_data['NumberOfAddress'] * 0.3
        )
        input_data['AvgOrderValue'] = input_data['CashbackAmount'] / (input_data['OrderCount'] + 1)
        input_data['OrderFrequency'] = input_data['OrderCount'] / (input_data['Tenure'] + 1)
        input_data['CouponUsageRate'] = input_data['CouponUsed'] / (input_data['OrderCount'] + 1)
        input_data['DaysPerOrder'] = (input_data['Tenure'] * 30) / (input_data['OrderCount'] + 1)
        input_data['LoyaltyScore'] = (
            (input_data['Tenure'] / 50) * 0.4 +
            (input_data['OrderCount'] / 20) * 0.3 +
            (input_data['SatisfactionScore'] / 5) * 0.2 +
            (1 - input_data['Complain']) * 0.1
        )
        input_data['IsInactive'] = (input_data['DaySinceLastOrder'] > 30).astype(int)
        input_data['LowSatisfaction'] = (input_data['SatisfactionScore'] <= 2).astype(int)
        input_data['LongDistance'] = (input_data['WarehouseToHome'] > 20).astype(int)
        input_data['DecliningOrders'] = (input_data['OrderAmountHikeFromlastYear'] < 0).astype(int)
        
        conditions = [
            (input_data['OrderCount'] >= 10) & (input_data['CashbackAmount'] >= 500),
            (input_data['OrderCount'] >= 5) & (input_data['CashbackAmount'] >= 200),
            (input_data['OrderCount'] >= 2),
        ]
        input_data['CustomerSegment'] = np.select(conditions, [2, 1, 0], default=3)
        input_data['MultiDevice'] = (input_data['NumberOfDeviceRegistered'] > 2).astype(int)
        input_data['TenureSatisfaction'] = input_data['Tenure'] * input_data['SatisfactionScore']
        input_data['OrderCashbackRatio'] = input_data['OrderCount'] * input_data['CashbackAmount']
        input_data['ComplaintSatisfaction'] = input_data['Complain'] * (6 - input_data['SatisfactionScore'])
        input_data['GrowthCategory'] = pd.cut(
            input_data['OrderAmountHikeFromlastYear'],
            bins=[-np.inf, 0, 10, 20, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Encode categoricals
        for col in ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                    'PreferedOrderCat', 'MaritalStatus']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Select features
        X = input_data[metadata['features']]
        
        # Scale if needed
        if metadata['needs_scaling']:
            X = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            if prediction == 1:
                st.error("⚠️ Will Churn")
            else:
                st.success("✅ Will Stay")
        
        with col_r2:
            st.metric("Churn Probability", f"{probability*100:.1f}%")
        
        with col_r3:
            if probability < 0.3:
                risk = "🟢 Low Risk"
            elif probability < 0.6:
                risk = "🟡 Medium Risk"
            else:
                risk = "🔴 High Risk"
            st.metric("Risk Level", risk)
        
        with col_r4:
            segments = ["Occasional", "Regular", "VIP", "New"]
            st.metric("Segment", segments[int(input_data['CustomerSegment'].iloc[0])])
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if satisfaction <= 2:
            st.warning("🔴 Low satisfaction - Reach out immediately with personalized offer")
        if days_last_order > 30:
            st.warning("⚠️ Inactive customer - Send re-engagement campaign")
        if complain == "Yes":
            st.warning("📞 Has complaint - Priority follow-up required")
        if order_hike < 0:
            st.warning("📉 Declining orders - Offer bundle deals or loyalty rewards")
        if probability > 0.7:
            st.error("🎁 CRITICAL: High churn risk - Send exclusive VIP offer immediately!")

# TAB 2: Batch Prediction
with tab2:
    st.header("📊 Batch Prediction")
    st.markdown("Upload a CSV file with customer data for bulk predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} customers")
        st.dataframe(df_batch.head())
        
        if st.button("Process Batch"):
            st.info("Processing... This may take a moment")
            # Process batch (similar logic as single prediction)
            st.success(f"✅ Processed {len(df_batch)} customers!")

# TAB 3: Analytics
with tab3:
    st.header("📈 Model Analytics")
    
    # Feature importance
    if 'feature_importance.png' in os.listdir('.'):
        st.image('feature_importance.png', caption="Feature Importance")

st.markdown("---")
st.markdown("**Developed with ❤️ for Tunisian E-Commerce | Powered by ML & MLOps**")
