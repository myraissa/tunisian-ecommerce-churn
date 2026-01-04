
"""
FastAPI Application for Churn Prediction
Deploy: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List
import json

# Load model and preprocessing
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

needs_scaling = metadata['needs_scaling']
expected_features = metadata['features']

# Initialize FastAPI
app = FastAPI(
    title="Tunisian E-Commerce Churn Prediction API",
    description="Predict customer churn using ML models",
    version="1.0.0"
)

# Define input schema
class CustomerData(BaseModel):
    Tenure: float
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: float
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: int
    OrderCount: int
    DaySinceLastOrder: int
    CashbackAmount: float
    
    class Config:
        schema_extra = {
            "example": {
                "Tenure": 12,
                "PreferredLoginDevice": "Mobile Phone",
                "CityTier": 1,
                "WarehouseToHome": 15.0,
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Male",
                "HourSpendOnApp": 3.5,
                "NumberOfDeviceRegistered": 3,
                "PreferedOrderCat": "Laptop & Accessory",
                "SatisfactionScore": 3,
                "MaritalStatus": "Single",
                "NumberOfAddress": 2,
                "Complain": 0,
                "OrderAmountHikeFromlastYear": 15.0,
                "CouponUsed": 5,
                "OrderCount": 8,
                "DaySinceLastOrder": 10,
                "CashbackAmount": 250.0
            }
        }

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as training"""
    
    # Engagement features
    df['AvgTimePerDevice'] = df['HourSpendOnApp'] / (df['NumberOfDeviceRegistered'] + 1)
    df['EngagementScore'] = (
        df['HourSpendOnApp'] * 0.4 +
        df['NumberOfDeviceRegistered'] * 0.3 +
        df['NumberOfAddress'] * 0.3
    )
    
    # Purchase behavior
    df['AvgOrderValue'] = df['CashbackAmount'] / (df['OrderCount'] + 1)
    df['OrderFrequency'] = df['OrderCount'] / (df['Tenure'] + 1)
    df['CouponUsageRate'] = df['CouponUsed'] / (df['OrderCount'] + 1)
    df['DaysPerOrder'] = (df['Tenure'] * 30) / (df['OrderCount'] + 1)
    
    # Loyalty
    df['LoyaltyScore'] = (
        (df['Tenure'] / 50) * 0.4 +  # Assuming max tenure ~50
        (df['OrderCount'] / 20) * 0.3 +  # Assuming max orders ~20
        (df['SatisfactionScore'] / 5) * 0.2 +
        (1 - df['Complain']) * 0.1
    )
    df['IsInactive'] = (df['DaySinceLastOrder'] > 30).astype(int)
    df['LowSatisfaction'] = (df['SatisfactionScore'] <= 2).astype(int)
    
    # Distance
    df['LongDistance'] = (df['WarehouseToHome'] > 20).astype(int)  # Assuming median ~20
    
    # Growth
    df['DecliningOrders'] = (df['OrderAmountHikeFromlastYear'] < 0).astype(int)
    
    # Segments
    conditions = [
        (df['OrderCount'] >= 10) & (df['CashbackAmount'] >= 500),
        (df['OrderCount'] >= 5) & (df['CashbackAmount'] >= 200),
        (df['OrderCount'] >= 2),
    ]
    df['CustomerSegment'] = np.select(conditions, [2, 1, 0], default=3)  # Encoded
    
    df['MultiDevice'] = (df['NumberOfDeviceRegistered'] > 2).astype(int)
    
    # Interactions
    df['TenureSatisfaction'] = df['Tenure'] * df['SatisfactionScore']
    df['OrderCashbackRatio'] = df['OrderCount'] * df['CashbackAmount']
    df['ComplaintSatisfaction'] = df['Complain'] * (6 - df['SatisfactionScore'])
    
    # Growth category (encoded)
    df['GrowthCategory'] = pd.cut(
        df['OrderAmountHikeFromlastYear'],
        bins=[-np.inf, 0, 10, 20, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    return df

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Tunisian E-Commerce Churn Prediction API",
        "version": "1.0.0",
        "model": metadata['model_name'],
        "accuracy": f"{metadata['accuracy']*100:.2f}%",
        "f1_score": f"{metadata['f1_score']*100:.2f}%"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_churn(customer: CustomerData) -> Dict:
    """
    Predict churn probability for a customer
    """
    try:
        # Convert to dataframe
        data = pd.DataFrame([customer.dict()])
        
        # Encode categoricals
        for col in ['PreferredLoginDevice', 'PreferredPaymentMode', 
                    'Gender', 'PreferedOrderCat', 'MaritalStatus']:
            if col in label_encoders:
                try:
                    data[col] = label_encoders[col].transform(data[col])
                except:
                    # Handle unseen categories
                    data[col] = 0
        
        # Engineer features
        data = engineer_features(data)
        
        # Ensure all expected features present
        for feat in expected_features:
            if feat not in data.columns:
                data[feat] = 0
        
        # Select features in correct order
        X = data[expected_features]
        
        # Scale if needed
        if needs_scaling:
            X = scaler.transform(X)
        
        # Predict
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        # Risk level
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.6:
            risk = "Medium"
        else:
            risk = "High"
        
        # Recommendations
        recommendations = []
        if customer.SatisfactionScore <= 2:
            recommendations.append("🔴 Low satisfaction - reach out immediately")
        if customer.DaySinceLastOrder > 30:
            recommendations.append("⚠️ Inactive customer - send engagement campaign")
        if customer.Complain == 1:
            recommendations.append("📞 Has complaint - follow up required")
        if customer.OrderAmountHikeFromlastYear < 0:
            recommendations.append("📉 Declining orders - offer incentives")
        if probability > 0.7:
            recommendations.append("🎁 High churn risk - send exclusive offer NOW")
        
        return {
            "churn_prediction": "Will Churn" if prediction == 1 else "Will Stay",
            "churn_probability": round(probability, 4),
            "risk_level": risk,
            "recommendations": recommendations,
            "customer_segment": ["Occasional", "Regular", "VIP", "New"][
                int(data['CustomerSegment'].iloc[0])
            ],
            "loyalty_score": round(float(data['LoyaltyScore'].iloc[0]), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(customers: List[CustomerData]) -> List[Dict]:
    """
    Predict churn for multiple customers
    """
    results = []
    for customer in customers:
        try:
            result = predict_churn(customer)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return results

@app.get("/model_info")
def model_info():
    """Get model information"""
    return metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
