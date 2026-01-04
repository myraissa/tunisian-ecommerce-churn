"""
Test Cases for Churn Prediction API
Run this script to test both churned and non-churned customer profiles

Usage:
1. Start API: uvicorn api:app --reload
2. Run this: python test_customers.py
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/predict"

def print_header(title, color="cyan"):
    colors = {
        "cyan": "\033[96m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "end": "\033[0m"
    }
    print(f"\n{colors.get(color, '')}" + "=" * 60)
    print(f"{title}")
    print("=" * 60 + colors["end"])

def test_customer(customer_data, test_name):
    """Test a single customer prediction"""
    try:
        response = requests.post(API_URL, json=customer_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API!")
        print("Make sure the API is running:")
        print("  1. Open terminal")
        print("  2. cd D:\\Churn_Prediction\\Churn_Prediction")
        print("  3. uvicorn api:app --reload")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

# ============================================
# TEST CASE 1: HIGH RISK - LIKELY TO CHURN 🔴
# ============================================
print_header("🔴 TEST CASE 1: HIGH-RISK CUSTOMER (Likely to CHURN)", "red")

churned_customer = {
    "Tenure": 2,
    "PreferredLoginDevice": "Computer",
    "CityTier": 3,
    "WarehouseToHome": 35,
    "PreferredPaymentMode": "COD",
    "Gender": "Female",
    "HourSpendOnApp": 0.5,
    "NumberOfDeviceRegistered": 1,
    "PreferedOrderCat": "Others",
    "SatisfactionScore": 1,
    "MaritalStatus": "Single",
    "NumberOfAddress": 1,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": -20.0,
    "CouponUsed": 0,
    "OrderCount": 2,
    "DaySinceLastOrder": 45,
    "CashbackAmount": 50.0
}

print("\n📊 Customer Profile:")
print(f"  - Tenure: {churned_customer['Tenure']} months (NEW)")
print(f"  - Satisfaction: {churned_customer['SatisfactionScore']}/5 ⚠️ VERY LOW")
print(f"  - Complaint: {'YES ⚠️' if churned_customer['Complain'] else 'NO'}")
print(f"  - Days Since Last Order: {churned_customer['DaySinceLastOrder']} ⚠️ INACTIVE")
print(f"  - Order Growth: {churned_customer['OrderAmountHikeFromlastYear']}% ⚠️ DECLINING")

result1 = test_customer(churned_customer, "Test 1")

if result1:
    print("\n🎯 PREDICTION RESULT:")
    print(f"  ├─ Prediction: {result1['churn_prediction']}")
    print(f"  ├─ Churn Probability: {result1['churn_probability']*100:.1f}%")
    print(f"  ├─ Risk Level: {result1['risk_level']}")
    print(f"  ├─ Customer Segment: {result1['customer_segment']}")
    print(f"  └─ Loyalty Score: {result1['loyalty_score']}")
    
    if result1['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in result1['recommendations']:
            print(f"  • {rec}")

# ============================================
# TEST CASE 2: LOW RISK - LIKELY TO STAY ✅
# ============================================
print_header("✅ TEST CASE 2: LOYAL CUSTOMER (Likely to STAY)", "green")

loyal_customer = {
    "Tenure": 24,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 1,
    "WarehouseToHome": 8,
    "PreferredPaymentMode": "Credit Card",
    "Gender": "Male",
    "HourSpendOnApp": 5.5,
    "NumberOfDeviceRegistered": 4,
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 5,
    "MaritalStatus": "Married",
    "NumberOfAddress": 3,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 25.0,
    "CouponUsed": 12,
    "OrderCount": 18,
    "DaySinceLastOrder": 3,
    "CashbackAmount": 800.0
}

print("\n📊 Customer Profile:")
print(f"  - Tenure: {loyal_customer['Tenure']} months (LOYAL)")
print(f"  - Satisfaction: {loyal_customer['SatisfactionScore']}/5 ✅ EXCELLENT")
print(f"  - Complaint: {'YES' if loyal_customer['Complain'] else 'NO ✅'}")
print(f"  - Days Since Last Order: {loyal_customer['DaySinceLastOrder']} ✅ VERY ACTIVE")
print(f"  - Order Growth: +{loyal_customer['OrderAmountHikeFromlastYear']}% ✅ GROWING")

result2 = test_customer(loyal_customer, "Test 2")

if result2:
    print("\n🎯 PREDICTION RESULT:")
    print(f"  ├─ Prediction: {result2['churn_prediction']}")
    print(f"  ├─ Churn Probability: {result2['churn_probability']*100:.1f}%")
    print(f"  ├─ Risk Level: {result2['risk_level']}")
    print(f"  ├─ Customer Segment: {result2['customer_segment']}")
    print(f"  └─ Loyalty Score: {result2['loyalty_score']}")
    
    if result2['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in result2['recommendations']:
            print(f"  • {rec}")
    else:
        print("\n💡 No immediate action needed - customer is healthy! ✅")

# Summary
print_header("📋 SUMMARY", "yellow")
print("  Test Case 1: Should predict HIGH churn risk (70-90%)")
print("  Test Case 2: Should predict LOW churn risk (10-30%)")
print("\n" + "=" * 60)