

def test_predict_success(client):
    payload = {
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
        "CashbackAmount": 250.0,
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "recommendations" in data
    assert "customer_segment" in data
    assert "loyalty_score" in data
