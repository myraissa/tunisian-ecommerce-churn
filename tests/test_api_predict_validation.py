
def test_predict_missing_field_returns_422(client):
    payload = {
        # Tenure missing on purpose
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
    assert r.status_code == 422
