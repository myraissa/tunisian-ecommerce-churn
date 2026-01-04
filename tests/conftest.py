# tests/conftest.py
import json
import builtins
import pytest
from fastapi.testclient import TestClient
import importlib


class DummyModel:
    def predict(self, X):
        # retourne 0/1
        return [0]

    def predict_proba(self, X):
        # [proba stay, proba churn]
        return [[0.8, 0.2]]


class DummyScaler:
    def transform(self, X):
        return X


class DummyLabelEncoder:
    def transform(self, values):
        # transforme toutes les catégories en 0
        return [0 for _ in values]


@pytest.fixture()
def client(monkeypatch):
    # 1) mock joblib.load
    import joblib

    def fake_joblib_load(path):
        if "best_model" in path:
            return DummyModel()
        if "scaler" in path:
            return DummyScaler()
        if "label_encoders" in path:
            return {
                "PreferredLoginDevice": DummyLabelEncoder(),
                "PreferredPaymentMode": DummyLabelEncoder(),
                "Gender": DummyLabelEncoder(),
                "PreferedOrderCat": DummyLabelEncoder(),
                "MaritalStatus": DummyLabelEncoder(),
            }
        return None

    monkeypatch.setattr(joblib, "load", fake_joblib_load)

    # 2) mock open('model_metadata.json')
    real_open = builtins.open

    fake_metadata = {
        "model_name": "4_XGBoost",
        "accuracy": 0.95,
        "f1_score": 0.90,
        "roc_auc": 0.97,
        "needs_scaling": False,
        "training_date": "2026-01-04",
        "features": [
            # minimal features list (doit matcher ce que l'API attend)
            "Tenure",
            "CityTier",
            "WarehouseToHome",
            "HourSpendOnApp",
            "NumberOfDeviceRegistered",
            "SatisfactionScore",
            "NumberOfAddress",
            "Complain",
            "OrderAmountHikeFromlastYear",
            "CouponUsed",
            "OrderCount",
            "DaySinceLastOrder",
            "CashbackAmount",
            # engineered examples
            "AvgTimePerDevice",
            "EngagementScore",
            "AvgOrderValue",
            "OrderFrequency",
            "CouponUsageRate",
            "DaysPerOrder",
            "LoyaltyScore",
            "IsInactive",
            "LowSatisfaction",
            "LongDistance",
            "DecliningOrders",
            "CustomerSegment",
            "MultiDevice",
            "TenureSatisfaction",
            "OrderCashbackRatio",
            "ComplaintSatisfaction",
            "GrowthCategory",
            # encoded categoricals
            "PreferredLoginDevice",
            "PreferredPaymentMode",
            "Gender",
            "PreferedOrderCat",
            "MaritalStatus",
        ],
    }

    def fake_open(file, mode="r", *args, **kwargs):
        if file == "model_metadata.json":
            return FakeFile(json.dumps(fake_metadata))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    api_module = importlib.import_module("app.api.main")  
    return TestClient(api_module.app)


class FakeFile:
    def __init__(self, content: str):
        self.content = content
        self._closed = False

    def read(self):
        return self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._closed = True
