# Tunisian E-Commerce Churn Prediction

## Project Overview

This project implements a complete MLOps pipeline for customer churn prediction, including:
- Data preprocessing and feature engineering
- Multiple ML model training and evaluation
- REST API deployment with FastAPI
- Interactive Streamlit dashboard
- Comprehensive model analytics

### Key Features

- **High Accuracy Models**: Achieved 97.7% accuracy with XGBoost
- **Real-time Predictions**: FastAPI-based REST API for instant predictions
- **Interactive Dashboard**: Streamlit interface for business users
- **Batch Processing**: Support for bulk customer predictions
- **Actionable Insights**: Automated recommendations based on churn risk

## Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 92.3% | 91.8% | 0.9654 |
| Random Forest | 97.3% | 97.1% | 0.9956 |
| Gradient Boosting | 96.7% | 96.5% | 0.9942 |
| **XGBoost** | **97.7%** | **97.5%** | **0.9968** |
| LightGBM | 97.4% | 97.3% | 0.9962 |

## Results (overview)
The confusion matrices show good performance, especially with XGBoost and LightGBM.


- XGBoost: TN=932, FP=4, FN=10, TP=180 (excellent compromis) [Image](assets/cm_4_XGBoost.png)  
- LightGBM: TN=931, FP=5, FN=12, TP=178 [Image](assets/cm_5_LightGBM.png)  
- Random Forest: TN=933, FP=3, FN=27, TP=163 [Image](assets/cm_2_Random_Forest.png)

Feature importance (XGBoost): DaysPerOrder, Complain, OrderCount are among the most influential [Image](assets/feature_importance.png)


## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/myraissa/tunisian-ecommerce-churn.git
cd tunisian-ecommerce-churn
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

#### 1. Run the API

```bash
cd api
uvicorn api:app --reload
```

Access the API documentation at `http://localhost:8000/docs`

#### 2. Launch the Dashboard

```bash
cd dashboard
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501`

## Usage

### API Endpoints

**Single Prediction**
```python
import requests

data = {
    "Tenure": 12,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 1,
    "Gender": "Male",
    "SatisfactionScore": 3,
    "OrderCount": 8,
    # ... other features
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

**Response Example**
```json
{
    "churn_prediction": "Will Stay",
    "churn_probability": 0.2345,
    "risk_level": "Low",
    "recommendations": [
        "✅ Customer shows healthy engagement"
    ],
    "customer_segment": "Regular",
    "loyalty_score": 0.65
}
```

### Dashboard Features

- **Single Customer Prediction**: Input customer data and get instant predictions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Analytics**: View model performance metrics and feature importance
- **Risk Assessment**: Visual gauge charts and risk categorization
- **Actionable Recommendations**: Automated suggestions based on customer profile

## Feature Engineering

The model uses 30+ engineered features including:

- **Engagement Metrics**: AvgTimePerDevice, EngagementScore
- **Purchase Behavior**: OrderFrequency, AvgOrderValue, CouponUsageRate
- **Loyalty Indicators**: LoyaltyScore, TenureSatisfaction
- **Risk Flags**: IsInactive, LowSatisfaction, DecliningOrders
- **Customer Segments**: Automated segmentation (New, Occasional, Regular, VIP)

## Key Insights

Based on feature importance analysis:

1. **DaysPerOrder** - Most critical predictor of churn
2. **Complain** - Customer complaints strongly indicate churn risk
3. **OrderCount** - Order frequency correlates with retention
4. **MaritalStatus** - Demographic factor influencing churn
5. **SatisfactionScore** - Direct correlation with loyalty

## Model Training

To retrain the model with new data:

```bash
python src/model_training.py --data data/processed/train.csv
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Deployment

### Docker Deployment (Coming Soon)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Cloud Deployment Options

- AWS Elastic Beanstalk
- Google Cloud Run
- Azure App Service
- Heroku

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

- **Mariem AISSA** - Initial work - [GitHub](https://github.com/myraissa)

## Acknowledgments

- Tunisian E-Commerce dataset providers
- Scikit-learn and XGBoost communities
- FastAPI and Streamlit frameworks


---

**Developed with ❤️ for Tunisian E-Commerce | Powered by ML & MLOps**