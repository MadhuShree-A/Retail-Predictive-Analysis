
# 🛍️ Retail Predictive Analytics Suite

> End-to-End Machine Learning Project for Retail Analytics  
> Sales Forecasting • Customer Segmentation • Churn Prediction • Power BI Dashboard

---

##  Project Overview

This project analyzes retail transaction data to generate actionable business insights using machine learning and business intelligence tools.

### 🔹 Key Modules

- 📈 **Sales Forecasting** – ARIMA and Prophet models  
- 👥 **Customer Segmentation** – RFM analysis + K-Means clustering  
- 🔍 **Churn Prediction** – ML models with SHAP explainability  
- 📊 **Power BI Dashboard** – Interactive reporting  

---

## 📂 Project Structure

retail-predictive-analytics/
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   └── processed/
│       ├── clean_retail.csv
│       ├── monthly_sales.csv
│       ├── daily_sales.csv
│       └── rfm_features.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_sales_forecasting.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_churn_prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting_models.py
│   ├── segmentation_models.py
│   ├── churn_models.py
│   ├── run_all.py
│   └── utils.py
│
├── models/
│   ├── arima_model.pkl
│   ├── prophet_model.pkl
│   ├── kmeans_model.pkl
│   └── churn_model_best.pkl
│
├── outputs/
│   ├── forecasting/
│   ├── segmentation/
│   ├── churn/
│   └── powerbi/
│
├── powerbi/
│   └── retail_analytics_dashboard.pbix
│
├── reports/
│   ├── project_documentation.md
│   └── business_insights.md
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🛠 Technology Stack

**Programming:** Python 3.11  
**Data Analysis:** pandas, numpy  
**Machine Learning:** scikit-learn, xgboost, statsmodels, prophet  
**Explainability:** shap  
**Visualization:** matplotlib, seaborn, plotly  
**Dashboard:** Power BI  

---

## 📊 Dataset

Online Retail II Dataset (UCI Repository)

- ~1,067,000 transactions  
- ~5,900 customers  
- ~4,600 products  
- 43 countries  
- Date Range: Dec 2009 – Dec 2011  

Kaggle Link:  
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

---

## ⚙️ Installation

### Clone Repository

git clone https://github.com/yourusername/retail-predictive-analytics.git  
cd retail-predictive-analytics

### Create Virtual Environment

Windows:
python -m venv venv  
venv\Scripts\activate  

macOS/Linux:
python -m venv venv  
source venv/bin/activate  

### Install Dependencies

pip install -r requirements.txt

---

## 🚀 Usage

Run notebooks in order:

1. 01_data_cleaning.ipynb  
2. 02_eda_analysis.ipynb  
3. 03_sales_forecasting.ipynb  
4. 04_customer_segmentation.ipynb  
5. 05_churn_prediction.ipynb  

Or run full pipeline:

python src/run_all.py

---

## 📊 Power BI Dashboard

Open:

powerbi/retail_analytics_dashboard.pbix

Dashboard Includes:
- Monthly Sales Trends  
- Forecast Comparison  
- Customer Segments  
- Churn Risk Distribution  
- High-Risk Customers  

---

## 📈 Model Performance

Sales Forecasting: RMSE, MAE, MAPE  
Segmentation: Elbow Method, Silhouette Score  
Churn Prediction: ROC-AUC, Confusion Matrix, SHAP  

---

## 👩‍💻 Author

Madhushree A  
Machine Learning & Data Analytics Enthusiast  

---

## 📜 License

MIT License
