# 📈 Stock Price Prediction Pipeline: Short-Term Forecasts with `yfinance`

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![yfinance](https://img.shields.io/badge/yfinance-API-yellow)](https://github.com/ranaroussi/yfinance)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-informational?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

> **"Bridging the gap between historical market data and predictive analytics using a robust machine learning pipeline for short-term stock price forecasting."**

---

## 📖 1. Project Overview
This project implements an end-to-end Machine Learning pipeline designed to predict the **next day's closing price** of a specific stock (defaulting to Apple Inc., ticker: `AAPL`). In the financial sector, short-term price forecasting is a critical component for algorithmic trading, risk management, and portfolio optimization.

By leveraging the `yfinance` API, this project demonstrates the ability to ingest real-world financial time-series data, transform it into a structured format for regression modeling, and evaluate predictive accuracy against historical benchmarks.

---

## 🎯 2. Task Objective
The primary goal of this internship project was to:
1. **Develop an automated data retrieval system** using the Yahoo Finance API.
2. **Apply feature engineering** techniques to transform time-series data into a supervised learning problem.
3. **Compare multiple regression architectures** (Linear vs. Ensemble) to identify the optimal predictor.
4. **Export a production-ready model artifact** for future integration into financial dashboards.

---

## 📊 3. Dataset Section
The data is dynamically fetched using the **yfinance** library, ensuring up-to-date analysis.

| Feature | Description |
| :--- | :--- |
| **Open / Close** | Opening and closing prices for the trading day. |
| **High / Low** | The day's price range. |
| **Adj Close** | Adjusted closing price (accounting for splits/dividends). |
| **Volume** | Total shares traded. |
| **Lags (1, 3, 5)** | Engineered features representing historical price points. |

- **Source**: Yahoo Finance API.
- **Period**: 5 Years of daily historical data.
- **Target Variable**: `Target_Next_Close` (Next day's closing price).
- **Preprocessing**: Handling MultiIndex columns, timezone-naive conversions, and chronological sorting.

---

## 🛠️ 4. Tech Stack
- **Python**: Primary engineering language.
- **yfinance**: Real-time financial data ingestion.
- **Pandas & NumPy**: Data manipulation and feature matrix construction.
- **Scikit-Learn**: Implementation of `LinearRegression`, `RandomForestRegressor`, and evaluation suite.
- **Matplotlib & Seaborn**: High-resolution financial plotting.
- **Joblib**: Model serialization and persistence.

---

## 🔄 5. Machine Learning Workflow
This project follows a rigorous AI engineering lifecycle:

1. **Ingestion & Fallback**: A dual-loading strategy attempts API retrieval first, with a local CSV fallback to ensure pipeline resilience.
2. **Cleaning & Standardization**: Automating the flattening of MultiIndex headers and ensuring numeric consistency.
3. **Feature Engineering (Time-Series Transformation)**:
    - **Target Alignment**: Shifting the 'Close' price by -1 to create a supervised target.
    - **Lagged Features**: Using historical windows to provide the model with context of recent trends.
4. **Validation Strategy**: Implementing a split (65% Train / 15% Validation / 20% Test) that respects temporal ordering to prevent data leakage.
5. **Architectural Comparison**: Training a baseline Linear model alongside a non-linear Random Forest to capture complex market dependencies.
6. **Evaluation & Artifact Saving**: Computing RMSE and R² scores, followed by serializing the best model for deployment.

---

## 🤖 6. Models Used
### **Linear Regression (Baseline)**
- **Why**: Used to establish a performance floor. It assumes a linear relationship between past prices and future value.
- **Pros**: Extremely fast, interpretable, and less prone to overfitting on small datasets.

### **Random Forest Regressor (Ensemble)**
- **Why**: To capture non-linear relationships and interactions between features (e.g., volume spikes and price changes).
- **Pros**: Robust to outliers and provides a more flexible decision boundary.

---

## 📏 7. Evaluation Metrics
We measure performance using industry-standard regression metrics:
- **MAE (Mean Absolute Error)**: Average dollar amount the prediction deviates from the actual close.
- **RMSE (Root Mean Squared Error)**: Penalizes larger prediction errors more heavily—critical for high-risk financial tasks.
- **R² Score**: Indicates the proportion of variance explained by the model features.

---

## 💡 8. Key Results & Findings
- **Trend Dependency**: The models showed a high dependency on the immediate prior day's price, confirming the "Random Walk" nature of short-term movements.
- **Model Comparison**: While Random Forest captures volatility better, the Linear model remains competitive in steady-state markets due to its lower variance.
- **Generalization**: The pipeline effectively predicts the general direction of the stock, though high-frequency volatility remains a significant challenge for daily-interval models.

---

## 🎨 9. Visualizations
The project generates several key insights:
- **Price History**: Visualizing the 5-year growth and volatility of the target ticker.
- **Feature Correlation**: Identifying which metrics (e.g., Volume vs. Adj Close) influence the next day's movement.
- **Actual vs. Predicted**: Line charts comparing model forecasts against real-world test data to visually assess drift.

---

## 🛡️ 10. Responsible AI & Ethics
- **Financial Disclaimer**: This model is for **educational and research purposes only**. It is not intended for real-world financial advice or live trading.
- **Model Limitations**: Stock markets are influenced by external "black swan" events (news, geopolitics) that historical price data alone cannot account for.
- **Transparency**: The use of R² and RMSE provides a realistic view of model performance, highlighting that prediction is never 100% certain.

---

## 📂 11. Project Structure
```text
Future Stock Prices Prediction/
│── yfinance-stock-prediction.ipynb    # Full ML Pipeline
│── stock_next_close_model.joblib      # Serialized Model
│── README.md                          # Documentation
```

---

## 🚀 12. Installation & Usage
1.  **Install requirements**:
   ```bash
   pip install yfinance pandas scikit-learn matplotlib seaborn joblib
   ```
2. **Execute**: Open the Jupyter notebook and run all cells to fetch the latest data and train the model.

---

## 🔮 13. Future Improvements
- **Sentiment Analysis**: Integrating news headlines using NLP to adjust forecasts.
- **Deep Learning**: Moving to LSTM (Long Short-Term Memory) networks for better sequential pattern recognition.
- **Hyperparameter Tuning**: Using `GridSearchCV` to optimize the Random Forest forest depth and estimators.

---

## 🏁 14. Conclusion
This project successfully demonstrates a structured approach to financial time-series forecasting. By combining automated data ingestion with rigorous feature engineering and ensemble modeling, we have built a foundation for quantitative analysis. This workflow highlights the importance of data quality and validation strategy in the volatile world of finance.

---
**Developed during the AI/ML Internship at 𝐃𝐞𝐯𝐞𝐥𝐨𝐩𝐞𝐫𝐬𝐇𝐮𝐛 𝐂𝐨𝐫𝐩𝐨𝐫𝐚𝐭𝐢𝐨𝐧 .**
