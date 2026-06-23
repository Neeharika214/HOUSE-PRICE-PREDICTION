# 🏡 House Price Prediction using Linear Regression

This project is a **Machine Learning model** that predicts house prices using the **Ames Housing Dataset**. It uses **Linear Regression** and evaluates performance using metrics like R² Score and MAPE-based accuracy. The final output also converts predicted prices into **Indian Rupees (₹)** for better understanding.

---

## 📌 Project Features

- Data cleaning using **IQR method (outlier removal)**
- Missing value handling using **median imputation**
- Feature selection for housing prediction
- Linear Regression model training
- Model evaluation using:
  - R² Score
  - Mean Absolute Percentage Error (MAPE)
- Visualization of:
  - Actual vs Predicted prices
  - Model Accuracy comparison
  - R² Score comparison
- Interactive user input system for real-time prediction
- Price conversion from **USD → INR**

---

## 📊 Dataset

- Dataset used: **Ames Housing Dataset**
- File name: `AmesHousing.csv`
- Target variable: `SalePrice`
- Features used:
  - Overall Qual
  - Gr Liv Area
  - Garage Cars
  - Garage Area
  - Total Bsmt SF
  - Year Built
  - Full Bath
  - TotRms AbvGrd

---

## 🧠 Machine Learning Model

- Algorithm: **Linear Regression**
- Library: `scikit-learn`
- Train-Test Split: **80% training / 20% testing**
- Evaluation Metrics:
  - R² Score
  - Accuracy (1 - MAPE)

---

## 📈 Visualizations

The project generates the following plots:

1. 📌 Actual vs Predicted House Prices  
2. 📊 Train vs Test Accuracy Comparison  
3. 📉 R² Score Comparison  

---

## 💱 Currency Conversion

- Prices are converted from **USD to INR**
- Conversion rate used:
