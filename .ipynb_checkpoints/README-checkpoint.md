# Bankruptcy Prediction Using Machine Learning

This repository contains a complete machine learning project to predict company bankruptcy using financial ratios.

## 🔧 Tech Stack
- Python
- Pandas
- Scikit-learn
- RandomForest
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

Bankruptcy_Prediction/
│
├── data/
│ └── bankruptcy_data.csv
│
├── models/
│ └── model.pkl
│
├── notebooks/
│ └── Bankruptcy_model.ipynb
│
├── app.py
├── requirements.txt
└── README.md


## 🚀 How to Run

pip install -r requirements.txt
jupyter notebook
streamlit run app/app.py


## 🎯 Goal
Use financial ratios to classify companies as bankrupt or not.

## 📊 Model Highlights

- **Algorithm:** RandomForestClassifier  
- **Handling Imbalance:** class_weight="balanced"  
- **Scaling:** StandardScaler  
- **Evaluation Metrics:**
  - ROC-AUC ≈ 0.94  
  - High overall accuracy  
  - Improved minority-class recall after threshold tuning  
- **Output:** bankruptcy probability + predicted class  


## 📈 Results

The model performs reliably on highly imbalanced data:
- Captures a majority of high-risk (bankrupt) companies  
- Maintains stable precision and accuracy  
- Generates probability scores for ranking company risk  

A `predictions.csv` file is created from test-set evaluation.


## 🧪 Features Used

The model uses key financial ratios including:
- Profitability ratios  
- Liquidity ratios  
- Leverage & solvency metrics  
- Cash-flow indicators  
- Growth and efficiency ratios  

## 🔮 Future Improvements

- Add hyperparameter tuning (GridSearch/Optuna)  
- Add Precision–Recall curves  
- Experiment with XGBoost / LightGBM  
- Deploy to Streamlit Cloud  

