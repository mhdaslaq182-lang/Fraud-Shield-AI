# 🛡️ AI Fraud Detection System
### 🇱🇰 Sri Lanka Edition

> Final Year Project — CG02 Group 07
> AI-powered fraud detection for Sri Lankan banking system

---

## 📋 Project Overview

This system detects multiple types of banking fraud in real-time
using Machine Learning, Face Recognition, and Graph Analysis.

### Fraud Types Detected
| Type | Description |
|------|-------------|
| 💳 Credit Card Fraud | Unauthorized card transactions |
| 💰 Money Laundering | Suspicious circular transfers |
| 👤 Account Takeover | Unauthorized account access |

---

## 🏗️ System Architecture
banking_fraud_detection/
├── data_generator.py   → Sri Lankan banking dataset (LKR)
├── features.py         → 18 engineered features
├── models.py           → 4 ML models
├── evaluate.py         → Model evaluation + charts
├── predict.py          → Single transaction checker
├── monitor.py          → Real-time transaction monitor
├── alert_system.py     → Email + log fraud alerts
├── explainer.py        → SHAP explainability
├── face_auth.py        → Face recognition login
├── aml_graph.py        → AML network analysis
├── app.py              → Streamlit web dashboard
└── main.py             → Master pipeline
---

## 🤖 ML Models Used

| Model | Type | Accuracy |
|-------|------|----------|
| Random Forest | Supervised | 100% |
| Gradient Boosting | Supervised | 100% |
| Logistic Regression | Supervised | 100% |
| Isolation Forest | Unsupervised | 100% |

---

## 🏦 Sri Lankan Banks Covered
- Bank of Ceylon (BOC)
- Peoples Bank
- Commercial Bank
- Sampath Bank
- HNB (Hatton National Bank)
- NSB (National Savings Bank)
- Seylan Bank
- NTB (Nations Trust Bank)
- DFCC Bank
- Pan Asia Bank
- And 5 more...

## 🛒 Sri Lankan Merchants
Daraz.lk, PickMe, Dialog, Keells, Cargills and 25 more

## 🌍 Sri Lankan Cities
89 cities across all 9 provinces

---

## ⚙️ Installation

```bash
# 1. Clone project
cd ~/banking_fraud_detection

# 2. Create environment
conda create -n fraud_env python=3.12 -y
conda activate fraud_env

# 3. Install libraries
pip install numpy pandas matplotlib seaborn scikit-learn
pip install streamlit plotly shap networkx
pip install face_recognition opencv-python
```

---

## 🚀 How to Run

```bash
# Activate environment
conda activate fraud_env

# Run full pipeline
python3 main.py

# Launch web dashboard
streamlit run app.py
# Open: http://localhost:8501

# Register face
python3 -c "from face_auth import register_face; register_face('admin')"

# Run AML analysis
python3 aml_graph.py

# Explain a prediction
python3 explainer.py
```

---

## 📊 Features

- ✅ Real-time transaction monitoring
- ✅ 18 engineered fraud detection features
- ✅ SHAP explainability (why flagged)
- ✅ Face recognition login
- ✅ AML network graph analysis
- ✅ Email fraud alerts
- ✅ 5-page web dashboard
- ✅ Sri Lankan LKR currency
- ✅ 89 Sri Lankan cities
- ✅ 10+ Sri Lankan banks

---

## 👥 Team Members
- M.L.F. Naheetha - 03241078
- M.M.Manasir -
- N.F. Nifsha - 03241198
- A.F. Asra - 03241131
- M.A.M. Hasheed - 03241046
- A.M.Aslaak-03241166
---

## 🏫 Institution
**CG02 — Group 07**
Final Project 2026
