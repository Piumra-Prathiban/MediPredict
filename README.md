# MediPredict 🩺

MediPredict is a web-based machine learning application designed to assess the risk of a heart attack based on a patient's medical history and lifestyle factors. By leveraging a Logistic Regression model trained on real-world medical data, it provides real-time risk probabilities to assist in early health screening.

## 🚀 Features

- **High Precision ML Model:** Uses a tuned Logistic Regression model optimized for **Recall** to ensure at-risk patients are not missed.
- **Robust Data Pipeline:** Implements advanced preprocessing including outlier capping, MinMaxScaler, and SMOTE to handle class imbalance.
- **Interactive Web UI:** A clean, responsive dashboard built with Flask and JavaScript for instant risk assessment.
- **Explainable Results:** Provides both a binary prediction (High/Low Risk) and a specific risk probability percentage.

## 🛠️ Technology Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Imbalanced-learn, Joblib
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

## 📁 Project Structure

```text
MediPredict/
├── app.py              # Flask backend server
├── heart_attack_model.pkl # Serialized model, scaler, and feature list
├── PreProcess.ipynb    # Data analysis and model training notebook
├── templates/
│   └── index.html      # Web application frontend
└── venv/               # Python virtual environment