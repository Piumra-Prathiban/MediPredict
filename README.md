# MediPredict 🩺

MediPredict is a web-based machine learning application designed to assess the risk of a heart attack based on a patient's medical history and lifestyle factors. By leveraging a Logistic Regression model trained on real-world medical data, it provides real-time risk probabilities to assist in early health screening.

![alt text](https://github.com/user-attachments/assets/8dbc0f02-00dd-4c1d-aa29-d46ae0e68ff2)

## Disclaimer

This project is intended for educational purposes only and is not designed for medical use or clinical decision-making. Use of this information is entirely at your own risk, and the authors are not responsible for any outcomes resulting from its use.

We are not affiliated with any organizations or entities associated with the dataset. The data used in this project was obtained from Kaggle and remains the property of its original creators and respective owners.

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
├── PreProcess.ipynb # Data analysis and model training notebook
|___ Requirements.txt #   Needed Dependencies to work the app
├── templates/
│   └── index.html      # Web application frontend
└── venv/               # Python virtual environment

## Project Deployment Method

1. Clone this repository to your local machine using Git.

2. Open the project folder in your preferred code editor.

3. Create and activate a Python virtual environment (venv) within the project directory.

4. Install the required dependencies using `pip` (pip install -r requirements.txt).

5. Run the Jupyter Notebook (`.ipynb`) using the activated virtual environment.  
   This process will train the model and generate the heart attack prediction model file in the project’s root directory.

6. After the model file is created, open `app.py` and run it using the same virtual environment.

7. The terminal will display the local server address and port where the web application is hosted.

8. Open the provided URL in a web browser, enter the requested input data.

9. click **Predict** to obtain the heart attack risk assessment.


