from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- 1. Load the Model Package ---

try:
    package = joblib.load('heart_attack_model.pkl')
    model = package['model']
    scaler = package['scaler']
    # REMOVED the 'selector' line because we don't need it!
    required_features = package['features'] # List of the 20 column names

    # --- DEBUG: Sanity Check ---
    print("-" * 50)
    print(f"Model loaded successfully.")
    print(f"The Scaler expects: {scaler.n_features_in_} columns.")
    print(f"The Model expects:  {len(required_features)} columns.")
    
    if scaler.n_features_in_ != len(required_features):
        print("❌ CRITICAL WARNING: Mismatch detected!")
        print("Your Scaler expects more columns than the Model.")
    else:
        print("✅ Success: Scaler and Model are synced.")
    print("-" * 50)

except Exception as e:
    print(f"❌ Error loading model file: {e}")

except Exception as e:
    print(f"❌ Error loading model file: {e}")
    # We continue so the app doesn't crash immediately, but predictions will fail.

# --- 2. Define the Home Page ---
@app.route('/')
def home():
    return render_template('index.html')

# --- 3. Define the Prediction API ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json # Get data from the frontend
        
        # A. Convert JSON to DataFrame
        input_data = pd.DataFrame([data])
        
        # B. Validation: Ensure we have the EXACT 20 columns the model needs
        try:
            # This reorders the input to match the training order perfectly
            ordered_data = input_data[required_features]
        except KeyError as e:
            return jsonify({'error': f"Missing required feature: {str(e)}"}), 400
        
        # C. Scale the Data
        # If your pickle is updated, this will now work perfectly (20 inputs -> 20 scaled outputs)
        scaled_data = scaler.transform(ordered_data)
        
        # D. Predict
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        # E. Return Result
        result = {
            'prediction': int(prediction),
            'risk_score': float(probability),
            'message': "High Risk" if prediction == 1 else "Low Risk"
        }
        
        return jsonify(result)

    except Exception as e:
        # This catches "Shape mismatch" errors if the pickle wasn't updated
        return jsonify({'error': f"Prediction Error: {str(e)}"}), 500

if __name__ == '__main__':
    # We use port 5001 to avoid the 'Address already in use' error
    app.run(debug=True, port=5001)