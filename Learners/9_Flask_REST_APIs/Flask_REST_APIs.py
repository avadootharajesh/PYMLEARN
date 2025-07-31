# Flask_REST_APIs.py
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

# --------- Part 1: Train and Save Model ---------

def train_and_save_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save scaler and model together
    joblib.dump((model, scaler), 'rf_model_scaler.pkl')
    print("Model and scaler saved to 'rf_model_scaler.pkl'.")

# --------- Part 2: Flask API ---------

app = Flask(__name__)

# Load model and scaler
model, scaler = None, None

@app.before_first_request
def load_model():
    global model, scaler
    model, scaler = joblib.load('rf_model_scaler.pkl')
    print("Model and scaler loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    input_data = request.get_json()

    # Expecting input data as dictionary of feature_name: value
    features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]

    try:
        # Extract feature values in correct order
        input_features = [float(input_data[feat]) for feat in features]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except ValueError:
        return jsonify({"error": "Feature values must be numeric"}), 400

    # Scale and predict
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]

    # Return result
    result = {
        "prediction": int(prediction),
        "class_name": 'malignant' if prediction == 0 else 'benign',
        "probability": prediction_proba
    }
    return jsonify(result), 200

# --------- Main ---------

if __name__ == '__main__':
    import sys
    if 'train' in sys.argv:
        train_and_save_model()
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)


# python flask_ml_api_demo.py train
# python flask_ml_api_demo.py

# send to api request
{
  "mean radius": 14.0,
  "mean texture": 20.0,
  "mean perimeter": 90.0,
  "mean area": 600.0,
  "mean smoothness": 0.1,
  "mean compactness": 0.1,
  "mean concavity": 0.1,
  "mean concave points": 0.05,
  "mean symmetry": 0.2,
  "mean fractal dimension": 0.06,
  "radius error": 0.5,
  "texture error": 1.0,
  "perimeter error": 3.0,
  "area error": 40.0,
  "smoothness error": 0.01,
  "compactness error": 0.02,
  "concavity error": 0.02,
  "concave points error": 0.01,
  "symmetry error": 0.02,
  "fractal dimension error": 0.003,
  "worst radius": 15.0,
  "worst texture": 25.0,
  "worst perimeter": 100.0,
  "worst area": 700.0,
  "worst smoothness": 0.15,
  "worst compactness": 0.2,
  "worst concavity": 0.3,
  "worst concave points": 0.1,
  "worst symmetry": 0.3,
  "worst fractal dimension": 0.07
}
