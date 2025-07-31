# Flask_Frontend.py
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template_string

app = Flask(__name__)

MODEL_PATH = 'rf_model_scaler.pkl'
FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

model, scaler = None, None

def train_and_save_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)

    model_local = RandomForestClassifier(random_state=42)
    model_local.fit(X_train_scaled, y_train)
    joblib.dump((model_local, scaler_local), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

@app.before_first_request
def load_model():
    global model, scaler
    try:
        model, scaler = joblib.load(MODEL_PATH)
        print("Model loaded from disk.")
    except:
        print("Model not found. Training new model...")
        train_and_save_model()
        model, scaler = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            input_features = []
            for feat in FEATURES:
                val = float(request.form.get(feat))
                input_features.append(val)

            scaled = scaler.transform([input_features])
            pred = model.predict(scaled)[0]
            prediction = 'malignant' if pred == 0 else 'benign'

        except Exception as e:
            error = f"Error processing input: {e}"

    # Simple HTML template with Jinja2 variables
    html = """
    <!doctype html>
    <html>
    <head><title>Breast Cancer Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      label { display: inline-block; width: 180px; margin-top: 8px; }
      input { width: 100px; padding: 4px; }
      .submit-btn { margin-top: 20px; padding: 8px 16px; }
      .result { margin-top: 20px; font-weight: bold; font-size: 1.2em; }
      .error { color: red; margin-top: 20px; }
      form { max-width: 600px; }
    </style>
    </head>
    <body>
      <h2>Breast Cancer Prediction Form</h2>
      <form method="post">
        {% for feat in features %}
          <label for="{{feat}}">{{feat}}</label>
          <input type="text" name="{{feat}}" required><br>
        {% endfor %}
        <button type="submit" class="submit-btn">Predict</button>
      </form>

      {% if prediction %}
        <div class="result">Prediction: <span>{{prediction}}</span></div>
      {% endif %}
      {% if error %}
        <div class="error">{{error}}</div>
      {% endif %}
    </body>
    </html>
    """

    return render_template_string(html, features=FEATURES, prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
