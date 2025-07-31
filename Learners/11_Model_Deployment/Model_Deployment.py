# Model_Deployment.py
import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

model, scaler = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


#  save model
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

joblib.dump((model, scaler), 'model.joblib')
print("Model and scaler saved!")


# docker file
# FROM python:3.9-slim

# WORKDIR /app

# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# COPY app.py model.joblib ./

# EXPOSE 5000

# CMD ["python", "app.py"]
# requirements.txt
# nginx
# flask
# scikit-learn
# joblib
# numpy
# How to deploy?
# Train and save model:

# python train_save_model.py
# Build Docker image:

# docker build -t ml-flask-api .
# Run container:

# docker run -p 5000:5000 ml-flask-api
# Test API with curl or Postman:

# curl -X POST -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' http://localhost:5000/pre
