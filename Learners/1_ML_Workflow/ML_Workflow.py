# ML_Workflow.py


# 1. Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 2. Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 6. Save model to disk
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 7. Load model back
with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 8. Predict on new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(new_sample)
print(f"Predicted Iris class for sample {new_sample}: {prediction[0]}")
