# Hyperparameter_Tuning.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load Dataset
# -------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -------------------------
# Train/Test Split + Scaling
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Base Model for Comparison
# -------------------------
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train_scaled, y_train)
y_pred_base = baseline.predict(X_test_scaled)
print("\n--- Baseline Model ---")
print("Accuracy:", accuracy_score(y_test, y_pred_base))

# -------------------------
# Grid Search
# -------------------------
grid_params = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid=grid_params,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("\n--- Grid Search Results ---")
print("Best Params:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Evaluate on test set
y_pred_grid = grid_search.predict(X_test_scaled)
print("Test Accuracy (Grid Search):", accuracy_score(y_test, y_pred_grid))

# -------------------------
# Randomized Search
# -------------------------
random_params = {
    'n_estimators': np.arange(50, 201, 10),
    'max_depth': [None] + list(np.arange(5, 31, 5)),
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                   param_distributions=random_params,
                                   n_iter=20,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42)
random_search.fit(X_train_scaled, y_train)

print("\n--- Randomized Search Results ---")
print("Best Params:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

# Evaluate on test set
y_pred_rand = random_search.predict(X_test_scaled)
print("Test Accuracy (Randomized Search):", accuracy_score(y_test, y_pred_rand))

# -------------------------
# Confusion Matrix Comparison
# -------------------------
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(y_test, y_pred_grid, "Confusion Matrix - GridSearchCV")
plot_conf_matrix(y_test, y_pred_rand, "Confusion Matrix - RandomizedSearchCV")
