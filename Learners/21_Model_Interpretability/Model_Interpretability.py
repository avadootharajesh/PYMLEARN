# Model_Interpretability.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# For LIME
import lime
import lime.lime_tabular

# For SHAP
import shap
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------- LIME Explanation ---------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=['malignant', 'benign'],
    mode='classification'
)

# Explain a test instance
i = 1  # example index
exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=5)
print("LIME Explanation for instance", i)
exp.show_in_notebook(show_table=True)

# Save LIME explanation as html
exp.save_to_file("lime_explanation.html")

# --------- SHAP Explanation ---------
# Use TreeExplainer for tree models
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test)

# Summary plot for global feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# Force plot for a single prediction
shap.initjs()
shap.force_plot(explainer_shap.expected_value[1], shap_values[1][i], X_test[i], feature_names=feature_names, matplotlib=True)
plt.show()
