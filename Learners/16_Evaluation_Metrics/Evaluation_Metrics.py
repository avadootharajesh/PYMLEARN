# Evaluation_Metrics.py
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
import numpy as np

# Sample true and predicted labels for classification
y_true_cls = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred_cls = [0, 1, 0, 0, 1, 0, 0, 1]
y_proba_cls = [0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.35, 0.7]  # Probabilities for positive class

print("Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_true_cls, y_pred_cls):.3f}")
print(f"Precision: {precision_score(y_true_cls, y_pred_cls):.3f}")
print(f"Recall: {recall_score(y_true_cls, y_pred_cls):.3f}")
print(f"F1 Score: {f1_score(y_true_cls, y_pred_cls):.3f}")
print(f"ROC AUC: {roc_auc_score(y_true_cls, y_proba_cls):.3f}")

# Sample true and predicted values for regression
y_true_reg = [3.0, -0.5, 2.0, 7.0]
y_pred_reg = [2.5, 0.0, 2.1, 7.8]

print("\nRegression Metrics:")
print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.3f}")
print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.3f}")
print(f"RMSE: {mean_squared_error(y_true_reg, y_pred_reg, squared=False):.3f}")
print(f"R2 Score: {r2_score(y_true_reg, y_pred_reg):.3f}")
