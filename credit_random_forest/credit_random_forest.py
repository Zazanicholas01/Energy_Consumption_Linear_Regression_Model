import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(url, header=1)

# Rename target columns
data.rename(columns={"default payment next month": "default"}, inplace=True)

# Features Selection
df = pd.DataFrame(data)
X = df.drop("default", axis=1)
y = df['default']

# Train Test Split and SMOTE oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)

y_proba = rf.predict_proba(X_test)[:,1]

# ROC AUC
print("Baseline ROC-AUC:", roc_auc_score(y_test, y_proba))

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2"]
}

# Cross Validation
rs = RandomizedSearchCV(rf, param_grid, n_iter=10, scoring="roc_auc", cv=3, random_state=42, n_jobs=-1)
rs.fit(X_train_res, y_train_res)
best_model = rs.best_estimator_
print("Best params: ", rs.best_params_)

# Final Prediction
final_proba = best_model.predict_proba(X_test)[:,1]

# CSV Saving
submission = pd.DataFrame({
    "ID": X_test.index,
    "Probability": final_proba
})
submission.to_csv('./credit_random_forest/submission.csv', index=False)

# Feature Importance
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
importances.nlargest(15).plot(kind='barh', figsize=(8,6))
plt.show()
