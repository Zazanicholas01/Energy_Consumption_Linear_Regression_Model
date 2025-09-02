import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import joblib

# Set random seed
np.random.seed(42)

# Create synthetic dataset
N = 20000
hour = np.random.randint(0, 24, N)
day = np.random.randint(1, 7, N)
is_weekend = (day >= 5).astype(int)

ambient_temp = np.random.normal(loc=20, scale=8, size=N)
ambient_humidity = np.clip(np.random.normal(loc=50, scale=15, size=N), 5, 95)
flow_rate = np.abs(np.random.normal(0.5, 0.2, N))
pressure = np.random.normal(1.5e5, 2e4, N)
pump_speed = np.random.normal(1500, 400, N)
setpoint_temp = ambient_temp + np.random.normal(5, 2, N)
vibration = np.abs(np.random.normal(0.02, 0.01, N))

# Coefficients for the linear model decided based on domain knowledge

alpha0, alpha1, alpha2, alpha3, alpha4 = 10.0, 120.0, 0.005, 1e-5, 3.0

# Energy consumption model

deltaT = setpoint_temp - ambient_temp
seasonal_hour = np.sin(2*np.pi * hour / 24)

noise = np.random.normal(0, 2 + 0.5*flow_rate, N)

y = (alpha0 + 
     alpha1*deltaT + 
     alpha2*pump_speed +
     alpha3 * (pressure - 1e5) +
     alpha4 * seasonal_hour +
     5.0 * vibration +
     2.0 * is_weekend +
     noise)

# Create DataFrame and split into train/test sets

df = pd.DataFrame({
    "ambient_temp": ambient_temp,
    "ambient_humidity": ambient_humidity,
    "flow_rate": flow_rate,
    "pressure": pressure,
    "pump_speed": pump_speed,
    "setpoint_temp": setpoint_temp,
    "vibration": vibration,
    "hour": hour,
    "day": day,
    "is_weekend": is_weekend,
    "energy_consumption": y
})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV files

train_df.to_csv("flask_linreg/data/train_data.csv", index=False)
test_df.to_csv("flask_linreg/data/test_data.csv", index=False)

numerical_cols = [column for column in train_df.columns][:7] # First 7 columns are numerical
categorical_cols = ["day"]

# Function to add derived features like deltaT, flow*deltaT, hour_sin, hour_cos

def add_derived(X_df):
    X = X_df.copy()
    X["deltaT"] = X["setpoint_temp"] - X["ambient_temp"]
    X["flow_deltaT"] = X["flow_rate"] * X["deltaT"]
    X["hour_sin"] = np.sin(2 * np.pi * X_df["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X_df["hour"] / 24)
    return X

# Transformer to add derived features

derived_transformer = FunctionTransformer(lambda x: add_derived(pd.DataFrame(x, columns=train_df.columns)), validate=False)

# Build pipeline with preprocessing, polynomial features, and ridge regression

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols + ["deltaT", "flow_deltaT", "hour_sin", "hour_cos"]),
    ("day_ohe", OneHotEncoder(handle_unknown="ignore"), ["day"])
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ("reg", Ridge())
])

# Hyperparameter tuning with GridSearchCV

param_grid = {
    "reg__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
}

# Use RepeatedKFold for cross-validation to get more robust estimates of model performance

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)

# Load data and fit the model

train_df = add_derived(pd.read_csv("flask_linreg/data/train_data.csv"))
test_df = add_derived(pd.read_csv("flask_linreg/data/test_data.csv"))

all_input_cols = train_df.columns.drop("energy_consumption")

numerical_cols += ["deltaT", "flow_deltaT", "hour_sin", "hour_cos"]

X_train = train_df[all_input_cols]
y_train = train_df["energy_consumption"]

grid.fit(X_train, y_train)

# Evaluate the best model on the test set

best_model = grid.best_estimator_
print(f"Best model parameters: {grid.best_params_}")

X_test = test_df[all_input_cols]
y_test = test_df["energy_consumption"]
y_pred = best_model.predict(X_test)

# Calculate metrics

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R^2: {r2:.3f}")
print(f"Test MAE: {mae:.3f}")

# Save the model and feature columns

joblib.dump(best_model, "flask_linreg/artifacts/linear_model.pkl")
feature_columns = {"numeric_cols": numerical_cols, "categoric_cols": categorical_cols}
with open("flask_linreg/artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

print("Model and feature columns saved to artifacts/")

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5))
# plot RMSE
train_rmse = np.sqrt(-train_scores.mean(axis=1))
test_rmse = np.sqrt(-test_scores.mean(axis=1))
plt.plot(train_sizes, train_rmse, label='Train RMSE')
plt.plot(train_sizes, test_rmse, label='Val RMSE')
plt.xlabel('Train size')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('flask_linreg/artifacts/learning_curve.png')
