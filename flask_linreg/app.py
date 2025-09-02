from flask import Flask, request, jsonify, render_template
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

ARTIFACT_DIR = Path("flask_linreg/artifacts")
MODEL_PATH = ARTIFACT_DIR / "linear_model.pkl"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required = feature_columns["numeric_cols"] + feature_columns["categoric_cols"] + ["hour"]
        missing = [col for col in required if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        df = pd.DataFrame([data])
        df["deltaT"] = df["setpoint_temp"] - df["ambient_temp"]
        df["flow_deltaT"] = df["flow_rate"] * df["deltaT"]
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        pred = model.predict(df)[0]
        return jsonify({"predicted_energy_consumption": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)