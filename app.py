from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================
# Load Trained Models
# ============================
model_phq = pickle.load(open("phq_model.pkl", "rb"))
model_gad = pickle.load(open("gad_model.pkl", "rb"))

# ============================
# Define Severity Levels
# ============================
def get_depression_severity(score):
    if score >= 20:
        return "Severe Depression"
    elif score >= 15:
        return "Moderately Severe Depression"
    elif score >= 10:
        return "Moderate Depression"
    elif score >= 5:
        return "Mild Depression"
    else:
        return "Minimal Depression"

def get_anxiety_severity(score):
    if score >= 15:
        return "Severe Anxiety"
    elif score >= 10:
        return "Moderate Anxiety"
    elif score >= 5:
        return "Mild Anxiety"
    else:
        return "Minimal Anxiety"

# ============================
# Home Route
# ============================
@app.route("/", methods=["GET"])
def home():
    return "✅ Mental Health Prediction API is running successfully!"

# ============================
# Prediction Route
# ============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data
        data = request.json  

        # Map frontend fields to the correct model features
        mapped_data = {
            'bmi': data.get('BMI', 0),
            'epworth': data.get('Epworth', 0),
            'suicidal': data.get('SuicidalThoughts', 0),
            'depressiveness': data.get('FeelingDepressed', 0),
            'anxiousness': data.get('FeelingAnxious', 0),
            'age': data.get('Age', 25)   # optional default if age was used in training
        }

        input_data = pd.DataFrame([mapped_data])

        # Align columns with model expectations to prevent feature mismatch
        for col in model_phq.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_phq.feature_names_in_]

        # Predict PHQ and GAD Scores
        phq_pred = model_phq.predict(input_data)[0]
        gad_pred = model_gad.predict(input_data)[0]

        # Determine Severity
        depression_severity = get_depression_severity(phq_pred)
        anxiety_severity = get_anxiety_severity(gad_pred)

        # Return JSON response
        return jsonify({
            "predicted_phq_score": round(float(phq_pred), 2),
            "depression_severity": depression_severity,
            "predicted_gad_score": round(float(gad_pred), 2),
            "anxiety_severity": anxiety_severity
        })

    except Exception as e:
        # Return detailed error in JSON for debugging
        return jsonify({"error": str(e)})

# ============================
# Run Flask App
# ============================
if __name__ == "__main__":
    app.run(debug=True)
