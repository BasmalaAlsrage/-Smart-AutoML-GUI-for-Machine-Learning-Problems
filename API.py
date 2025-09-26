import joblib
from flask import Flask, request, jsonify
import os

# Helper Functions 
def load_model(model_filename):
    """Load model dynamically by filename"""
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file '{model_filename}' not found.")
    return joblib.load(model_filename)

def scale_features(x, scaler_filename="scaler.pkl"):
    """Scale features if scaler exists"""
    if os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
        x_scaled = scaler.transform([x])
        return x_scaled
    else:
        return [x]

# Flask App 
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json


        model_filename = data.get("model_filename", "best_model.pkl")  
        model = load_model(model_filename)

        
        x = data.get("features")
        if x is None:
            return jsonify({"error": "Missing features in request"}), 400

       
        x_scaled = scale_features(x)

        # prediction
        prediction = model.predict(x_scaled)
        return jsonify({
            "model": model_filename,
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)