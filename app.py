# app.py — Minimal version with no file saving, no history, pure in-memory processing.
# CORS configured to allow all origins, methods and headers (dev friendly).

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io
import json

app = Flask(_name_)

# Allow all origins, methods and headers (development)
CORS(app, resources={r"/": {"origins": ""}})

# Extra safeguard: ensure responses always include CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization,Accept,Origin,User-Agent,X-Requested-With")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# Respond to OPTIONS preflight explicitly (optional, but helpful)
@app.route('/api/predict', methods=['OPTIONS'])
def predict_options():
    resp = make_response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,User-Agent,X-Requested-With'
    return resp

MODEL_PATH = "model/best_wheat_model.keras"

# ---- Safe model load with minor 'antialias' fixer ----
def safe_load_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        if "Resizing" in str(e) or "antialias" in str(e):
            app.logger.warning("Fixing model JSON (removing antialias) and reloading")
            with open(path, "r") as f:
                config = json.load(f)
            for layer in config.get("config", {}).get("layers", []):
                if layer.get("class_name") == "Resizing" and "antialias" in layer.get("config", {}):
                    layer["config"].pop("antialias", None)
            fixed_path = "fixed_model.keras"
            with open(fixed_path, "w") as f:
                json.dump(config, f)
            return load_model(fixed_path, compile=False)
        raise e

# load model once at startup
model = safe_load_model(MODEL_PATH)

# class labels
class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut", "Yellow Rust"
]

@app.route("/")
def home():
    return "CropCareAI Model Server Running"

# ---------------- API PREDICT (no saving to disk) ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    img_file = request.files["file"]

    try:
        # read bytes
        img_bytes = img_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # preprocess
        img = img.resize((256, 256))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, 0)

        # predict
        preds = model.predict(img_array)[0]
        index = int(np.argmax(preds))

        result = class_names[index]
        confidence = round(float(preds[index]) * 100, 2)

        return jsonify({
            "result": result,
            "confidence": confidence,
            "image_url": None
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if _name_ == "_main_":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)