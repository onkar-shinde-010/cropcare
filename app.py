from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)   # allow all origins

MODEL_PATH = "model/best_wheat_model.keras"

# ---- Safe model load ----
def safe_load_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        if "Resizing" in str(e) or "antialias" in str(e):
            with open(path, "r") as f:
                config = json.load(f)

            for layer in config.get("config", {}).get("layers", []):
                if layer.get("class_name") == "Resizing":
                    layer["config"].pop("antialias", None)

            fixed_path = "fixed_model.keras"
            with open(fixed_path, "w") as f:
                json.dump(config, f)

            return load_model(fixed_path, compile=False)

        raise e

# Load model
model = safe_load_model(MODEL_PATH)

class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut", "Yellow Rust"
]

@app.route("/")
def home():
    return "CropCareAI Model Server Running"

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    img_file = request.files["file"]

    try:
        img_bytes = img_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img = img.resize((256, 256))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, 0)

        preds = model.predict(img_array)[0]
        index = int(np.argmax(preds))

        return jsonify({
            "result": class_names[index],
            "confidence": round(float(preds[index]) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
