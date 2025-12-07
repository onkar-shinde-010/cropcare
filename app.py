# app.py (replace or update your current file)
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import os, json, datetime
from werkzeug.utils import secure_filename

app = Flask(_name_, static_folder="static", template_folder="templates")

# Enable CORS for all routes (you can restrict origins if needed)
CORS(app)  # <- this adds Access-Control-Allow-Origin: *

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = r"model/best_wheat_model.keras"

# safe_load_model (your existing function)
def safe_load_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        if "Resizing" in str(e) or "antialias" in str(e):
            print("⚠ Model contains unsupported 'antialias' argument. Fixing...")
            with open(path, "r") as f:
                config = json.load(f)
            for layer in config["config"]["layers"]:
                if layer["class_name"] == "Resizing" and "antialias" in layer["config"]:
                    layer["config"].pop("antialias")
            fixed_path = "fixed_model.keras"
            with open(fixed_path, "w") as f:
                json.dump(config, f)
            print("✔ Model fixed. Reloading...")
            return load_model(fixed_path, compile=False)
        else:
            raise ValueError(str(e))

model = safe_load_model(MODEL_PATH)

class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut", "Yellow Rust"
]

@app.route("/")
def home():
    return render_template("index.html")

# Keep existing form endpoint (optional)
@app.route("/predict", methods=["POST"])
def predict_form():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    try:
        img = load_img(filepath, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array)[0]
        index = prediction.argmax()
        return render_template(
            "index.html",
            file_path=filepath,
            result=class_names[index],
            confidence=round(float(prediction[index]) * 100, 2)
        )
    except Exception as e:
        return render_template("index.html", error=f"Prediction Error: {str(e)}")

# JSON API for programmatic clients (Flutter web, JS)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    filename = secure_filename(file.filename)
    # make filename unique with timestamp
    filename = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f_") + filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    try:
        img = load_img(filepath, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array)[0]
        index = int(prediction.argmax())
        result = class_names[index]
        confidence = round(float(prediction[index]) * 100, 2)
        # return JSON and image path for client to show
        return jsonify({
            "result": result,
            "confidence": confidence,
            "filename": filename,
            "image_url": f"/{app.config['UPLOAD_FOLDER']}/{filename}"
        })
    except Exception as e:
        return jsonify({"error": "Prediction Error", "details": str(e)}), 500

# history endpoint (if you already have one)
@app.route("/api/history", methods=["GET"])
def api_history():
    # implement your history load (or return empty list)
    hist_path = "history.json"
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            try:
                return jsonify(json.load(f))
            except Exception:
                return jsonify([])
    return jsonify([])

# Serve uploaded files
@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)