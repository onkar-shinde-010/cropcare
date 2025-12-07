from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import os, json
from werkzeug.utils import secure_filename

app = Flask(_name_)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = r"model/best_wheat_model.keras"


# --------- FIX MODEL DESERIALIZATION ISSUE ----------
def safe_load_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        if "Resizing" in str(e) or "antialias" in str(e):
            print("⚠ Model contains unsupported 'antialias' argument. Fixing...")

            # Read model architecture
            with open(path, "r") as f:
                config = json.load(f)

            # Remove 'antialias' from resizing layer if found
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


# -------- Load Model Safely --------
model = safe_load_model(MODEL_PATH)


class_names = [
    "Aphid", "Brown Rust", "Healthy", "Leaf Blight",
    "Mildew", "Mite", "Septoria", "Smut", "Yellow Rust"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
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


if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)