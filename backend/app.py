"""
Mulberry Leaf Disease Detection - Flask API
"""

import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import predict, load_model, CLASSES, is_leaf_image

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

UPLOAD_FOLDER      = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "jfif"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model...")
model = load_model()
print("Model ready.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health():
    import os
    model_exists = os.path.exists("mulberry_model.h5")
    model_size_mb = round(os.path.getsize("mulberry_model.h5") / (1024*1024), 1) if model_exists else 0
    return jsonify({
        "status": "ok",
        "classes": CLASSES,
        "model_trained": model_exists,
        "model_size_mb": model_size_mb,
        "note": "Model is trained" if model_exists else "WARNING: No trained model found"
    })


@app.route("/predict", methods=["POST"])
def predict_disease():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    if ext == "jfif":
        ext = "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        result = predict(filepath, model)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
