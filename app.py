from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model("resnet50_forgery_detection.h5")
class_labels = {0: "Authentic", 1: "Forged"}  # Adjust labels based on your dataset

@app.route("/")
def index():
    return render_template("index.html")  # Render the upload form

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    try:
        # Preprocess the uploaded image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Make prediction
        prediction = model.predict(image)
        class_index = int(np.round(prediction[0][0]))
        class_name = class_labels[class_index]

        return jsonify({"prediction": class_name})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
