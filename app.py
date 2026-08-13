import logging
# ==========================================================
# Import Required Libraries
# ==========================================================

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================================
# Create Flask App
# ==========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================================
# Configure Logging
# ==========================================================

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ==========================================================
# Load CNN Model
# ==========================================================

print("Loading CNN model...")

model = tf.keras.models.load_model("model/cnn_cifar10_model.keras")

print("Model loaded successfully!")

# ==========================================================
# CIFAR-10 Class Names
# ==========================================================

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================================
# About Page
# ==========================================================

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================================
# Predict Image
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
    "error.html",
    message="No image was uploaded."
)

    file = request.files["image"]

    if file.filename == "":
        return render_template(
    "error.html",
    message="Please select an image before clicking Predict."
)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Load image
    image = Image.open(filepath)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to CIFAR-10 size
    image = image.resize((32,32))

    # Convert to NumPy
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_class = class_names[predicted_index]
    logging.info(
    f"Prediction={predicted_class}, Confidence={confidence:.2f}%"
)

    return render_template(

        "result.html",

        image=file.filename,

        prediction=predicted_class,

        confidence=round(confidence,2)

    )

# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)

