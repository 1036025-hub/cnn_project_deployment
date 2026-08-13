import logging
import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================================
# Create Flask App
# ==========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
LOG_FOLDER = "logs"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create required folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)


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

model = tf.keras.models.load_model(
    "model/cnn_cifar10_model.keras"
)

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

    print("========== PREDICT ROUTE CALLED ==========")

    # ------------------------------------------------------
    # Check uploaded image
    # ------------------------------------------------------

    if "image" not in request.files:

        print("ERROR: No image in request")

        return render_template(
            "error.html",
            message="No image was uploaded."
        )

    file = request.files["image"]

    print("Image received:", file.filename)


    # ------------------------------------------------------
    # Check filename
    # ------------------------------------------------------

    if file.filename == "":

        print("ERROR: Empty filename")

        return render_template(
            "error.html",
            message="Please select an image before clicking Predict."
        )


    # ------------------------------------------------------
    # Create safe filename
    # ------------------------------------------------------

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    print("Saving image to:", filepath)


    # ------------------------------------------------------
    # Save uploaded image
    # ------------------------------------------------------

    file.save(filepath)

    print("Image saved successfully")


    # ------------------------------------------------------
    # Load image
    # ------------------------------------------------------

    image = Image.open(filepath)

    print("Original image size:", image.size)


    # ------------------------------------------------------
    # Convert to RGB
    # ------------------------------------------------------

    image = image.convert("RGB")


    # ------------------------------------------------------
    # Resize to CIFAR-10 size
    # ------------------------------------------------------

    image = image.resize((32, 32))


    # ------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------

    image = np.array(image)


    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------

    image = image / 255.0


    # ------------------------------------------------------
    # Add batch dimension
    # ------------------------------------------------------

    image = np.expand_dims(image, axis=0)


    print("Image shape:", image.shape)
    print("Image range:", image.min(), image.max())


    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    print("Starting prediction...")

    prediction = model.predict(
        image,
        verbose=0
    )

    print("Prediction probabilities:", prediction)


    # ------------------------------------------------------
    # Get predicted class
    # ------------------------------------------------------

    predicted_index = np.argmax(prediction)

    print("Predicted index:", predicted_index)


    confidence = np.max(prediction) * 100


    predicted_class = class_names[predicted_index]

    print("Predicted class:", predicted_class)
    print("Confidence:", confidence)


    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    logging.info(
        f"Prediction={predicted_class}, "
        f"Confidence={confidence:.2f}%"
    )


    # ------------------------------------------------------
    # Display result
    # ------------------------------------------------------

    return render_template(
        "result.html",
        image=filename,
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )


# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":

    app.run(debug=True)
