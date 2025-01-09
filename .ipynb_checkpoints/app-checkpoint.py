import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
vgg_model = load_model("vgg.h5")
mob_model = load_model("mob.h5")

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Create an "index.html" file for the interface

# Route for uploading and predicting
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Process the uploaded image
    try:
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Get predictions from both models
        vgg_pred = vgg_model.predict(img_array)
        mob_pred = mob_model.predict(img_array)

        # Average predictions
        avg_pred = (vgg_pred + mob_pred) / 2
        pred_class = class_labels[np.argmax(avg_pred)]

        # Disease information
        disease_info = {
            "glioma": "Glioma is a type of tumor that occurs in the brain and spinal cord.",
            "meningioma": "Meningioma is a tumor that forms on membranes covering the brain and spinal cord just inside the skull.",
            "notumor": "The image shows no evidence of a tumor.",
            "pituitary": "Pituitary tumors are abnormal growths that develop in the pituitary gland."
        }

        return jsonify({
            "prediction": pred_class,
            "description": disease_info[pred_class]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
