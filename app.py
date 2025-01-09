import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
try:
    vgg_model = load_model("vgg.h5")
    vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Explicitly compile
    mob_model = load_model("mob.h5")
    mob_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Explicitly compile
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Create an "index.html" file for the interface

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Use JPG, JPEG, or PNG."}), 400

    try:
        # Process the uploaded image
        img = load_img(file.stream, target_size=(224, 224))
        img_array = img_to_array(img)
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
        return jsonify({"error": f"Unable to process the image. Details: {str(e)}"}), 500

if __name__ == "__main__":
    # Ensure models exist before running the app
    if not os.path.exists("vgg.h5") or not os.path.exists("mob.h5"):
        print("Model files not found. Ensure 'vgg.h5' and 'mob.h5' are in the same directory.")
    else:
        app.run(debug=False)  # Disable debug mode in production