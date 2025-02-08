from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load your TensorFlow model
MODEL_PATH = "new_saved_model"  # Path to your .pd model
model = tf.saved_model.load(MODEL_PATH)
print("Model loaded successfully")

# Preprocess the image
import numpy as np
from PIL import Image


def preprocess_image(image):
    print("Preprocessing image...")

    # Convert the image to RGB (if not already in RGB format)
    img = image.convert('RGB')  # Ensure the image has 3 channels (R, G, B)

    # Resize the image to the desired dimensions (e.g., 32x32)
    img = img.resize((32, 32))  # Resize to 32x32

    # Convert the image to a NumPy array and normalize pixel values to [0, 1]
    img_array = np.array(img) / 255.0  # Normalize pixel values

    # Ensure the dtype is float32
    img_array = img_array.astype(np.float32)

    # Add a batch dimension (to make it compatible with model input)
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 32, 32, 3)

    print(f"Processed image shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array
# Route for the homepage
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route to handle image uploads and predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        print(f"Processing file: {file.filename}")
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        print("Making prediction...")
        predictions = model(processed_image)
        result = predictions.numpy()[0]
        fake_prob, real_prob = result[0], result[1]

        # Return probabilities as percentages
        output = {
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2),
        }

        print(f"Returning probabilities: {output}")
        return jsonify(output)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)