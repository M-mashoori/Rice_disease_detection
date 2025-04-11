from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("rice_model.h5")



# Class names (update with your dataset classes)
class_names = ["Bacterial Leaf Blight", "Brown Spot", "Healthy"]

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            return render_template("index.html", filename=file.filename, prediction=predicted_class)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
