from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
model = load_model('model.h5')  

class_labels = ['melanoma', 'nevus']

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file)
        img = img.resize((150, 150))  
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0  # Normalize pixel values to the range [0, 1]

        predictions = model.predict(img)
        predicted_class = class_labels[int(round(predictions[0][0]))]

        return jsonify({'predicted_class': predicted_class, 'class_probabilities': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
