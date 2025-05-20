from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
# Load your trained model
model = tf.keras.models.load_model('mnist_cnn_augmented.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()['image']
    header, encoded = data.split(',', 1)
    img_data = base64.b64decode(encoded)
    # img = Image.open(io.BytesIO(img_data)).convert('L').resize((28,28))
    # arr = np.array(img) / 255.0
    # arr = arr.reshape(1,28,28,1)

    # new, smarter preprocessing:
    img = Image.open(io.BytesIO(img_data)).convert('L') \
            .resize((28,28), resample=Image.LANCZOS)
    arr = np.array(img).astype('float32') / 255.0

    # Binarize: make everything pure 0 or 1
    arr = (arr > 0.5).astype('float32')

    # If the image’s “background” is dark (mean < .5), invert it
    # (we want background=0, strokes=1)
    if np.mean(arr) > 0.5:
        arr = 1.0 - arr

    # Ready for model
    arr = arr.reshape(1,28,28,1)
    pred = model.predict(arr).argmax(axis=1)[0]
    return jsonify({'digit': int(pred)})

if __name__ == '__main__':
    app.run(debug=True)