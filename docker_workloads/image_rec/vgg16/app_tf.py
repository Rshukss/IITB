from flask import Flask, request, jsonify
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO


app = Flask(__name__)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('GPU is available!')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print('GPU is not available')

# Initialize VGG16 model with pre-trained weights
vgg16_model = tf.keras.applications.VGG16(weights="imagenet")

# Initialize the input tensor shape for VGG16 model
input_shape = vgg16_model.input_shape[1:3]

@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL of the image from the JSON body of the request
    url = request.json['url']

    # Download the image and convert it to a PIL Image object
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Resize the image to match the input shape of VGG16 model
    img = img.resize(input_shape)

    # Convert PIL Image to numpy array and preprocess for VGG16 model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)

    # Make a prediction using the VGG16 model
    with tf.device('/device:GPU:0'):
        output = vgg16_model.predict(img_array)

    # Get the predicted class and score
    class_id = output[0].argmax()
    score = output[0][class_id]
    category_name = tf.keras.applications.vgg16.decode_predictions(output, top=1)[0][0][1]

    # Return the predicted class and score as a JSON response
    response = {
        'class': category_name,
        'score': float(score)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)
